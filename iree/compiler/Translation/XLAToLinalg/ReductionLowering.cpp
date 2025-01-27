// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- ReductionLowering.cpp ------------------------------------*-C++//-*-===//
//
// Lower reduction dispatch regions to Linalg.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/map_xla_to_scalar_op.h"

namespace mlir {
namespace iree_compiler {

namespace {

/// Pass to lower the reduction dispatch functions to Linalg.
struct HLOReductionToLinalgPass : public ModulePass<HLOReductionToLinalgPass> {
  void runOnModule() override;
};

/// Returns an ArrayAttr that contains `nLoops` attributes. All the attributes
/// are "parallel" except the `reductionDim`-th element.
// TODO(hanchung): Use helpers in StructuredOpsUtils.h instead of hardcoded
// strings once the build system is set up.
ArrayAttr getInnermostReductionIterAttrs(Builder b, unsigned nLoops) {
  SmallVector<Attribute, 3> attrs(nLoops, b.getStringAttr("parallel"));
  attrs.back() = b.getStringAttr("reduction");
  return b.getArrayAttr(attrs);
}

/// Base class for legalization of operations within the reduction apply
/// function (and the function itself).
template <typename OpTy>
class ApplyFunctionConversion : public OpConversionPattern<OpTy> {
 public:
  ApplyFunctionConversion(MLIRContext *context, TypeConverter &converter,
                          PatternBenefit benefit = 1)
      : OpConversionPattern<OpTy>(context, benefit), converter(converter) {}

 protected:
  TypeConverter &converter;
};

/// The apply function has a signature (lhs, rhs) -> output, all of the same
/// tensor type t. This is converted to a function with the same signature but
/// with element types. E.g., "(tensor<f32>, tensor<f32>) -> tensor<f32>" will
/// be converted to "(f32, f32) -> f32".
struct ReductionApplyFnConversion final : ApplyFunctionConversion<FuncOp> {
  using ApplyFunctionConversion<FuncOp>::ApplyFunctionConversion;

  PatternMatchResult matchAndRewrite(
      FuncOp funcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override;
};

}  // namespace

// Returns a permutation AffineMap that puts `reductionDim` to the last. The
// order of the first (`rank` - 1) can be unsorted. E.g., if `rank` is 4 and
// `reductionDim` is 1, then "(d0, d1, d2, d3) -> (d0, d3, d2, d1)" can be
// returned.
static AffineMap getTransposeMapForReduction(OpBuilder &builder, int rank,
                                             int reductionDim) {
  SmallVector<unsigned, 4> permutation;
  for (int i = 0; i < rank; ++i) permutation.push_back(i);
  std::swap(permutation[reductionDim], permutation[rank - 1]);
  return AffineMap::getPermutationMap(permutation, builder.getContext());
}

//===----------------------------------------------------------------------===//
// Reduction entry function body
//===----------------------------------------------------------------------===//

/// Adds a body with linalg ops for the reduction entry function `fn`. `fn` is
/// assumed to be empty. The dimension to reduce must be set as an attribute on
/// `fn`.
static LogicalResult addReductionEntryFnBody(OpBuilder &builder, FuncOp fn,
                                             FuncOp applyFn) {
  if (fn.getNumArguments() != 3) return failure();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(fn.addEntryBlock());
  auto src = fn.getArgument(0);
  auto initVal = fn.getArgument(1);
  auto dst = fn.getArgument(2);

  auto srcArgType = src.getType().template cast<ShapedType>();
  unsigned nInputRank = srcArgType.getRank();
  if (!nInputRank) return failure();

  auto loc = fn.getLoc();
  int reductionDim =
      fn.getAttrOfType<IntegerAttr>("iree.executable.reduction.dimension")
          .getInt();

  // Prepare indexing maps for linalg generic op. The elements are for src,
  // initial value and dst, respectively.
  // Transpose `src` to make the reduction loop be the innermost, because it's
  // easier to fully utilize processors.
  SmallVector<Attribute, 3> indexingMaps;
  indexingMaps.emplace_back(AffineMapAttr::get(
      getTransposeMapForReduction(builder, nInputRank, reductionDim)));
  indexingMaps.emplace_back(AffineMapAttr::get(AffineMap::get(
      nInputRank, /*symbolCount=*/0, {builder.getAffineConstantExpr(0)})));
  // Since the reduction loop now is the innermost, the indexing map of `dst`
  // should drop the latest dimension, e.g., (d0, d1, d2) -> (d0, d1).
  SmallVector<AffineExpr, 4> exprs;
  for (int i = 0; i < nInputRank - 1; ++i)
    exprs.push_back(builder.getAffineDimExpr(i));
  if (exprs.empty()) exprs.push_back(builder.getAffineConstantExpr(0));
  indexingMaps.emplace_back(
      AffineMapAttr::get(AffineMap::get(nInputRank, /*symbolCount=*/0, exprs)));

  SmallVector<Type, 2> resultTypes = {};
  SmallVector<Value, 2> linalgOpArgs = {src, initVal, dst};
  auto linalgOp = builder.create<linalg::IndexedGenericOp>(
      loc, resultTypes, linalgOpArgs,
      builder.getI64IntegerAttr(2),  // args_in
      builder.getI64IntegerAttr(1),  // args_out
      builder.getArrayAttr(indexingMaps),
      getInnermostReductionIterAttrs(builder, nInputRank),
      /*doc=*/nullptr, /*fun=*/nullptr, /*library_call=*/nullptr);

  // Add a block to the region.
  auto *region = &linalgOp.region();
  auto *block = builder.createBlock(region, region->end());
  Type indexType = builder.getIndexType();
  Type elemType = srcArgType.getElementType();
  for (int i = 0; i < nInputRank; ++i) block->addArguments(indexType);
  for (int i = 0; i < linalgOp.getNumInputsAndOutputs(); ++i)
    block->addArguments(elemType);
  auto numArgs = block->getNumArguments();
  auto blockSrcArg = block->getArgument(numArgs - 3);
  auto blockInitArg = block->getArgument(numArgs - 2);
  auto blockDstArg = block->getArgument(numArgs - 1);
  auto zero = builder.create<ConstantOp>(loc, indexType,
                                         builder.getIntegerAttr(indexType, 0));
  // The reduction dimension is the innermost loop now, so compare the innermost
  // index to zero.
  auto cond = builder.create<CmpIOp>(loc, CmpIPredicate::eq,
                                     block->getArgument(nInputRank - 1),
                                     zero.getResult());
  auto lhs = builder.create<SelectOp>(loc, cond, blockInitArg, blockDstArg);
  auto rhs = blockSrcArg;

  // Inline the function call into the region.
  // TODO(hanchung): Use the MLIR inline pass. This requires to implement Linalg
  // call interface.
  BlockAndValueMapping mapper;
  assert(applyFn.getNumArguments() == 2);
  mapper.map(applyFn.getArgument(0), lhs);
  mapper.map(applyFn.getArgument(1), rhs);
  assert(mlir::has_single_element(applyFn.getBlocks()) &&
         "apply function with multiple blocks is not support");
  applyFn.getBody().cloneInto(block->getParent(), mapper);

  Block &newBlock = *block->getParent()->getBlocks().rbegin();
  block->getOperations().splice(block->end(), newBlock.getOperations());
  newBlock.erase();

  // Replace the terminator with linalg::YieldOp is required by linalg.generic
  // op.
  Operation *terminator = block->getTerminator();
  builder.create<linalg::YieldOp>(loc, terminator->getOperands());
  terminator->erase();

  builder.setInsertionPointAfter(linalgOp);
  builder.create<ReturnOp>(loc);

  return success();
}

//===----------------------------------------------------------------------===//
// Apply function conversion
//===----------------------------------------------------------------------===//

PatternMatchResult ReductionApplyFnConversion::matchAndRewrite(
    FuncOp funcOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto fnType = funcOp.getType();
  if (fnType.getNumInputs() != 2 || fnType.getNumResults() != 1)
    return matchFailure();
  if (fnType.getInput(0) != fnType.getInput(1) ||
      fnType.getInput(0) != fnType.getResult(0))
    return matchFailure();

  TypeConverter::SignatureConversion signatureConverter(fnType.getNumInputs());
  auto convertedType = converter.convertType(fnType.getInput(0));
  if (!convertedType) return matchFailure();
  signatureConverter.addInputs(0, convertedType);
  signatureConverter.addInputs(1, convertedType);
  auto newFn = rewriter.cloneWithoutRegions(funcOp);
  rewriter.inlineRegionBefore(funcOp.getBody(), newFn.getBody(), newFn.end());
  newFn.setType(rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                                         convertedType));
  rewriter.applySignatureConversion(&newFn.getBody(), signatureConverter);
  rewriter.eraseOp(funcOp);
  return matchSuccess();
}

//===----------------------------------------------------------------------===//
// ReductionOp
//===----------------------------------------------------------------------===//

/// Operations within the apply function need to be converted to standard ops.
template <typename OpTy>
struct ReductionOpConversion final : public ApplyFunctionConversion<OpTy> {
  using ApplyFunctionConversion<OpTy>::ApplyFunctionConversion;

  PatternMatchResult matchAndRewrite(
      OpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (operands.size() != 2) return this->matchFailure();
    SmallVector<Type, 1> resultElemTypes = {operands[0].getType()};
    Value opResult = xla_lhlo::MapXlaOpToStdScalarOp<OpTy>(op, resultElemTypes,
                                                           operands, &rewriter);
    rewriter.replaceOp(op, opResult);
    return this->matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

// Appends all the apply functions to `applyFns`.
static LogicalResult getApplyFunctions(ModuleOp module,
                                       SmallVectorImpl<Operation *> &applyFns) {
  auto fns = module.getOps<FuncOp>();
  SymbolTable table(module);
  for (auto funcOp : fns) {
    if (!funcOp.getAttr("iree.executable.reduction")) continue;
    auto applyFnSymRef = funcOp.template getAttrOfType<FlatSymbolRefAttr>(
        "iree.executable.reduction.apply");
    auto applyFn = table.lookup<FuncOp>(applyFnSymRef.getValue());
    if (!applyFn) return module.emitError("can't find the apply function");
    applyFns.push_back(applyFn.getOperation());
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pattern builder
//===----------------------------------------------------------------------===//

static LogicalResult lowerReductionApplyFnToLinalg(MLIRContext *context,
                                                   ArrayRef<Operation *> fns) {
  TypeConverter converter;
  converter.addConversion([](Type type) {
    return type.isSignlessIntOrFloat() ? type : Optional<Type>();
  });
  converter.addConversion([](RankedTensorType type) {
    return type.getRank() == 0 ? type.getElementType() : Optional<Type>();
  });

  OwningRewritePatternList patterns;
  patterns
      .insert<ReductionApplyFnConversion, ReductionOpConversion<xla_hlo::AddOp>,
              ReductionOpConversion<xla_hlo::MinOp>,
              ReductionOpConversion<xla_hlo::MaxOp> >(context, converter);
  ConversionTarget target(*context);
  target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect>();
  target.addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
  if (failed(applyFullConversion(fns, target, patterns))) return failure();
  return success();
}

static LogicalResult lowerReductionEntryFnToLinalg(MLIRContext *context,
                                                   ModuleOp module) {
  OwningRewritePatternList patterns;
  OpBuilder builder(module.getBodyRegion());
  SymbolTable table(module);
  for (auto fn : module.getOps<FuncOp>()) {
    if (!fn.getAttr("iree.executable.reduction") || !fn.empty()) continue;
    auto applyFnSymRef = fn.template getAttrOfType<FlatSymbolRefAttr>(
        "iree.executable.reduction.apply");
    auto applyFn = table.lookup<FuncOp>(applyFnSymRef.getValue());
    if (failed(addReductionEntryFnBody(builder, fn, applyFn))) return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pass for invoking the conversion
//===----------------------------------------------------------------------===//

void HLOReductionToLinalgPass::runOnModule() {
  MLIRContext *context = &getContext();
  auto module = getModule();

  SmallVector<Operation *, 1> applyFns;
  if (failed(getApplyFunctions(module, applyFns))) return signalPassFailure();

  if (failed(lowerReductionApplyFnToLinalg(context, applyFns)))
    return signalPassFailure();

  if (failed(lowerReductionEntryFnToLinalg(context, module)))
    return signalPassFailure();

  // Erase all the apply functions because they are already inlined into entry
  // functions and there are no users.
  applyFns.clear();
  if (failed(getApplyFunctions(getModule(), applyFns)))
    return signalPassFailure();
  for (auto f : applyFns) f->erase();
}

static PassRegistration<HLOReductionToLinalgPass> pass(
    "iree-hlo-reduction-to-linalg",
    "Convert the reduction dispatch functions to Linalg");

std::unique_ptr<Pass> createHLOReductionToLinalgPass() {
  return std::make_unique<HLOReductionToLinalgPass>();
}

}  // namespace iree_compiler
}  // namespace mlir

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

//===- GPUKernelOutlining.cpp - Generate GPU device-side code -------------===//
//
// Implements a pass to convert a launch operation into a device-side code.
//
//===----------------------------------------------------------------------===//
#include "iree/compiler/Translation/CodegenUtils/CodegenUtils.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

namespace {
// Pattern to get the gpu.GPUModuleOp from the gpu.LaunchOp.
struct ConvertToGPUFuncOp : public OpRewritePattern<gpu::LaunchOp> {
  using OpRewritePattern<gpu::LaunchOp>::OpRewritePattern;
  PatternMatchResult matchAndRewrite(gpu::LaunchOp launchOp,
                                     PatternRewriter &rewriter) const final;
};

// Pass to outline the region of the gpu.LaunchOp.
class IREEGpuKernelOutliningPass
    : public FunctionPass<IREEGpuKernelOutliningPass> {
 public:
  void runOnFunction() override;
};
}  // namespace

PatternMatchResult ConvertToGPUFuncOp::matchAndRewrite(
    gpu::LaunchOp launchOp, PatternRewriter &rewriter) const {
  OpBuilder::InsertionGuard guard(rewriter);
  auto funcOp = launchOp.getParentOfType<FuncOp>();
  if (!isDispatchFunction(funcOp)) return matchFailure();

  // The arguments of the funcOp must be the arguments of the launchOp, in the
  // same order.
  SmallVector<Value, 4> arguments(funcOp.args_begin(), funcOp.args_end());
  gpu::GPUFuncOp gpuFuncOp =
      outlineKernelFunc(launchOp, funcOp.getName(), arguments);
  // If any additional arguments are needed, then the launch op cannot be
  // converted.
  if (arguments.size() != funcOp.getNumArguments()) return matchFailure();

  // Wrap this within a gpu.module
  std::string moduleName = Twine(funcOp.getName(), "_gpumodule").str();
  OperationState state(funcOp.getLoc(), gpu::GPUModuleOp::getOperationName());
  gpu::GPUModuleOp::build(&rewriter, state, moduleName);
  auto kernelModule = cast<gpu::GPUModuleOp>(Operation::create(state));
  SymbolTable symbolTable(kernelModule);
  symbolTable.insert(gpuFuncOp);

  rewriter.setInsertionPoint(funcOp);
  rewriter.insert(kernelModule);
  rewriter.eraseOp(launchOp);
  return matchSuccess();
}

void IREEGpuKernelOutliningPass::runOnFunction() {
  OwningRewritePatternList patterns;
  FuncOp func = getFunction();
  patterns.insert<ConvertToGPUFuncOp>(func.getContext());
  applyPatternsGreedily(func.getOperation(), patterns);
}

std::unique_ptr<OpPassBase<FuncOp>> createIREEGpuKernelOutliningPass() {
  return std::make_unique<IREEGpuKernelOutliningPass>();
}
}  // namespace iree_compiler
}  // namespace mlir

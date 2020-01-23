// Copyright 2019 Google LLC
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

#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"

#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/STLExtras.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

//===----------------------------------------------------------------------===//
// iree.get_ranked_shape
//===----------------------------------------------------------------------===//

static ParseResult parseGetRankedShapeOp(OpAsmParser &parser,
                                         OperationState &state) {
  OpAsmParser::OperandType operandType;
  Type resultType;
  return failure(
      parser.parseOperand(operandType) || parser.parseColonType(resultType) ||
      parser.parseOptionalArrowTypeList(state.types) ||
      parser.resolveOperand(operandType, resultType, state.operands));
}

static void printGetRankedShapeOp(OpAsmPrinter &p, GetRankedShapeOp op) {
  p << "shape.get_ranked_shape ";
  p.printOperand(op.operand());
  p << " : ";
  p.printType(op.operand().getType());
  p << " -> ";
  p.printType(op.shape().getType());
}

static LogicalResult verifyGetRankedShapeOp(GetRankedShapeOp op) {
  auto tensorType = op.operand().getType().cast<TensorType>();
  auto rsType = op.shape().getType().cast<RankedShapeType>();
  if (tensorType.getRank() != rsType.getRank()) {
    return op.emitOpError("operand and result must be of same rank");
  }
  SmallVector<int64_t, 4> rsDims;
  rsType.getAllDims(rsDims);
  if (!std::equal(rsDims.begin(), rsDims.end(),
                  tensorType.getShape().begin())) {
    return op.emitOpError("operand tensor and result shape must be equal");
  }
  return success();
}

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.cpp.inc"

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir
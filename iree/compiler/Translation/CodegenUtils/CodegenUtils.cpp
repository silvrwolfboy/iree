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

#include "iree/compiler/Translation/CodegenUtils/CodegenUtils.h"

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace iree_compiler {

ArrayRef<int64_t> dropTrailingOnes(ArrayRef<int64_t> vector) {
  if (vector.empty()) return vector;
  auto numTrailingOnes = 0;
  for (unsigned i = vector.size() - 1; i > 0; --i) {
    if (vector[i] != 1) {
      break;
    }
    numTrailingOnes++;
  }
  return vector.drop_back(numTrailingOnes);
}

bool isDispatchFunction(FuncOp funcOp) {
  return funcOp.getAttr("iree.executable.export") != nullptr;
}

/// Helper function to check shapes are equal. Only care that the number of
/// elements be equal.
static bool areShapesEqual(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
  auto reduceFn = [](ArrayRef<int64_t> vector) -> int64_t {
    int64_t init = 1;
    for (auto val : vector) init *= val;
    return init;
  };
  return reduceFn(lhs) == reduceFn(rhs);
}

/// Get the shape to use for a type. For now this is returning shapes as static
/// value.
// TODO(ravishankarm) : Modify this to return the Values to use for the extent
// to handle dynamic shapes.
static LogicalResult getExtentFromStoreOpSrc(Operation *storeOp,
                                             SmallVectorImpl<int64_t> &extent) {
  Value srcVal = storeOp->getOperand(0);
  if (srcVal.getType().isSignlessIntOrFloat()) {
    extent.clear();
    extent.push_back(1);
    return success();
  }
  if (auto shapedType = srcVal.getType().dyn_cast<ShapedType>()) {
    if (shapedType.hasStaticShape()) {
      extent.assign(shapedType.getShape().rbegin(),
                    shapedType.getShape().rend());
      if (extent.empty()) {
        extent.clear();
        extent.push_back(1);
      }
      return success();
    }
  }
  return storeOp->emitError(
      "unable to extract domain size from store operation");
}

// TODO(ravishankarm) : Modify this to return the Values to support dynamic
// shapes.
LogicalResult getLaunchSize(FuncOp funcOp,
                            SmallVectorImpl<int64_t> &launchSize) {
  auto &body = funcOp.getBody();
  if (!mlir::has_single_element(body)) {
    return funcOp.emitError(
        "unhandled multiple blocks within dispatch function");
  }
  SmallVector<Operation *, 1> storeOperations;
  auto storeOps = body.front().getOps<IREE::StoreOutputOp>();
  storeOperations.assign(storeOps.begin(), storeOps.end());
  auto storeReduceOps = body.front().getOps<IREE::StoreReduceOp>();
  storeOperations.append(storeReduceOps.begin(), storeReduceOps.end());
  if (storeOperations.begin() == storeOperations.end()) {
    return funcOp.emitError(
        "expected dispatch function to have at least one iree.store_output "
        "instruction");
  }

  Operation *firstStoreOp = *storeOperations.begin();
  if (failed(getExtentFromStoreOpSrc(firstStoreOp, launchSize))) {
    return firstStoreOp->emitError("unhandled type of the output tensor");
  }
  for (auto it = std::next(storeOperations.begin()), ie = storeOperations.end();
       it != ie; ++it) {
    SmallVector<int64_t, 3> checkShape;
    Operation *storeOp = *it;
    if (failed(getExtentFromStoreOpSrc(storeOp, checkShape))) {
      return storeOp->emitError("unhandled type of the output tensor");
    }
    if (!areShapesEqual(launchSize, checkShape)) {
      return storeOp->emitError("mismatch in shapes of the output tensors");
    }
  }
  return success();
}

/// Gets the workgroup size.
template <typename intType>
LogicalResult getWorkGroupSize(FuncOp funcOp,
                               SmallVectorImpl<intType> &workGroupSize) {
  if (!isDispatchFunction(funcOp)) {
    return funcOp.emitError(
        "expected operation to be in dispatch function to get launch size");
  }
  auto workGroupSizeAttr =
      funcOp.getAttrOfType<DenseElementsAttr>("iree.executable.workgroup_size");
  if (!workGroupSizeAttr) {
    return funcOp.emitError(
        "unable to find workgroup size, missing attribute "
        "iree.executable.workgroup_size in dispatch function");
  }
  workGroupSize.clear();
  for (auto value : workGroupSizeAttr.getValues<APInt>()) {
    workGroupSize.push_back(value.getSExtValue());
  }
  return success();
}

template LogicalResult getWorkGroupSize<int32_t>(
    FuncOp funcOp, SmallVectorImpl<int32_t> &workGroupSize);
template LogicalResult getWorkGroupSize<int64_t>(
    FuncOp funcOp, SmallVectorImpl<int64_t> &workGroupSize);

}  // namespace iree_compiler
}  // namespace mlir

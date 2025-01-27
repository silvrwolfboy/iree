# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_subdirectory(test)

iree_cc_library(
  NAME
    XLAToLinalg
  HDRS
    "Passes.h"
  SRCS
    "LinalgFusion.cpp"
    "LinalgTensorToBuffer.cpp"
    "XLAToLinalg.cpp"
    "XLAToLinalgOpsOnBuffers.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRLinalgOps
    MLIRLinalgTransforms
    MLIRPass
    MLIRStandardOps
    MLIRSupport
    MLIRTransforms
    iree::compiler::Dialect::IREE::IR
    tensorflow::mlir_xla
  ALWAYSLINK
  PUBLIC
)

iree_cc_library(
  NAME
    ReductionLowering
  HDRS
    "ReductionLowering.h"
  SRCS
    "ReductionLowering.cpp"
  DEPS
    MLIRIR
    MLIRLinalgOps
    MLIRPass
    MLIRStandardOps
    MLIRTransforms
    iree::compiler::Dialect::IREE::IR
    tensorflow::mlir_xla
  ALWAYSLINK
  PUBLIC
)

# Copyright 2019 Google LLC
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
    Transforms
  HDRS
    "Passes.h"
  SRCS
    "MaterializeInterfaces.cpp"
    "Passes.cpp"
    "RewriteLegacyIO.cpp"
    "TranslateExecutables.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRPass
    MLIRStandardOps
    MLIRSupport
    MLIRTransforms
    iree::compiler::Dialect::Flow::IR
    iree::compiler::Dialect::HAL::Conversion::FlowToHAL
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::HAL::Target::ExecutableTarget
    iree::compiler::Dialect::IREE::IR
    iree::compiler::Utils
    tensorflow::mlir_xla
  ALWAYSLINK
  PUBLIC
)

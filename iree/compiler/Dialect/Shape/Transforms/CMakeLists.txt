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
    Transforms
  HDRS
    "Passes.h"
  SRCS
    "CleanupPlaceholders.cpp"
    "ConvertHLOToShapeDialect.cpp"
    "ExpandFunctionDynamicDims.cpp"
    "HoistShapeCalculations.cpp"
    "MaterializeShapeCalculations.cpp"
    "Passes.cpp"
    "TieDynamicShapes.cpp"
  DEPS
    MLIRAnalysis
    MLIRIR
    MLIRPass
    MLIRStandardOps
    MLIRTransforms
    iree::compiler::Dialect::Shape::IR
    iree::compiler::Dialect::Shape::Plugins::XLA::XlaHloShapeBuilder
    tensorflow::mlir_xla
  ALWAYSLINK
  PUBLIC
)

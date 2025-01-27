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

file(GLOB _GLOB_X_TD CONFIGURE_DEPENDS *.td)
iree_cc_library(
  NAME
    IR
  HDRS
    "HALEnums.h.inc"
    "HALOpInterface.h.inc"
    "HALOps.h"
    "HALOps.h.inc"
    "HALTypes.h"
  TEXTUAL_HDRS
    "HALOps.cpp.inc"
  SRCS
    "HALEnums.cpp.inc"
    "HALOpFolders.cpp"
    "HALOpInterface.cpp.inc"
    "HALOps.cpp"
    "HALTypes.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRParser
    MLIRStandardOps
    MLIRSupport
    MLIRTransformUtils
    iree::compiler::Dialect::IREE::IR
  PUBLIC
)

iree_cc_library(
  NAME
    HALDialect
  HDRS
    "HALDialect.h"
  SRCS
    "HALDialect.cpp"
  DEPS
    ::IR
    LLVMSupport
    MLIRIR
    MLIRParser
    MLIRStandardOps
    MLIRSupport
    MLIRTransformUtils
    iree::compiler::Dialect::HAL::Conversion::HALToVM
    iree::compiler::Dialect::HAL::hal_imports
    iree::compiler::Dialect::IREE::IR
    iree::compiler::Dialect::VM::Conversion
  ALWAYSLINK
  PUBLIC
)

iree_tablegen_library(
  NAME
    HALEnumsGen
  TD_FILE
    "HALBase.td"
  OUTS
    -gen-enum-decls HALEnums.h.inc
    -gen-enum-defs HALEnums.cpp.inc
)

iree_tablegen_library(
  NAME
    HALOpInterfaceGen
  TD_FILE
    "HALBase.td"
  OUTS
    -gen-op-interface-decls HALOpInterface.h.inc
    -gen-op-interface-defs HALOpInterface.cpp.inc
)

iree_tablegen_library(
  NAME
    HALOpsGen
  TD_FILE
    "HALOps.td"
  OUTS
    -gen-op-decls HALOps.h.inc
    -gen-op-defs HALOps.cpp.inc
)

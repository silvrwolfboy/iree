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
    Common
  HDRS
    "CommonDialect.h"
    "CommonOps.h"
    "CommonOps.h.inc"
  SRCS
    "CommonDialect.cpp"
    "CommonOps.cpp"
    "CommonOps.cpp.inc"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRStandardOps
    MLIRSupport
    iree::compiler::Dialect::IREE::IR
  ALWAYSLINK
  PUBLIC
)

iree_tablegen_library(
  NAME
    CommonOpsGen
  TD_FILE
    "CommonOps.td"
  OUTS
    -gen-op-decls CommonOps.h.inc
    -gen-op-defs CommonOps.cpp.inc
)

iree_cc_library(
  NAME
    IR
  HDRS
    "HLDialect.h"
    "HLOps.h"
    "HLOps.h.inc"
    "LLDialect.h"
    "LLOps.h"
    "LLOps.h.inc"
    "OpWriters.h"
  SRCS
    "HLDialect.cpp"
    "HLOps.cpp"
    "HLOps.cpp.inc"
    "LLDialect.cpp"
    "LLOps.cpp"
    "LLOps.cpp.inc"
    "OpWriters.cpp"
  DEPS
    ::Common
    LLVMSupport
    MLIRIR
    MLIRStandardOps
    MLIRSupport
    iree::compiler::Dialect::IREE::IR
    iree::compiler::Translation::Interpreter::Serialization
    iree::compiler::Translation::Interpreter::Utils
    iree::compiler::Utils
    iree::schemas::bytecode::interpreter_bytecode_v0
  ALWAYSLINK
  PUBLIC
)

iree_tablegen_library(
  NAME
    HLOpsGen
  TD_FILE
    "HLOps.td"
  OUTS
    -gen-op-decls HLOps.h.inc
    -gen-op-defs HLOps.cpp.inc
)

iree_tablegen_library(
  NAME
    LLOpsGen
  TD_FILE
    "LLOps.td"
  OUTS
    -gen-op-decls LLOps.h.inc
    -gen-op-defs LLOps.cpp.inc
)

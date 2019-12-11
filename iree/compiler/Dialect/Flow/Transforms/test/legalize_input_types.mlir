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

// RUN: iree-opt -split-input-file -verify-diagnostics -iree-flow-legalize-input-types %s | IreeFileCheck %s

// CHECK-LABEL: func @constantI64
// CHECK-SAME: () -> i32
func @constantI64() -> i64 {
  // CHECK-NEXT: constant 123 : i32
  %c123 = constant 123 : i64
  return %c123 : i64
}

// -----

// CHECK-LABEL: func @constantF64
// CHECK-SAME: () -> f32
func @constantF64() -> f64 {
  // CHECK-NEXT: constant 1.234000e+02 : f32
  %c1234 = constant 123.4 : f64
  return %c1234 : f64
}

// -----

// CHECK-LABEL: func @constantSplatTensorI64
// CHECK-SAME: () -> tensor<4xi32>
func @constantSplatTensorI64() -> tensor<4xi64> {
  // CHECK-NEXT: constant dense<123> : tensor<4xi32>
  %c123 = constant dense<123> : tensor<4xi64>
  return %c123 : tensor<4xi64>
}

// -----

// CHECK-LABEL: func @constantDenseTensorI64
// CHECK-SAME: () -> tensor<4xi32>
func @constantDenseTensorI64() -> tensor<4xi64> {
  // CHECK-NEXT: constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %c123 = constant dense<[0, 1, 2, 3]> : tensor<4xi64>
  return %c123 : tensor<4xi64>
}

// -----

// CHECK-LABEL: func @typesIndex
// CHECK-SAME: (%arg0: i32) -> i32
func @typesIndex(%arg0 : index) -> index {
  // CHECK-NEXT: return %arg0 : i32
  return %arg0 : index
}

// -----

// CHECK-LABEL: func @typesI64
// CHECK-SAME: (%arg0: i32) -> i32
func @typesI64(%arg0 : i64) -> i64 {
  // CHECK-NEXT: return %arg0 : i32
  return %arg0 : i64
}

// -----

// CHECK-LABEL: func @tensorTypesI64
// CHECK-SAME: (%arg0: tensor<4x4xi32>) -> tensor<4x4xi32>
func @tensorTypesI64(%arg0 : tensor<4x4xi64>) -> tensor<4x4xi64> {
  // CHECK-NEXT: return %arg0 : tensor<4x4xi32>
  return %arg0 : tensor<4x4xi64>
}

// -----

// CHECK-LABEL: func @tensorTypesF64
// CHECK-SAME: (%arg0: tensor<4x4xf32>) -> tensor<4x4xf32>
func @tensorTypesF64(%arg0 : tensor<4x4xf64>) -> tensor<4x4xf64> {
  // CHECK-NEXT: return %arg0 : tensor<4x4xf32>
  return %arg0 : tensor<4x4xf64>
}

// -----
// expected-error@+1 {{'func' op unable to legalize type of input 0}}
func @tensorUnrankedArg(%arg0 : tensor<*xi64>) -> tensor<*xi64> {
  return %arg0 : tensor<*xi64>
}

// -----
func @tensorUnrankedValue(%arg0 : tensor<4xi64>) -> tensor<4xi64> {
  // expected-error@+1 {{'std.tensor_cast' op unable to legalize operation types}}
  %0 = tensor_cast %arg0 : tensor<4xi64> to tensor<*xi64>
  %1 = tensor_cast %0 : tensor<*xi64> to tensor<4xi64>
  return %1 : tensor<4xi64>
}

// -----

// CHECK-LABEL: func @compareI64
// CHECK-SAME: (%arg0: tensor<i32>, %arg1: tensor<i32>) -> (i1, tensor<i32>)
func @compareI64(%arg0 : tensor<i64>, %arg1 : tensor<i64>) -> (i1, tensor<i64>) {
  // CHECK-NEXT: %0 = "xla_hlo.compare"(%arg0, %arg1) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT: %1 = extract_element %0[] : tensor<i1>
  // CHECK-NEXT: cond_br %1, ^bb1(%1, %arg0 : i1, tensor<i32>), ^bb2(%1, %arg1 : i1, tensor<i32>)
  // CHECK-NEXT: ^bb1(%2: i1, %3: tensor<i32>): // pred: ^bb0
  // CHECK-NEXT: return %2, %3 : i1, tensor<i32>
  // CHECK-NEXT: ^bb2(%4: i1, %5: tensor<i32>): // pred: ^bb0
  // CHECK-NEXT: return %4, %5 : i1, tensor<i32>
  %0 = "xla_hlo.compare"(%arg0, %arg1) {comparison_direction = "LT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %1 = extract_element %0[] : tensor<i1>
  cond_br %1, ^bb1(%1, %arg0 : i1, tensor<i64>), ^bb2(%1, %arg1 : i1, tensor<i64>)
^bb1(%2 : i1, %3 : tensor<i64>):
  return %2, %3 : i1, tensor<i64>
^bb2(%4 : i1, %5 : tensor<i64>):
  return %4, %5 : i1, tensor<i64>
}
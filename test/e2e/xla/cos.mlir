// RUN: iree-run-mlir -iree-hal-target-backends=interpreter-bytecode %s | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @tensor
func @tensor() -> tensor<4xf32> {
  %input = iree.unfoldable_constant dense<[0.0, 1.0, 1.5, 2.0]> : tensor<4xf32>
  %result = "xla_hlo.cos"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}
// CHECK: 4xf32=1 0.540302 0.0707372 -0.416147

// -----

// CHECK-LABEL: EXEC @scalar
func @scalar() -> tensor<f32> {
  %input = iree.unfoldable_constant dense<3.0> : tensor<f32>
  %result = "xla_hlo.cos"(%input) : (tensor<f32>) -> tensor<f32>
  return %result : tensor<f32>
}
// CHECK: f32=-0.989992

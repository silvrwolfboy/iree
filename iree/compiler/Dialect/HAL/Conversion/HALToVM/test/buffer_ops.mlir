// RUN: iree-opt -split-input-file -iree-convert-hal-to-vm %s | IreeFileCheck %s

// CHECK-LABEL: @buffer_subspan
func @buffer_subspan(%arg0 : !hal.buffer) -> !hal.buffer {
  %0 = "test_hal.device_size"() : () -> i32
  %1 = "test_hal.device_size"() : () -> i32
  // CHECK: %ref = vm.call @hal.buffer.subspan(%arg0, %0, %1) : (!vm.ref<!hal.buffer>, i32, i32) -> !vm.ref<!hal.buffer>
  %buffer = hal.buffer.subspan %arg0, %0, %1 : !hal.buffer
  return %buffer : !hal.buffer
}

// -----

// CHECK-LABEL: @buffer_fill
func @buffer_fill(%arg0 : !hal.buffer) {
  %0 = "test_hal.device_size"() : () -> i32
  %1 = "test_hal.device_size"() : () -> i32
  %2 = "test_hal.pattern"() : () -> i32
  // CHECK: vm.call @hal.buffer.fill(%arg0, %0, %1, %2) : (!vm.ref<!hal.buffer>, i32, i32, i32) -> ()
  hal.buffer.fill %arg0, %0, %1, %2
  return
}

// -----

// CHECK-LABEL: @buffer_read_data
func @buffer_read_data(%arg0 : !hal.buffer, %arg1 : !iree.mutable_byte_buffer) {
  %0 = "test_hal.device_size"() : () -> i32
  %1 = "test_hal.device_size"() : () -> i32
  %2 = "test_hal.device_size"() : () -> i32
  // CHECK: vm.call @hal.buffer.read_data(%arg0, %0, %arg1, %1, %2) : (!vm.ref<!hal.buffer>, i32, !vm.ref<!iree.mutable_byte_buffer>, i32, i32) -> ()
  hal.buffer.read_data %arg0, %0, %arg1, %1, %2 : !iree.mutable_byte_buffer
  return
}

// -----

// CHECK-LABEL: @buffer_write_data
func @buffer_write_data(%arg0 : !hal.buffer, %arg1 : !iree.mutable_byte_buffer) {
  %0 = "test_hal.device_size"() : () -> i32
  %1 = "test_hal.device_size"() : () -> i32
  %2 = "test_hal.device_size"() : () -> i32
  // CHECK: vm.call @hal.buffer.write_data(%arg1, %0, %arg0, %1, %2) : (!vm.ref<!iree.mutable_byte_buffer>, i32, !vm.ref<!hal.buffer>, i32, i32) -> ()
  hal.buffer.write_data %arg1, %0, %arg0, %1, %2 : !iree.mutable_byte_buffer
  return
}

// -----

// CHECK-LABEL: @buffer_copy_data
func @buffer_copy_data(%arg0 : !hal.buffer, %arg1 : !hal.buffer) {
  %0 = "test_hal.device_size"() : () -> i32
  %1 = "test_hal.device_size"() : () -> i32
  %2 = "test_hal.device_size"() : () -> i32
  // CHECK: vm.call @hal.buffer.copy_data(%arg0, %0, %arg1, %1, %2) : (!vm.ref<!hal.buffer>, i32, !vm.ref<!hal.buffer>, i32, i32) -> ()
  hal.buffer.copy_data %arg0, %0, %arg1, %1, %2
  return
}

// -----

// CHECK-LABEL: @buffer_load
func @buffer_load(%arg0 : !hal.buffer) -> (i8, i16, i32) {
  %0 = "test_hal.device_size"() : () -> i32
  // CHECK: %1 = vm.call @hal.buffer.load(%arg0, %0, %c1) : (!vm.ref<!hal.buffer>, i32, i32) -> i32
  %1 = hal.buffer.load %arg0[%0] : i8
  // CHECK: %2 = vm.call @hal.buffer.load(%arg0, %0, %c2) : (!vm.ref<!hal.buffer>, i32, i32) -> i32
  %2 = hal.buffer.load %arg0[%0] : i16
  // CHECK: %3 = vm.call @hal.buffer.load(%arg0, %0, %c4) : (!vm.ref<!hal.buffer>, i32, i32) -> i32
  %3 = hal.buffer.load %arg0[%0] : i32
  return %1, %2, %3 : i8, i16, i32
}

// -----

// CHECK-LABEL: @buffer_store
func @buffer_store(%arg0 : !hal.buffer, %arg1 : i8, %arg2 : i16, %arg3 : i32) {
  %0 = "test_hal.device_size"() : () -> i32
  // CHECK: vm.call @hal.buffer.store(%arg1, %arg0, %0, %c1) : (i32, !vm.ref<!hal.buffer>, i32, i32) -> ()
  hal.buffer.store %arg1, %arg0[%0] : i8
  // CHECK: vm.call @hal.buffer.store(%arg2, %arg0, %0, %c2) : (i32, !vm.ref<!hal.buffer>, i32, i32) -> ()
  hal.buffer.store %arg2, %arg0[%0] : i16
  // CHECK: vm.call @hal.buffer.store(%arg3, %arg0, %0, %c4) : (i32, !vm.ref<!hal.buffer>, i32, i32) -> ()
  hal.buffer.store %arg3, %arg0[%0] : i32
  return
}

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

// clang-format off

// NOLINTNEXTLINE
// RUN: test-simple-jit -runtime-support=$(dirname %s)/runtime-support.so 2>&1 | IreeFileCheck %s

// clang-format on

#include "experimental/ModelBuilder/MemRefUtils.h"
#include "experimental/ModelBuilder/ModelBuilder.h"
#include "experimental/ModelBuilder/ModelRunner.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;  // NOLINT

static llvm::cl::opt<std::string> runtimeSupport(
    "runtime-support", llvm::cl::desc("Runtime support library filename"),
    llvm::cl::value_desc("filename"), llvm::cl::init("-"));

// Flush the different output streams. Needed to ensure that the IreeFileCheck
// utility sees the various buffered streams in a reasonable order.
static void flush() {
  fflush(stderr);
  fflush(stdout);
  llvm::errs().flush();
  llvm::outs().flush();
}

template <unsigned M>
void testVectorAdd1d(StringLiteral funcName, unsigned kNumElements) {
  ModelBuilder modelBuilder;

  auto f32 = modelBuilder.f32;
  auto mVectorType = modelBuilder.getVectorType({M}, f32);
  auto typeA = modelBuilder.getMemRefType({kNumElements}, mVectorType);
  auto typeB = modelBuilder.getMemRefType({kNumElements}, mVectorType);
  auto typeC = modelBuilder.getMemRefType({kNumElements}, mVectorType);

  // 1. Build a simple vector_add.
  {
    auto f = modelBuilder.makeFunction(funcName, {}, {typeA, typeB, typeC});
    OpBuilder b(&f.getBody());
    ScopedContext scope(b, f.getLoc());

    StdIndexedValue A(f.getArgument(0)), B(f.getArgument(1)),
        C(f.getArgument(2));
    auto last = std_constant_index(kNumElements - 1);
    C(last) = A(last) + B(last);

    (vector_print(*A(last)));
    (vector_print(*B(last)));
    (vector_print(*C(last)));

    std_ret();
  }

  modelBuilder.getModuleRef()->dump();
  flush();

  // 2. Compile the function, pass in runtime support library
  //    to the execution engine for vector.print.
  ModelRunner runner(modelBuilder.getModuleRef());
  runner.compile(/*llvmOptLevel=*/3, /*llcOptLevel=*/3, runtimeSupport);

  // 3. Allocate data within data structures that interoperate with the MLIR ABI
  // conventions used by codegen.
  auto oneInit = [](unsigned idx, Vector1D<M, float> *ptr) {
    for (unsigned i = 0; i < M; ++i) ptr[idx][i] = 1.0f;
  };
  auto incInit = [](unsigned idx, Vector1D<M, float> *ptr) {
    for (unsigned i = 0; i < M; ++i) ptr[idx][i] = 1.0f + idx * M + i;
  };
  auto zeroInit = [](unsigned idx, Vector1D<M, float> *ptr) {
    for (unsigned i = 0; i < M; ++i) ptr[idx][i] = 0.0f;
  };
  auto A = makeInitializedStridedMemRefDescriptor<Vector1D<M, float>, 1>(
      {kNumElements}, oneInit);
  auto B = makeInitializedStridedMemRefDescriptor<Vector1D<M, float>, 1>(
      {kNumElements}, incInit);
  auto C = makeInitializedStridedMemRefDescriptor<Vector1D<M, float>, 1>(
      {kNumElements}, zeroInit);

  // 5. Call the funcOp named `funcName`.
  const std::string kFuncAdapterName =
      (llvm::Twine("_mlir_ciface_") + funcName).str();
  auto *bufferA = A.get();
  auto *bufferB = B.get();
  auto *bufferC = C.get();
  void *args[3] = {&bufferA, &bufferB, &bufferC};

  auto err =
      runner.engine->invoke(kFuncAdapterName, MutableArrayRef<void *>{args});
  flush();

  if (err) llvm_unreachable("Error running function.");
}

template <unsigned M, unsigned N>
void testVectorAdd2d(StringLiteral funcName, unsigned kNumElements) {
  ModelBuilder modelBuilder;

  auto f32 = modelBuilder.f32;
  auto mnVectorType = modelBuilder.getVectorType({M, N}, f32);
  auto typeA = modelBuilder.getMemRefType({kNumElements}, mnVectorType);
  auto typeB = modelBuilder.getMemRefType({kNumElements}, mnVectorType);
  auto typeC = modelBuilder.getMemRefType({kNumElements}, mnVectorType);

  // 1. Build a simple vector_add.
  {
    auto f = modelBuilder.makeFunction(funcName, {}, {typeA, typeB, typeC});
    OpBuilder b(&f.getBody());
    ScopedContext scope(b, f.getLoc());

    StdIndexedValue A(f.getArgument(0)), B(f.getArgument(1)),
        C(f.getArgument(2));
    auto last = std_constant_index(kNumElements - 1);
    C(last) = A(last) + B(last);

    (vector_print(*A(last)));
    (vector_print(*B(last)));
    (vector_print(*C(last)));

    std_ret();
  }

  modelBuilder.getModuleRef()->dump();
  flush();

  // 2. Compile the function, pass in runtime support library
  //    to the execution engine for vector.print.
  ModelRunner runner(modelBuilder.getModuleRef());
  runner.compile(/*llvmOptLevel=*/3, /*llcOptLevel=*/3, runtimeSupport);

  // 3. Allocate data within data structures that interoperate with the MLIR ABI
  // conventions used by codegen.
  auto oneInit = [](unsigned idx, Vector2D<M, N, float> *ptr) {
    for (unsigned i = 0; i < M; ++i)
      for (unsigned j = 0; j < N; ++j) ptr[idx][i][j] = 1.0f;
  };
  auto incInit = [](unsigned idx, Vector2D<M, N, float> *ptr) {
    for (unsigned i = 0; i < M; ++i)
      for (unsigned j = 0; j < N; ++j)
        ptr[idx][i][j] = 1.0f + idx * M * N + i * N + j;
  };
  auto zeroInit = [](unsigned idx, Vector2D<M, N, float> *ptr) {
    for (unsigned i = 0; i < M; ++i)
      for (unsigned j = 0; j < N; ++j) ptr[idx][i][j] = 0.0f;
  };
  auto A = makeInitializedStridedMemRefDescriptor<Vector2D<M, N, float>, 1>(
      {kNumElements}, oneInit);
  auto B = makeInitializedStridedMemRefDescriptor<Vector2D<M, N, float>, 1>(
      {kNumElements}, incInit);
  auto C = makeInitializedStridedMemRefDescriptor<Vector2D<M, N, float>, 1>(
      {kNumElements}, zeroInit);

  // 5. Call the funcOp named `funcName`.
  const std::string kFuncAdapterName =
      (llvm::Twine("_mlir_ciface_") + funcName).str();
  auto *bufferA = A.get();
  auto *bufferB = B.get();
  auto *bufferC = C.get();
  void *args[3] = {&bufferA, &bufferB, &bufferC};

  auto err =
      runner.engine->invoke(kFuncAdapterName, MutableArrayRef<void *>{args});
  flush();
  if (err) llvm_unreachable("Error running function.");
}

template <unsigned M, unsigned N, unsigned K>
void testMatmulOnVectors(StringLiteral funcName) {
  ModelBuilder modelBuilder;

  auto f32 = modelBuilder.f32;
  auto mkVectorType = modelBuilder.getVectorType({M, K}, f32);
  auto typeA = modelBuilder.getMemRefType({-1, -1}, mkVectorType);
  auto knVectorType = modelBuilder.getVectorType({K, N}, f32);
  auto typeB = modelBuilder.getMemRefType({-1, -1}, knVectorType);
  auto mnVectorType = modelBuilder.getVectorType({M, N}, f32);
  auto typeC = modelBuilder.getMemRefType({-1, -1}, mnVectorType);

  auto func = modelBuilder.makeFunction(funcName, {}, {typeA, typeB, typeC});

  OpBuilder b(&func.getBody());
  ScopedContext scope(b, func.getLoc());
  ValueHandle A(func.getArgument(0)), B(func.getArgument(1)),
      C(func.getArgument(2));
  auto contractionBuilder = [](ArrayRef<BlockArgument> args) {
    assert(args.size() == 3 && "expected 3 block arguments");
    (linalg_yield(vector_matmul(args[0], args[1], args[2])));
  };

  linalg_matmul(A, B, C, contractionBuilder);
  std_ret();

  modelBuilder.getModuleRef()->dump();
  flush();
}

int main(int argc, char **argv) {
  // Allow LLVM setup through command line and parse the
  // test specific option for a runtime support library.
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "TestSimpleJIT\n");

  // Verify generated MLIR:
  //
  // CHECK-LABEL: func @test_vector_add_1d_1x3f32
  //  CHECK-SAME: %[[A:.*0]]: memref<1xvector<3xf32>>,
  //  CHECK-SAME: %[[B:.*1]]: memref<1xvector<3xf32>>,
  //  CHECK-SAME: %[[C:.*2]]: memref<1xvector<3xf32>>)
  //   CHECK-DAG: %[[z:.*]] = constant 0 : index
  //   CHECK-DAG: %[[a:.*]] = load %[[A]][%[[z]]] : memref<1xvector<3xf32>>
  //   CHECK-DAG: %[[b:.*]] = load %[[B]][%[[z]]] : memref<1xvector<3xf32>>
  //       CHECK: %[[c:.*]] = addf %[[a]], %[[b]] : vector<3xf32>
  //       CHECK: store %[[c]], %[[C]][%[[z]]] : memref<1xvector<3xf32>>
  //       CHECK: %[[p:.*]] = load %[[A]][%[[z]]] : memref<1xvector<3xf32>>
  //       CHECK: vector.print %[[p]] : vector<3xf32>
  //       CHECK: %[[q:.*]] = load %[[B]][%[[z]]] : memref<1xvector<3xf32>>
  //       CHECK: vector.print %[[q]] : vector<3xf32>
  //       CHECK: %[[r:.*]] = load %[[C]][%[[z]]] : memref<1xvector<3xf32>>
  //       CHECK: vector.print %[[r]] : vector<3xf32>
  //
  // Verify runtime output:
  //       CHECK: ( 1, 1, 1 )
  //       CHECK: ( 1, 2, 3 )
  //       CHECK: ( 2, 3, 4 )
  testVectorAdd1d<3>("test_vector_add_1d_1x3f32", /*kNumElements=*/1);

  // CHECK-LABEL: func @test_vector_add_1d_2x3f32
  //       CHECK: ( 1, 1, 1 )
  //       CHECK: ( 4, 5, 6 )
  //       CHECK: ( 5, 6, 7 )
  testVectorAdd1d<3>("test_vector_add_1d_2x3f32", /*kNumElements=*/2);

  // CHECK-LABEL: func @test_vector_add_1d_2x5f32
  //       CHECK: ( 1, 1, 1, 1, 1 )
  //       CHECK: ( 6, 7, 8, 9, 10 )
  //       CHECK: ( 7, 8, 9, 10, 11 )
  testVectorAdd1d<5>("test_vector_add_1d_2x5f32", /*kNumElements=*/2);

  // Verify generated MLIR:
  //
  // CHECK-LABEL: func @test_vector_add_2d_1x2_3f32
  //  CHECK-SAME: %[[A:.*0]]: memref<1xvector<2x3xf32>>,
  //  CHECK-SAME: %[[B:.*1]]: memref<1xvector<2x3xf32>>,
  //  CHECK-SAME: %[[C:.*2]]: memref<1xvector<2x3xf32>>)
  //   CHECK-DAG: %[[z:.*]] = constant 0 : index
  //   CHECK-DAG: %[[a:.*]] = load %[[A]][%[[z]]] : memref<1xvector<2x3xf32>>
  //   CHECK-DAG: %[[b:.*]] = load %[[B]][%[[z]]] : memref<1xvector<2x3xf32>>
  //       CHECK: %[[c:.*]] = addf %[[a]], %[[b]] : vector<2x3xf32>
  //       CHECK: store %[[c]], %[[C]][%[[z]]] : memref<1xvector<2x3xf32>>
  //       CHECK: %[[p:.*]] = load %[[A]][%[[z]]] : memref<1xvector<2x3xf32>>
  //       CHECK: vector.print %[[p]] : vector<2x3xf32>
  //       CHECK: %[[q:.*]] = load %[[B]][%[[z]]] : memref<1xvector<2x3xf32>>
  //       CHECK: vector.print %[[q]] : vector<2x3xf32>
  //       CHECK: %[[r:.*]] = load %[[C]][%[[z]]] : memref<1xvector<2x3xf32>>
  //       CHECK: vector.print %[[r]] : vector<2x3xf32>
  //
  // Verify runtime output:
  //       CHECK: ( ( 1, 1, 1 ), ( 1, 1, 1 ) )
  //       CHECK: ( ( 1, 2, 3 ), ( 4, 5, 6 ) )
  //       CHECK: ( ( 2, 3, 4 ), ( 5, 6, 7 ) )
  testVectorAdd2d<2, 3>("test_vector_add_2d_1x2_3f32", /*kNumElements=*/1);

  // CHECK-LABEL: func @test_vector_add_2d_3x3_5f32
  // CHECK: ( ( 1, 1, 1, 1, 1 ), ( 1, 1, 1, 1, 1 ), ( 1, 1, 1, 1, 1 ) )
  // CHECK: ( ( 31, 32, 33, 34, 35 ), ( 36{{.*}}40 ), ( 41, 42, 43, 44, 45 ) )
  // CHECK: ( ( 32, 33, 34, 35, 36 ), ( 37{{.*}}41 ), ( 42, 43, 44, 45, 46 ) )
  testVectorAdd2d<3, 5>("test_vector_add_2d_3x3_5f32", /*kNumElements=*/3);

  // Verify generated MLIR:
  //
  // CHECK-LABEL: func @test_vector_matmul(
  //  CHECK-SAME:   %[[A:.*]]: memref<?x?xvector<4x16xf32>>,
  //  CHECK-SAME:   %[[B:.*]]: memref<?x?xvector<16x8xf32>>,
  //  CHECK-SAME:   %[[C:.*]]: memref<?x?xvector<4x8xf32>>)
  //       CHECK:   linalg.generic {{.*}} %[[A]], %[[B]], %[[C]]
  //       CHECK:     vector.contract {{.*}} : vector<4x16xf32>,
  //       vector<16x8xf32>
  //  CHECK-SAME:       into vector<4x8xf32>
  //
  // Verify runtime output:
  // TBD.
  testMatmulOnVectors<4, 8, 16>("test_vector_matmul");
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination_clamp.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

torch::Tensor holoq_int4_mm_cuda(torch::Tensor a_packed, torch::Tensor b_packed, int64_t k) {
  const at::cuda::CUDAGuard device_guard(a_packed.device());
  TORCH_CHECK(a_packed.get_device() == b_packed.get_device(), "A and B must share a CUDA device");
  const int64_t m = a_packed.size(0);
  const int64_t n = b_packed.size(0);
  TORCH_CHECK(m > 0 && n > 0 && k > 0, "M, N, and K must be positive");

  auto output = torch::empty({m, n}, a_packed.options().dtype(torch::kInt32));

  using Element = cutlass::int4b_t;
  using Accumulator = int32_t;
  using Output = int32_t;
  using Epilogue = cutlass::epilogue::thread::LinearCombinationClamp<
      Output,
      128 / cutlass::sizeof_bits<Output>::value,
      Accumulator,
      Accumulator>;
  using Gemm = cutlass::gemm::device::Gemm<
      Element,
      cutlass::layout::RowMajor,
      Element,
      cutlass::layout::ColumnMajor,
      Output,
      cutlass::layout::RowMajor,
      Accumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 128>,
      cutlass::gemm::GemmShape<64, 64, 128>,
      cutlass::gemm::GemmShape<16, 8, 64>,
      Epilogue,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      3>;

  auto* a = reinterpret_cast<Element const*>(a_packed.data_ptr<uint8_t>());
  // B is physically [N, K] row-major, exactly the same byte layout as B^T [K, N]
  // column-major. This avoids an unpack or transpose before the tensor-core GEMM.
  auto* b = reinterpret_cast<Element const*>(b_packed.data_ptr<uint8_t>());
  auto* d = output.data_ptr<int32_t>();
  typename Gemm::Arguments arguments(
      {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k)},
      {a, static_cast<int>(k)},
      {b, static_cast<int>(k)},
      {d, static_cast<int>(n)},
      {d, static_cast<int>(n)},
      {1, 0});
  Gemm gemm;
  cutlass::Status status = gemm(arguments, nullptr, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS INT4 GEMM failed: ", cutlassGetStatusString(status));
  return output;
}

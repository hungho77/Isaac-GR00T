// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

torch::Tensor holoq_int4_mm_cuda(torch::Tensor a_packed, torch::Tensor b_packed, int64_t k);

torch::Tensor int4_mm(torch::Tensor a_packed, torch::Tensor b_packed, int64_t k) {
  TORCH_CHECK(a_packed.is_cuda() && b_packed.is_cuda(), "HoloQ INT4 GEMM is CUDA-only");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8, "A must be uint8 packed INT4");
  TORCH_CHECK(b_packed.scalar_type() == torch::kUInt8, "B must be uint8 packed INT4");
  TORCH_CHECK(a_packed.dim() == 2 && b_packed.dim() == 2, "A and B must be rank-2");
  TORCH_CHECK(a_packed.size(1) == b_packed.size(1), "packed K dimensions must match");
  TORCH_CHECK(k == a_packed.size(1) * 2 && k % 64 == 0, "K must match storage and align to 64");
  return holoq_int4_mm_cuda(a_packed.contiguous(), b_packed.contiguous(), k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int4_mm", &int4_mm, "Packed signed INT4 x INT4 GEMM (CUDA/CUTLASS)");
}

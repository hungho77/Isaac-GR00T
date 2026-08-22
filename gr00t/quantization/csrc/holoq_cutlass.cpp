// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include <vector>

torch::Tensor holoq_int4_mm_cuda(torch::Tensor a_packed, torch::Tensor b_packed, int64_t k);
std::vector<torch::Tensor> holoq_quantize_pack_cuda(torch::Tensor input, int64_t padded_k);
torch::Tensor holoq_dequantize_cuda(
    torch::Tensor accumulator,
    torch::Tensor activation_scale,
    torch::Tensor weight_scale,
    torch::Tensor bias,
    int64_t output_dtype);

torch::Tensor int4_mm(torch::Tensor a_packed, torch::Tensor b_packed, int64_t k) {
  TORCH_CHECK(a_packed.is_cuda() && b_packed.is_cuda(), "HoloQ INT4 GEMM is CUDA-only");
  TORCH_CHECK(a_packed.scalar_type() == torch::kUInt8, "A must be uint8 packed INT4");
  TORCH_CHECK(b_packed.scalar_type() == torch::kUInt8, "B must be uint8 packed INT4");
  TORCH_CHECK(a_packed.dim() == 2 && b_packed.dim() == 2, "A and B must be rank-2");
  TORCH_CHECK(a_packed.size(1) == b_packed.size(1), "packed K dimensions must match");
  TORCH_CHECK(k == a_packed.size(1) * 2 && k % 64 == 0, "K must match storage and align to 64");
  return holoq_int4_mm_cuda(a_packed.contiguous(), b_packed.contiguous(), k);
}

std::vector<torch::Tensor> quantize_pack(torch::Tensor input, int64_t padded_k) {
  TORCH_CHECK(input.is_cuda(), "HoloQ activation quantization is CUDA-only");
  TORCH_CHECK(input.dim() == 2, "activation input must be rank-2");
  TORCH_CHECK(input.is_floating_point(), "activation input must be floating point");
  TORCH_CHECK(padded_k >= input.size(1) && padded_k % 64 == 0,
              "padded K must cover the input and align to 64");
  return holoq_quantize_pack_cuda(input.contiguous(), padded_k);
}

torch::Tensor dequantize(
    torch::Tensor accumulator,
    torch::Tensor activation_scale,
    torch::Tensor weight_scale,
    torch::Tensor bias,
    int64_t output_dtype) {
  TORCH_CHECK(accumulator.is_cuda(), "HoloQ dequantization is CUDA-only");
  TORCH_CHECK(accumulator.scalar_type() == torch::kInt32 && accumulator.dim() == 2,
              "accumulator must be a rank-2 int32 CUDA tensor");
  TORCH_CHECK(activation_scale.is_cuda() && weight_scale.is_cuda() && bias.is_cuda(),
              "all scale and bias tensors must be CUDA tensors");
  TORCH_CHECK(activation_scale.scalar_type() == torch::kFloat32 &&
                  weight_scale.scalar_type() == torch::kFloat32,
              "native scales must be float32");
  TORCH_CHECK(output_dtype == 0 || output_dtype == 1 || output_dtype == 2,
              "output dtype code must be 0=float32, 1=float16, or 2=bfloat16");
  return holoq_dequantize_cuda(
      accumulator.contiguous(), activation_scale.contiguous(), weight_scale.contiguous(),
      bias.contiguous(), output_dtype);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int4_mm", &int4_mm, "Packed signed INT4 x INT4 GEMM (CUDA/CUTLASS)");
  m.def("quantize_pack", &quantize_pack,
        "Fused dynamic-per-token signed INT4 quantization and packing (CUDA)");
  m.def("dequantize", &dequantize,
        "Fused INT32 scale, bias, and output conversion (CUDA)");
}

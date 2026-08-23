// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination_clamp.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int kQuantThreads = 256;

__device__ int clamp_int4(int value) {
  return value < -7 ? -7 : (value > 7 ? 7 : value);
}

template <typename scalar_t>
__global__ void quantize_pack_kernel(
    scalar_t const* input,
    uint8_t* packed,
    float* scales,
    int64_t rows,
    int64_t logical_k,
    int64_t padded_k) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  float local_max = 0.0f;
  for (int64_t column = threadIdx.x; column < logical_k; column += blockDim.x) {
    local_max = fmaxf(local_max, fabsf(static_cast<float>(input[row * logical_k + column])));
  }
  __shared__ float reduction[kQuantThreads];
  reduction[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float scale = fmaxf(reduction[0], 1.0e-8f) / 7.0f;
  if (threadIdx.x == 0) {
    scales[row] = scale;
  }
  const int64_t packed_k = padded_k / 2;
  for (int64_t byte_index = threadIdx.x; byte_index < packed_k; byte_index += blockDim.x) {
    const int64_t first = byte_index * 2;
    const int64_t second = first + 1;
    int low = first < logical_k
        ? clamp_int4(static_cast<int>(nearbyintf(
              static_cast<float>(input[row * logical_k + first]) / scale)))
        : 0;
    int high = second < logical_k
        ? clamp_int4(static_cast<int>(nearbyintf(
              static_cast<float>(input[row * logical_k + second]) / scale)))
        : 0;
    packed[row * packed_k + byte_index] =
        static_cast<uint8_t>((low & 0xF) | ((high & 0xF) << 4));
  }
}

template <typename output_t>
__global__ void dequantize_kernel(
    int32_t const* accumulator,
    float const* activation_scale,
    float const* weight_scale,
    float const* bias,
    output_t* output,
    int64_t rows,
    int64_t columns,
    bool has_bias) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * columns;
  if (index >= total) {
    return;
  }
  const int64_t row = index / columns;
  const int64_t column = index % columns;
  float value = static_cast<float>(accumulator[index]) * activation_scale[row] * weight_scale[column];
  if (has_bias) {
    value += bias[column];
  }
  output[index] = static_cast<output_t>(value);
}

}  // namespace

std::vector<torch::Tensor> holoq_quantize_pack_cuda(torch::Tensor input, int64_t padded_k) {
  const at::cuda::CUDAGuard device_guard(input.device());
  const int64_t rows = input.size(0);
  const int64_t logical_k = input.size(1);
  auto packed = torch::empty({rows, padded_k / 2}, input.options().dtype(torch::kUInt8));
  auto scales = torch::empty({rows}, input.options().dtype(torch::kFloat32));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "holoq_quantize_pack_cuda",
      [&] {
        quantize_pack_kernel<scalar_t><<<rows, kQuantThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<scalar_t>(), packed.data_ptr<uint8_t>(), scales.data_ptr<float>(),
            rows, logical_k, padded_k);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scales};
}

torch::Tensor holoq_dequantize_cuda(
    torch::Tensor accumulator,
    torch::Tensor activation_scale,
    torch::Tensor weight_scale,
    torch::Tensor bias,
    int64_t output_dtype) {
  const at::cuda::CUDAGuard device_guard(accumulator.device());
  const int64_t rows = accumulator.size(0);
  const int64_t columns = accumulator.size(1);
  TORCH_CHECK(activation_scale.numel() == rows, "activation scale must contain one value per row");
  TORCH_CHECK(weight_scale.numel() == columns, "weight scale must contain one value per output");
  TORCH_CHECK(bias.numel() == 0 || bias.numel() == columns, "bias must be empty or match output width");
  const auto dtype = output_dtype == 0
      ? torch::kFloat32
      : (output_dtype == 1 ? torch::kFloat16 : torch::kBFloat16);
  auto output = torch::empty({rows, columns}, accumulator.options().dtype(dtype));
  const int blocks = static_cast<int>((rows * columns + 255) / 256);
  if (output_dtype == 0) {
    dequantize_kernel<float><<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        accumulator.data_ptr<int32_t>(), activation_scale.data_ptr<float>(),
        weight_scale.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        rows, columns, bias.numel() != 0);
  } else if (output_dtype == 1) {
    dequantize_kernel<c10::Half><<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        accumulator.data_ptr<int32_t>(), activation_scale.data_ptr<float>(),
        weight_scale.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<c10::Half>(),
        rows, columns, bias.numel() != 0);
  } else {
    dequantize_kernel<c10::BFloat16><<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        accumulator.data_ptr<int32_t>(), activation_scale.data_ptr<float>(),
        weight_scale.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<c10::BFloat16>(),
        rows, columns, bias.numel() != 0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

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

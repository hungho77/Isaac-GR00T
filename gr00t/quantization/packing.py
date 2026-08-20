# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical building blocks for paper-compatible GR00T-N1.7 W4A4 packs."""

from __future__ import annotations

import hashlib
import math

import torch


PACK_FORMAT_VERSION = 1
WEIGHT_BITS = 4
ACTIVATION_BITS = 4
SIGNED_QMAX = 7


def stable_layer_seed(base_seed: int, layer_name: str) -> int:
    digest = hashlib.sha256(layer_name.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:8], "little")) % (2**63 - 1)


def normalized_hadamard(size: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return a normalized Sylvester Hadamard matrix."""

    if size <= 0 or size & (size - 1):
        raise ValueError(f"Hadamard size must be a positive power of two, got {size}")
    matrix = torch.ones((1, 1), dtype=torch.float32, device=device)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), dim=1), torch.cat((matrix, -matrix), dim=1)),
            dim=0,
        )
    return matrix / math.sqrt(size)


def zigzag_weight_permutation(weight: torch.Tensor, block_size: int) -> torch.Tensor:
    """Spread descending input-channel weight energy across contiguous blocks."""

    if weight.ndim != 2:
        raise ValueError(f"Expected a 2-D weight, got shape {tuple(weight.shape)}")
    in_features = weight.shape[1]
    if in_features % block_size:
        raise ValueError(f"in_features={in_features} must be divisible by block_size={block_size}")
    order = torch.argsort(weight.float().square().sum(dim=0), descending=True).tolist()
    block_count = in_features // block_size
    buckets: list[list[int]] = [[] for _ in range(block_count)]
    for rank, channel in enumerate(order):
        cycle, offset = divmod(rank, block_count)
        block = offset if cycle % 2 == 0 else block_count - 1 - offset
        buckets[block].append(channel)
    if any(len(bucket) != block_size for bucket in buckets):
        raise AssertionError("Zigzag assignment produced uneven blocks")
    return torch.tensor([channel for bucket in buckets for channel in bucket], dtype=torch.long)


def build_svd_hadamard_transform(
    weight: torch.Tensor,
    *,
    block_size: int = 64,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build zigzag permutation, randomized SVD-Hadamard blocks, and rotated weight.

    PyTorch stores a linear weight as ``[out, in]``. For row-vector inputs the
    invariant transform is ``X' = X[:, perm] R`` and
    ``W' = W[:, perm] R``, so ``X' W'^T == X W^T`` before quantization.
    """

    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError(f"block_size must be a positive power of two, got {block_size}")
    weight_fp = weight.detach().to(dtype=torch.float32)
    permutation = zigzag_weight_permutation(weight_fp, block_size).to(weight_fp.device)
    permuted = weight_fp.index_select(1, permutation)
    hadamard = normalized_hadamard(block_size, device=weight_fp.device)
    generator = torch.Generator(device=weight_fp.device)
    generator.manual_seed(seed)

    rotations = []
    rotated_blocks = []
    for start in range(0, permuted.shape[1], block_size):
        block = permuted[:, start : start + block_size]
        # The paper's W is [input, output], so the left singular vectors come
        # from the transpose of a PyTorch Linear weight block.
        u, _, _ = torch.linalg.svd(block.T, full_matrices=True)
        signs = torch.randint(
            0,
            2,
            (block_size,),
            generator=generator,
            device=weight_fp.device,
            dtype=torch.int64,
        ).float()
        signs = signs.mul_(2).sub_(1)
        randomized_hadamard = signs[:, None] * hadamard
        rotation = u @ randomized_hadamard
        rotations.append(rotation)
        rotated_blocks.append(block @ rotation)

    return (
        permutation.cpu(),
        torch.stack(rotations).cpu(),
        torch.cat(rotated_blocks, dim=1).cpu(),
    )


def apply_input_transform(
    inputs: torch.Tensor, permutation: torch.Tensor, rotations: torch.Tensor
) -> torch.Tensor:
    """Apply a packed block-diagonal input transform to the final dimension."""

    permutation = permutation.to(device=inputs.device)
    rotations = rotations.to(device=inputs.device, dtype=inputs.dtype)
    block_count, block_size, _ = rotations.shape
    if inputs.shape[-1] != block_count * block_size:
        raise ValueError(
            f"Input width {inputs.shape[-1]} does not match transform width "
            f"{block_count * block_size}"
        )
    shape = inputs.shape
    flattened = inputs.reshape(-1, shape[-1]).index_select(1, permutation)
    blocked = flattened.reshape(-1, block_count, block_size).transpose(0, 1)
    transformed = torch.bmm(blocked, rotations).transpose(0, 1)
    return transformed.reshape(*shape)


def apply_weight_transform(
    weight: torch.Tensor, permutation: torch.Tensor, rotations: torch.Tensor
) -> torch.Tensor:
    """Apply the matching transform to a PyTorch ``[out, in]`` weight."""

    permutation = permutation.to(device=weight.device)
    rotations = rotations.to(device=weight.device, dtype=weight.dtype)
    block_count, block_size, _ = rotations.shape
    if weight.shape[1] != block_count * block_size:
        raise ValueError("Weight width does not match rotation blocks")
    permuted = weight.index_select(1, permutation)
    blocked = permuted.reshape(weight.shape[0], block_count, block_size).transpose(0, 1)
    return torch.bmm(blocked, rotations).transpose(0, 1).reshape_as(weight)


def symmetric_quantize_per_output_channel(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight to signed symmetric INT4 values in [-7, 7]."""

    weight_fp = weight.float()
    scale = weight_fp.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / SIGNED_QMAX
    quantized = torch.round(weight_fp / scale).clamp(-SIGNED_QMAX, SIGNED_QMAX)
    return quantized.to(torch.int8), scale


def symmetric_fake_quantize(inputs: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scale = scale.to(device=inputs.device, dtype=inputs.dtype).clamp_min(1e-8)
    quantized = torch.round(inputs / scale).clamp(-SIGNED_QMAX, SIGNED_QMAX)
    return quantized * scale


def gptq_quantize_blockwise(
    weight: torch.Tensor,
    hessian_blocks: torch.Tensor,
    *,
    block_size: int = 128,
    damping: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPTQ-style error compensation within 128-column Hessian blocks.

    Scales remain per output channel as required by HoloQ-VLA. The supplied
    Hessian must already be measured after the permutation and rotation.
    """

    if damping < 0:
        raise ValueError(f"damping must be non-negative, got {damping}")
    weight_work = weight.float().clone()
    if weight_work.shape[1] % block_size:
        raise ValueError(
            f"Weight width {weight_work.shape[1]} is not divisible by GPTQ block {block_size}"
        )
    expected_shape = (weight_work.shape[1] // block_size, block_size, block_size)
    if tuple(hessian_blocks.shape) != expected_shape:
        raise ValueError(
            f"Expected Hessian blocks {expected_shape}, got {tuple(hessian_blocks.shape)}"
        )

    scale = weight_work.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / SIGNED_QMAX
    quantized = torch.empty_like(weight_work, dtype=torch.int8)
    for block_index, start in enumerate(range(0, weight_work.shape[1], block_size)):
        stop = start + block_size
        current = weight_work[:, start:stop].clone()
        hessian = hessian_blocks[block_index].to(current.device, torch.float32).clone()
        diagonal = torch.diagonal(hessian)
        dead = diagonal <= 0
        if dead.any():
            hessian[dead, dead] = 1.0
            current[:, dead] = 0.0
        damp = damping * torch.diagonal(hessian).mean().clamp_min(1e-8)
        hessian.diagonal().add_(damp)
        try:
            chol = torch.linalg.cholesky(hessian)
            inverse = torch.cholesky_inverse(chol)
            inverse_factor = torch.linalg.cholesky(inverse, upper=True)
        except torch.linalg.LinAlgError as exc:
            raise ValueError(f"Hessian block {block_index} is not positive definite") from exc

        for column in range(block_size):
            vector = current[:, column]
            q = torch.round(vector / scale[:, 0]).clamp(-SIGNED_QMAX, SIGNED_QMAX)
            quantized[:, start + column] = q.to(torch.int8)
            reconstructed = q * scale[:, 0]
            denom = inverse_factor[column, column].clamp_min(1e-8)
            error = (vector - reconstructed) / denom
            current[:, column:] -= error[:, None] * inverse_factor[column, column:]
    return quantized.cpu(), scale.cpu()

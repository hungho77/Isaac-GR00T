# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict CUTLASS-backed signed W4A4 GEMM loader.

Native mode deliberately has no fake/dequant fallback. The extension fuses
dynamic activation quantization and packing, consumes two signed INT4 values
per byte in CUTLASS, and fuses the INT32 scaling/bias/cast epilogue. The same
native path serves LLM, DiT, ViT, and patch-Conv3d lowering.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import ModuleType

import torch


_EXTENSION: ModuleType | None = None


def _cuda_dependency_include_paths() -> list[Path]:
    """Find CUDA development headers shipped by PyTorch's NVIDIA wheels."""

    torch_file = getattr(torch, "__file__", None)
    if not torch_file:
        return []
    nvidia_root = Path(torch_file).resolve().parent.parent / "nvidia"
    return sorted(path.resolve() for path in nvidia_root.glob("*/include") if path.is_dir())


def _cutlass_root(explicit: str | Path | None = None) -> Path:
    candidates = [
        explicit,
        os.environ.get("HOLOQ_CUTLASS_ROOT"),
        Path.home() / ".cache" / "holoq" / "cutlass-v3.9.2",
        Path("/opt/cutlass"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser().resolve()
        if (path / "include" / "cutlass" / "cutlass.h").is_file():
            return path
    raise RuntimeError(
        "Native HoloQ W4A4 requires CUTLASS headers. Set HOLOQ_CUTLASS_ROOT to a "
        "CUTLASS checkout containing include/cutlass/cutlass.h. Native mode never "
        "falls back to fake quantization."
    )


def load_native_extension(*, cutlass_root: str | Path | None = None) -> ModuleType:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    if not torch.cuda.is_available():
        raise RuntimeError("Native HoloQ W4A4 requires a CUDA device")
    if torch.cuda.get_device_capability() < (8, 0):
        raise RuntimeError("Native HoloQ W4A4 requires an NVIDIA SM80-or-newer GPU")
    root = _cutlass_root(cutlass_root)
    source_root = Path(__file__).resolve().parent / "csrc"
    sources = [source_root / "holoq_cutlass.cpp", source_root / "holoq_cutlass_kernel.cu"]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise RuntimeError(f"Native HoloQ extension sources are missing: {missing}")
    from torch.utils.cpp_extension import load

    cuda_dependency_includes = _cuda_dependency_include_paths()
    identity = hashlib.sha256(
        (
            str(root)
            + "\0"
            + hashlib.sha256((root / "include" / "cutlass" / "cutlass.h").read_bytes()).hexdigest()
            + "\0"
            + str(torch.__version__)
            + "\0"
            + str(torch.version.cuda)
            + "\0"
            + "\0".join(str(path) for path in cuda_dependency_includes)
            + "\0"
            + "\0".join(path.read_text(encoding="utf-8") for path in sources)
        ).encode()
    ).hexdigest()[:12]
    _EXTENSION = load(
        name=f"holoq_cutlass_{identity}",
        sources=[str(path) for path in sources],
        extra_include_paths=[
            str(root / "include"),
            str(root / "tools" / "util" / "include"),
            *(str(path) for path in cuda_dependency_includes),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        with_cuda=True,
        verbose=os.environ.get("HOLOQ_NATIVE_BUILD_VERBOSE", "0") == "1",
    )
    return _EXTENSION


def native_int4_mm(
    activation_packed: torch.Tensor,
    weight_packed: torch.Tensor,
    *,
    logical_k: int,
    cutlass_root: str | Path | None = None,
) -> torch.Tensor:
    """Run packed signed INT4 GEMM and return an ``[M, N]`` INT32 tensor."""

    if activation_packed.dtype != torch.uint8 or weight_packed.dtype != torch.uint8:
        raise ValueError("Native W4A4 GEMM expects uint8 nibble-packed operands")
    if activation_packed.ndim != 2 or weight_packed.ndim != 2:
        raise ValueError("Native W4A4 GEMM operands must be rank-2")
    if activation_packed.shape[1] != weight_packed.shape[1]:
        raise ValueError("Activation and weight packed K dimensions do not match")
    packed_k = activation_packed.shape[1] * 2
    if logical_k <= 0 or logical_k > packed_k or packed_k % 64:
        raise ValueError(
            f"Native W4A4 requires 0 < logical_k <= packed_k and packed_k % 64 == 0; "
            f"got logical_k={logical_k}, packed_k={packed_k}"
        )
    if not activation_packed.is_cuda or not weight_packed.is_cuda:
        raise ValueError("Native W4A4 operands must reside on CUDA")
    extension = load_native_extension(cutlass_root=cutlass_root)
    return extension.int4_mm(
        activation_packed.contiguous(), weight_packed.contiguous(), int(packed_k)
    )


def native_quantize_pack(
    activations: torch.Tensor,
    *,
    padded_k: int,
    cutlass_root: str | Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused CUDA dynamic-per-token quantization and signed INT4 packing."""

    if not activations.is_cuda or activations.ndim != 2 or not activations.is_floating_point():
        raise ValueError("Native activation quantization expects a rank-2 floating CUDA tensor")
    if padded_k < activations.shape[1] or padded_k % 64:
        raise ValueError("padded_k must cover the logical width and align to 64")
    extension = load_native_extension(cutlass_root=cutlass_root)
    packed, scale = extension.quantize_pack(activations.contiguous(), int(padded_k))
    return packed, scale


def native_dequantize(
    accumulator: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    output_dtype: torch.dtype,
    cutlass_root: str | Path | None = None,
) -> torch.Tensor:
    """Fused CUDA scaling, optional bias, and output dtype conversion."""

    dtype_codes = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
    if output_dtype not in dtype_codes:
        raise ValueError(f"Unsupported native output dtype: {output_dtype}")
    device = accumulator.device
    empty_bias = torch.empty(0, dtype=torch.float32, device=device)
    extension = load_native_extension(cutlass_root=cutlass_root)
    return extension.dequantize(
        accumulator,
        activation_scale.to(device=device, dtype=torch.float32).reshape(-1),
        weight_scale.to(device=device, dtype=torch.float32).reshape(-1),
        empty_bias if bias is None else bias.to(device=device, dtype=torch.float32).reshape(-1),
        dtype_codes[output_dtype],
    )


def native_backend_info(*, cutlass_root: str | Path | None = None) -> dict[str, object]:
    return {
        "backend": "cutlass-int4-tensorcore",
        "packed_storage": "signed-int4-low-high-nibble",
        "accumulator": "int32",
        "activation_quantization": "fused-cuda-dynamic-per-token",
        "activation_packing": "fused-cuda-low-high-nibble",
        "dequantization_epilogue": "fused-cuda-scale-bias-cast",
        "cuda_available": torch.cuda.is_available(),
        "compute_capability": (
            list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
        ),
        "cutlass_root": (str(_cutlass_root(cutlass_root)) if torch.cuda.is_available() else None),
        "silent_fallback": False,
    }

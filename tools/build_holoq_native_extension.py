# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precompile and validate the strict CUTLASS W4A4 extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from gr00t.quantization.native import load_native_extension, native_backend_info
from gr00t.quantization.packing import pack_signed_int4
import torch


CUTLASS_REPOSITORY = "https://github.com/NVIDIA/cutlass.git"
CUTLASS_REVISION = "v3.9.2"


def _validate_cutlass_revision(target: Path) -> None:
    if not (target / ".git").is_dir():
        raise RuntimeError(f"Strict build requires a Git-backed CUTLASS checkout: {target}")
    result = subprocess.run(
        ["git", "-C", str(target), "describe", "--tags", "--exact-match", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    )
    if result.stdout.strip() != CUTLASS_REVISION:
        raise RuntimeError(
            f"CUTLASS checkout is {result.stdout.strip()!r}; expected {CUTLASS_REVISION!r}"
        )
    status = subprocess.run(
        ["git", "-C", str(target), "status", "--porcelain"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("Strict build refuses a locally modified CUTLASS checkout")


def _fetch_cutlass(target: Path) -> None:
    header = target / "include" / "cutlass" / "cutlass.h"
    if header.is_file():
        _validate_cutlass_revision(target)
        return
    if target.exists():
        raise RuntimeError(f"Refusing to overwrite incomplete CUTLASS directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            CUTLASS_REVISION,
            "--depth",
            "1",
            CUTLASS_REPOSITORY,
            str(target),
        ],
        check=True,
    )
    if not header.is_file():
        raise RuntimeError(f"CUTLASS checkout is missing expected header: {header}")
    _validate_cutlass_revision(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutlass-root", default=None)
    parser.add_argument("--fetch-cutlass", action="store_true")
    args = parser.parse_args()
    cutlass_root = (
        Path(args.cutlass_root).expanduser().resolve()
        if args.cutlass_root
        else Path.home() / ".cache" / "holoq" / f"cutlass-{CUTLASS_REVISION}"
    )
    if args.fetch_cutlass:
        _fetch_cutlass(cutlass_root)
    extension = load_native_extension(cutlass_root=cutlass_root)
    if not hasattr(extension, "int4_mm"):
        raise RuntimeError("Built native extension does not expose int4_mm")
    generator = torch.Generator(device="cuda").manual_seed(17)
    activation = torch.randint(
        -7, 8, (37, 128), dtype=torch.int8, device="cuda", generator=generator
    )
    weight = torch.randint(-7, 8, (72, 128), dtype=torch.int8, device="cuda", generator=generator)
    activation_packed, _ = pack_signed_int4(activation, pad_to=64)
    weight_packed, _ = pack_signed_int4(weight, pad_to=64)
    actual = extension.int4_mm(activation_packed, weight_packed, 128)
    expected = activation.int() @ weight.int().T
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    floating = torch.randn((37, 113), dtype=torch.bfloat16, device="cuda", generator=generator)
    packed, scales = extension.quantize_pack(floating, 128)
    reference_scales = floating.float().abs().amax(dim=1).clamp_min(1e-8) / 7
    reference_codes = torch.round(floating.float() / reference_scales[:, None]).clamp(-7, 7)
    reference_codes = reference_codes.to(torch.int8)
    reference_packed, _ = pack_signed_int4(reference_codes, pad_to=128)
    torch.testing.assert_close(scales, reference_scales, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(packed, reference_packed, rtol=0, atol=0)
    output_scales = torch.rand(72, dtype=torch.float32, device="cuda", generator=generator)
    bias = torch.rand(72, dtype=torch.float32, device="cuda", generator=generator)
    fused = extension.dequantize(actual, scales[:37], output_scales, bias, 2)
    reference = (
        actual.float() * scales[:37, None] * output_scales[None, :] + bias[None, :]
    ).bfloat16()
    torch.testing.assert_close(fused, reference, rtol=0, atol=0)
    print(json.dumps(native_backend_info(cutlass_root=cutlass_root), indent=2))


if __name__ == "__main__":
    main()

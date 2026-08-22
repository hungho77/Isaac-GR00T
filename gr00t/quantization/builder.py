# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build suite-specific packed W4A4 artifacts for fake and native execution."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from .calibration import CALIBRATION_FORMAT_VERSION
from .packing import (
    ACTIVATION_BITS,
    PACK_FORMAT_VERSION,
    WEIGHT_BITS,
    apply_weight_transform,
    gptq_quantize_blockwise,
    pack_signed_int4,
    symmetric_quantize_per_output_channel,
)
from .runtime import model_config_sha256, tensor_record_sha256
from .scope import discover_n1d7_targets, normalize_scopes, validate_n1d7_scope


def _load_artifact(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover
        return torch.load(path, map_location="cpu")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weight_matrix(module: nn.Module) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        return module.weight.detach().float()
    if isinstance(module, nn.Conv3d):
        return module.weight.detach().float().flatten(1)
    raise TypeError(f"Unsupported quantization target {type(module).__name__}")


def build_holoq_pack(
    model: nn.Module,
    *,
    calibration_path: str | Path,
    output_path: str | Path,
    suite: str,
    checkpoint: str,
    checkpoint_revision: str,
    source_revision: str,
    scopes: Iterable[str] | str | None = None,
    include_vit_mergers: bool | None = None,
    include_vit_patch_embed: bool | None = None,
    dit_activation_granularity: str = "static-per-step-per-channel",
) -> Path:
    """Build a v2 nibble-packed artifact consumable by fake or native runtime.

    Native execution requires dynamic-per-token DiT activations because the
    calibrated per-channel scale varies inside GEMM's K reduction. Fake mode
    retains the paper-reference static per-step/per-channel option.
    """

    if dit_activation_granularity not in {
        "static-per-step-per-channel",
        "dynamic-per-token",
    }:
        raise ValueError("Unsupported DiT activation granularity")
    calibration_path = Path(calibration_path)
    calibration = _load_artifact(calibration_path)
    calibration_manifest = calibration.get("manifest", {})
    stats = calibration.get("layers", {})
    if calibration_manifest.get("format_version") not in {1, CALIBRATION_FORMAT_VERSION}:
        raise ValueError("Unsupported calibration artifact format")
    if calibration_manifest.get("algorithm") != "holoq-vla-calibration":
        raise ValueError("Not a HoloQ-VLA calibration artifact")
    if calibration_manifest.get("suite") != suite:
        raise ValueError(
            f"Calibration suite {calibration_manifest.get('suite')!r} != requested {suite!r}"
        )
    config_hash = model_config_sha256(model)
    if calibration_manifest.get("config_sha256") != config_hash:
        raise ValueError("Calibration config hash does not match the loaded checkpoint")

    selected_scopes = normalize_scopes(scopes or calibration_manifest.get("scopes"))
    if include_vit_mergers is None:
        include_vit_mergers = bool(calibration_manifest.get("include_vit_mergers", False))
    if include_vit_patch_embed is None:
        include_vit_patch_embed = bool(calibration_manifest.get("include_vit_patch_embed", False))
    targets = discover_n1d7_targets(
        model,
        scopes=selected_scopes,
        include_vit_mergers=include_vit_mergers,
        include_vit_patch_embed=include_vit_patch_embed,
    )
    summary = validate_n1d7_scope(
        targets,
        expected_llm_layers=int(calibration_manifest.get("llm_layers", 0)),
        expected_dit_layers=int(calibration_manifest.get("dit_layers", 0)),
        expected_vit_layers=int(calibration_manifest.get("vit_layers", 0)),
        scopes=selected_scopes,
    )
    target_names = {target.name for target in targets}
    if set(stats) != target_names:
        missing = sorted(target_names - set(stats))
        extra = sorted(set(stats) - target_names)
        raise ValueError(
            f"Calibration coverage mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )

    gptq_block_size = int(calibration_manifest["gptq_block_size"])
    damping = float(calibration_manifest["gptq_damping"])
    records: dict[str, dict[str, Any]] = {}
    for target in targets:
        stat = stats[target.name]
        weight = _weight_matrix(target.module)
        transform = str(stat.get("transform", "svd-hadamard"))
        permutation = stat.get("permutation")
        rotations = stat.get("rotation_blocks")
        if transform == "identity":
            rotated_weight = weight
        else:
            rotations = rotations.float()
            rotated_weight = apply_weight_transform(weight, permutation, rotations)

        if target.module_kind == "conv3d_patch":
            weight_q, weight_scale = symmetric_quantize_per_output_channel(rotated_weight)
            solver = "rtn"
        elif target.scope in {"llm", "vit"}:
            weight_q, weight_scale = gptq_quantize_blockwise(
                rotated_weight,
                stat["hessian_blocks"],
                block_size=gptq_block_size,
                damping=damping,
            )
            solver = "gptq"
        else:
            weight_q, weight_scale = symmetric_quantize_per_output_channel(rotated_weight)
            solver = "rtn"

        activation_granularity = (
            dit_activation_granularity if target.scope == "dit" else "dynamic-per-token"
        )
        activation_scale = (
            stat["activation_scale"].float()
            if activation_granularity == "static-per-step-per-channel"
            else None
        )
        weight_packed, logical_width = pack_signed_int4(weight_q.cpu(), pad_to=64)
        record = {
            "scope": target.scope,
            "module_kind": target.module_kind,
            "projection": target.projection,
            "solver": solver,
            "weight_packed": weight_packed,
            "weight_shape": [int(weight_q.shape[0]), int(logical_width)],
            "padded_in_features": int(weight_packed.shape[1] * 2),
            "weight_storage": "signed-int4-low-high-nibble",
            "weight_scale": weight_scale.float().cpu(),
            "transform": transform,
            "permutation": None if permutation is None else permutation.long().cpu(),
            "rotation_blocks": None if rotations is None else rotations.half().cpu(),
            "activation_granularity": activation_granularity,
            "activation_scale": activation_scale,
            "bias": (
                None
                if getattr(target.module, "bias", None) is None
                else target.module.bias.detach().half().cpu()
            ),
        }
        record["sha256"] = tensor_record_sha256(record)
        records[target.name] = record

    native_compatible = all(
        record["activation_granularity"] == "dynamic-per-token" for record in records.values()
    )
    manifest = {
        "format_version": PACK_FORMAT_VERSION,
        "algorithm": "holoq-vla",
        "model_type": "Gr00tN1d7",
        "suite": suite,
        "checkpoint": checkpoint,
        "checkpoint_revision": checkpoint_revision,
        "source_revision": source_revision,
        "config_sha256": config_hash,
        "calibration_sha256": _file_sha256(calibration_path),
        "calibration_run_id": calibration_manifest["run_id"],
        "weight_bits": WEIGHT_BITS,
        "activation_bits": ACTIVATION_BITS,
        "signed_range": [-7, 7],
        "weight_storage": "two-signed-int4-values-per-byte",
        "native_k_alignment": 64,
        "native_accumulator": "int32",
        "runtime_compatible_backends": ["fake", "native"] if native_compatible else ["fake"],
        "scopes": list(selected_scopes),
        "include_vit_mergers": include_vit_mergers,
        "include_vit_patch_embed": include_vit_patch_embed,
        "high_precision_vision_ops": [
            "embedding",
            "layernorm",
            "rotary",
            "attention_qk_av",
            "softmax",
        ],
        "num_inference_timesteps": int(calibration_manifest["num_inference_timesteps"]),
        "llm_layers": summary.llm_layers,
        "dit_layers": summary.dit_layers,
        "vit_layers": summary.vit_layers,
        "llm_linears": summary.llm_linears,
        "dit_linears": summary.dit_linears,
        "vit_linears": summary.vit_linears,
        "vit_patch_convs": summary.vit_patch_convs,
        "total_linears": summary.total_linears,
        "total_quantized_modules": summary.total_modules,
        "rotation": calibration_manifest["rotation"],
        "rotation_block_size": calibration_manifest["rotation_block_size"],
        "permutation": calibration_manifest["permutation"],
        "llm_solver": "gptq" if "llm" in selected_scopes else None,
        "vit_solver": "gptq-linear-rtn-patch" if "vit" in selected_scopes else None,
        "gptq_block_size": gptq_block_size,
        "gptq_damping": damping,
        "dit_solver": "rtn" if "dit" in selected_scopes else None,
        "dit_activation_granularity": dit_activation_granularity,
        "activation_percentile": calibration_manifest["activation_percentile"],
        "seed": calibration_manifest["seed"],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"manifest": manifest, "layers": records}, output_path)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        _file_sha256(output_path) + "\n", encoding="ascii"
    )
    return output_path

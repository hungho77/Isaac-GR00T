# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a strict suite-specific HoloQ W4A4 pack from calibration statistics."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .calibration import CALIBRATION_FORMAT_VERSION
from .packing import (
    ACTIVATION_BITS,
    PACK_FORMAT_VERSION,
    WEIGHT_BITS,
    apply_weight_transform,
    gptq_quantize_blockwise,
    symmetric_quantize_per_output_channel,
)
from .runtime import model_config_sha256, tensor_record_sha256
from .scope import discover_n1d7_targets, validate_n1d7_scope


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


def build_holoq_pack(
    model: nn.Module,
    *,
    calibration_path: str | Path,
    output_path: str | Path,
    suite: str,
    checkpoint: str,
    checkpoint_revision: str,
    source_revision: str,
) -> Path:
    """Build LLM SVD-Hadamard+GPTQ and DiT SVD-Hadamard+RTN records."""

    calibration_path = Path(calibration_path)
    calibration = _load_artifact(calibration_path)
    calibration_manifest = calibration.get("manifest", {})
    stats = calibration.get("layers", {})
    if calibration_manifest.get("format_version") != CALIBRATION_FORMAT_VERSION:
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

    targets = discover_n1d7_targets(model)
    summary = validate_n1d7_scope(
        targets,
        expected_llm_layers=int(calibration_manifest["llm_layers"]),
        expected_dit_layers=int(calibration_manifest["dit_layers"]),
    )
    target_names = {target.name for target in targets}
    if set(stats) != target_names:
        raise ValueError("Calibration layer coverage does not exactly match the model")

    gptq_block_size = int(calibration_manifest["gptq_block_size"])
    damping = float(calibration_manifest["gptq_damping"])
    records: dict[str, dict[str, Any]] = {}
    for target in targets:
        stat = stats[target.name]
        permutation = stat["permutation"]
        rotations = stat["rotation_blocks"].float()
        rotated_weight = apply_weight_transform(
            target.module.weight.detach().float(), permutation, rotations
        )
        if target.scope == "llm":
            weight_q, weight_scale = gptq_quantize_blockwise(
                rotated_weight,
                stat["hessian_blocks"],
                block_size=gptq_block_size,
                damping=damping,
            )
            solver = "gptq"
            activation_scale = None
        else:
            weight_q, weight_scale = symmetric_quantize_per_output_channel(rotated_weight)
            solver = "rtn"
            activation_scale = stat["activation_scale"].float()
        record = {
            "scope": target.scope,
            "solver": solver,
            "weight_q": weight_q.cpu(),
            "weight_scale": weight_scale.float().cpu(),
            "permutation": permutation.long().cpu(),
            "rotation_blocks": rotations.half().cpu(),
            "activation_scale": activation_scale,
        }
        record["sha256"] = tensor_record_sha256(record)
        records[target.name] = record

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
        "num_inference_timesteps": int(calibration_manifest["num_inference_timesteps"]),
        "llm_layers": summary.llm_layers,
        "dit_layers": summary.dit_layers,
        "llm_linears": summary.llm_linears,
        "dit_linears": summary.dit_linears,
        "total_linears": summary.total_linears,
        "rotation": calibration_manifest["rotation"],
        "rotation_block_size": calibration_manifest["rotation_block_size"],
        "permutation": calibration_manifest["permutation"],
        "llm_solver": "gptq",
        "gptq_block_size": gptq_block_size,
        "gptq_damping": damping,
        "dit_solver": "rtn",
        "activation_percentile": calibration_manifest["activation_percentile"],
        "seed": calibration_manifest["seed"],
        "runtime": "fake-quant-reference",
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"manifest": manifest, "layers": records}, output_path)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        _file_sha256(output_path) + "\n", encoding="ascii"
    )
    return output_path

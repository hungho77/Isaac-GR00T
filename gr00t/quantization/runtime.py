# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict fake-quant runtime for GR00T-N1.7 HoloQ W4A4 packs.

This path validates numerical fidelity. It stores INT4 values in int8 tensors
and dequantizes before ``F.linear``; it does not claim native INT4 latency or
memory savings. A native W4A4 kernel can consume the same validated pack later.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .context import get_dit_quant_step, get_dit_quant_total_steps
from .packing import (
    ACTIVATION_BITS,
    PACK_FORMAT_VERSION,
    SIGNED_QMAX,
    WEIGHT_BITS,
    apply_input_transform,
    symmetric_fake_quantize,
)
from .scope import ScopeSummary, discover_n1d7_targets, validate_n1d7_scope


def model_config_sha256(model: nn.Module) -> str:
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "to_dict"):
        raise ValueError("Model does not expose a serializable Hugging Face config")
    payload = json.dumps(config.to_dict(), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tensor_record_sha256(record: dict[str, Any]) -> str:
    """Hash the tensor-bearing part of one layer record deterministically."""

    digest = hashlib.sha256()
    for key in sorted(record):
        if key == "sha256":
            continue
        value = record[key]
        digest.update(key.encode("utf-8"))
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


class HoloQLinear(nn.Module):
    """Numerical W4A4 replacement for one paper-covered ``nn.Linear``."""

    def __init__(self, linear: nn.Linear, *, name: str, record: dict[str, Any]) -> None:
        super().__init__()
        self.name = name
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.scope = str(record["scope"])
        self.solver = str(record["solver"])

        weight_q = record["weight_q"]
        weight_scale = record["weight_scale"]
        permutation = record["permutation"]
        rotations = record["rotation_blocks"]
        expected_weight_shape = (self.out_features, self.in_features)
        if tuple(weight_q.shape) != expected_weight_shape:
            raise ValueError(
                f"{name}: weight_q shape {tuple(weight_q.shape)} != {expected_weight_shape}"
            )
        if weight_q.dtype != torch.int8:
            raise ValueError(f"{name}: weight_q must use int8 storage, got {weight_q.dtype}")
        if int(weight_q.min()) < -SIGNED_QMAX or int(weight_q.max()) > SIGNED_QMAX:
            raise ValueError(f"{name}: weight_q contains values outside [-7, 7]")
        if tuple(weight_scale.shape) != (self.out_features, 1):
            raise ValueError(
                f"{name}: weight_scale shape must be {(self.out_features, 1)}, "
                f"got {tuple(weight_scale.shape)}"
            )
        if tuple(permutation.shape) != (self.in_features,):
            raise ValueError(f"{name}: invalid permutation shape {tuple(permutation.shape)}")
        if rotations.ndim != 3 or rotations.shape[1] != rotations.shape[2]:
            raise ValueError(f"{name}: rotation_blocks must have shape [blocks, B, B]")
        if rotations.shape[0] * rotations.shape[1] != self.in_features:
            raise ValueError(f"{name}: rotation blocks do not cover all input channels")

        self.register_buffer("weight_q", weight_q.detach().cpu())
        self.register_buffer("weight_scale", weight_scale.detach().float().cpu())
        self.register_buffer("permutation", permutation.detach().long().cpu())
        self.register_buffer("rotation_blocks", rotations.detach().float().cpu())
        if linear.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", linear.bias.detach().clone())

        activation_scale = record.get("activation_scale")
        if self.scope == "dit":
            if not isinstance(activation_scale, torch.Tensor):
                raise ValueError(f"{name}: DiT record is missing its per-step activation table")
            if activation_scale.ndim != 2 or activation_scale.shape[1] != self.in_features:
                raise ValueError(
                    f"{name}: activation_scale must have shape [steps, {self.in_features}]"
                )
            self.register_buffer("activation_scale", activation_scale.detach().float().cpu())
        elif self.scope == "llm":
            if activation_scale is not None:
                raise ValueError(f"{name}: LLM activations must use dynamic per-token scales")
            self.activation_scale = None
        else:
            raise ValueError(f"{name}: unsupported quantization scope {self.scope!r}")

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"scope={self.scope}, solver={self.solver}, W4A4"
        )

    def _quantize_activation(self, transformed: torch.Tensor) -> torch.Tensor:
        if self.scope == "llm":
            scale = transformed.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
            return symmetric_fake_quantize(transformed, scale / SIGNED_QMAX)

        step = get_dit_quant_step()
        total_steps = get_dit_quant_total_steps()
        if step is None or total_steps is None:
            raise RuntimeError(
                f"{self.name}: DiT W4A4 inference requires an active denoising-step context"
            )
        table_steps = self.activation_scale.shape[0]
        if total_steps != table_steps:
            raise RuntimeError(
                f"{self.name}: runtime uses {total_steps} denoising steps but pack has "
                f"{table_steps}; build a matching pack instead of falling back"
            )
        return symmetric_fake_quantize(transformed, self.activation_scale[step])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        transformed = apply_input_transform(inputs, self.permutation, self.rotation_blocks)
        quantized_inputs = self._quantize_activation(transformed)
        weight = self.weight_q.to(device=inputs.device, dtype=inputs.dtype)
        weight = weight * self.weight_scale.to(device=inputs.device, dtype=inputs.dtype)
        bias = None if self.bias is None else self.bias.to(device=inputs.device, dtype=inputs.dtype)
        return F.linear(quantized_inputs, weight, bias)


def _parent_and_attribute(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def load_holoq_pack(path: str | Path) -> dict[str, Any]:
    pack_path = Path(path)
    if not pack_path.is_file():
        raise FileNotFoundError(f"HoloQ pack not found: {pack_path}")
    checksum_path = pack_path.with_suffix(pack_path.suffix + ".sha256")
    if not checksum_path.is_file():
        raise FileNotFoundError(f"HoloQ pack checksum sidecar not found: {checksum_path}")
    expected_checksum = checksum_path.read_text(encoding="ascii").strip().split()[0]
    digest = hashlib.sha256()
    with pack_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_checksum:
        raise ValueError(f"HoloQ pack file checksum mismatch: {pack_path}")
    try:
        pack = torch.load(pack_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        pack = torch.load(pack_path, map_location="cpu")
    if not isinstance(pack, dict):
        raise ValueError("HoloQ pack root must be a dictionary")
    return pack


def apply_holoq_pack(
    model: nn.Module, path: str | Path, *, expected_suite: str | None = None
) -> ScopeSummary:
    """Validate and apply a complete pack; partial/fallback loading is forbidden."""

    pack = load_holoq_pack(path)
    manifest = pack.get("manifest")
    records = pack.get("layers")
    if not isinstance(manifest, dict) or not isinstance(records, dict):
        raise ValueError("Pack must contain dictionary fields 'manifest' and 'layers'")
    if manifest.get("format_version") != PACK_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported pack format {manifest.get('format_version')}; "
            f"expected {PACK_FORMAT_VERSION}"
        )
    if manifest.get("algorithm") != "holoq-vla":
        raise ValueError(f"Unsupported pack algorithm {manifest.get('algorithm')!r}")
    if manifest.get("model_type") != "Gr00tN1d7":
        raise ValueError(f"Pack model_type must be Gr00tN1d7, got {manifest.get('model_type')}")
    if manifest.get("suite") not in {"object", "spatial", "goal", "long"}:
        raise ValueError(f"Invalid LIBERO suite in pack: {manifest.get('suite')!r}")
    if expected_suite is not None and manifest.get("suite") != expected_suite:
        raise ValueError(
            f"Pack suite {manifest.get('suite')!r} does not match requested {expected_suite!r}"
        )
    if (
        manifest.get("weight_bits") != WEIGHT_BITS
        or manifest.get("activation_bits") != ACTIVATION_BITS
    ):
        raise ValueError("Only uniform W4A4 packs are supported")

    targets = discover_n1d7_targets(model)
    summary = validate_n1d7_scope(
        targets,
        expected_llm_layers=int(manifest["llm_layers"]),
        expected_dit_layers=int(manifest["dit_layers"]),
    )
    target_names = {target.name for target in targets}
    record_names = set(records)
    missing = sorted(target_names - record_names)
    extra = sorted(record_names - target_names)
    if missing or extra:
        raise ValueError(
            "Pack coverage does not exactly match the model: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    if manifest.get("config_sha256") != model_config_sha256(model):
        raise ValueError("Pack config hash does not match the loaded checkpoint")

    model_steps = int(model.action_head.num_inference_timesteps)
    pack_steps = int(manifest["num_inference_timesteps"])
    if model_steps != pack_steps:
        raise ValueError(
            f"Checkpoint uses {model_steps} denoising steps but pack uses {pack_steps}"
        )

    for target in targets:
        record = records[target.name]
        if record.get("sha256") != tensor_record_sha256(record):
            raise ValueError(f"Layer record checksum mismatch: {target.name}")
        if record.get("scope") != target.scope:
            raise ValueError(f"Layer scope mismatch for {target.name}")
        parent, attribute = _parent_and_attribute(model, target.name)
        setattr(parent, attribute, HoloQLinear(target.module, name=target.name, record=record))

    model.holoq_manifest = manifest
    return summary

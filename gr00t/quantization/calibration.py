# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline calibration collector for GR00T-N1.7 HoloQ W4A4 packs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .context import get_dit_quant_step, get_dit_quant_total_steps
from .packing import (
    SIGNED_QMAX,
    apply_input_transform,
    build_svd_hadamard_transform,
    stable_layer_seed,
)
from .runtime import model_config_sha256
from .scope import QuantTarget, discover_n1d7_targets, normalize_scopes, validate_n1d7_scope


CALIBRATION_FORMAT_VERSION = 2


class HoloQCalibrationCollector:
    """Collect block Hessians for LLM/ViT GPTQ and q99.9 tables for DiT RTN.

    DiT percentiles use a streaming per-channel top-k order statistic. This is
    exact while the q99.9 rank-from-the-top is no larger than ``topk``; finalizing
    fails rather than silently approximating if more storage is required.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        suite: str,
        run_id: str,
        rotation_block_size: int = 64,
        gptq_block_size: int = 128,
        percentile: float = 99.9,
        topk: int = 512,
        seed: int = 0,
        scopes: str | tuple[str, ...] | list[str] | None = None,
        include_vit_mergers: bool = True,
        include_vit_patch_embed: bool = False,
    ) -> None:
        if suite not in {"object", "spatial", "goal", "long"}:
            raise ValueError(f"Unsupported LIBERO suite {suite!r}")
        if not run_id.strip():
            raise ValueError("run_id must identify the calibration rollout set")
        if not 0 < percentile < 100:
            raise ValueError(f"percentile must be in (0, 100), got {percentile}")
        if topk <= 0:
            raise ValueError(f"topk must be positive, got {topk}")

        self.model = model
        self.suite = suite
        self.run_id = run_id
        self.rotation_block_size = rotation_block_size
        self.gptq_block_size = gptq_block_size
        self.percentile = percentile
        self.topk = topk
        self.seed = seed
        self.scopes = normalize_scopes(scopes)
        self.include_vit_mergers = include_vit_mergers
        self.include_vit_patch_embed = include_vit_patch_embed
        self.num_steps = int(model.action_head.num_inference_timesteps)
        self.targets = discover_n1d7_targets(
            model,
            scopes=self.scopes,
            include_vit_mergers=include_vit_mergers,
            include_vit_patch_embed=include_vit_patch_embed,
        )
        self.summary = validate_n1d7_scope(
            self.targets,
            expected_llm_layers=16 if "llm" in self.scopes else None,
            expected_dit_layers=32 if "dit" in self.scopes else None,
            scopes=self.scopes,
        )
        self._transforms: dict[str, tuple[torch.Tensor, torch.Tensor] | None] = {}
        self._llm_hessian: dict[str, torch.Tensor] = {}
        self._llm_rows: dict[str, int] = {}
        self._dit_topk: dict[str, list[torch.Tensor | None]] = {}
        self._dit_rows: dict[str, list[int]] = {}
        self._identity_rows: dict[str, int] = {}
        self._handles: list[Any] = []
        self._closed = False

        for target in self.targets:
            if target.module_kind == "conv3d_patch":
                # Patch width is not generally divisible by the rotation block.
                # It is lowered to a padded GEMM and intentionally uses identity.
                self._transforms[target.name] = None
                self._identity_rows[target.name] = 0
            else:
                layer_seed = stable_layer_seed(seed, target.name)
                permutation, rotations, _ = build_svd_hadamard_transform(
                    target.module.weight,
                    block_size=rotation_block_size,
                    seed=layer_seed,
                )
                self._transforms[target.name] = (
                    permutation.to(target.module.weight.device),
                    rotations.to(target.module.weight.device, dtype=torch.float16),
                )
            if target.scope == "dit":
                self._dit_topk[target.name] = [None] * self.num_steps
                self._dit_rows[target.name] = [0] * self.num_steps
            self._handles.append(target.module.register_forward_pre_hook(self._hook(target)))

    def _hook(self, target: QuantTarget):
        def collect(_module: nn.Module, args: tuple[Any, ...]) -> None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError(f"{target.name}: expected tensor input during calibration")
            transform = self._transforms[target.name]
            inputs = args[0].detach()
            if target.module_kind == "conv3d_patch":
                transformed = inputs.flatten(1)
            else:
                permutation, rotations = transform
                transformed = apply_input_transform(inputs, permutation, rotations)
            rows = transformed.reshape(-1, transformed.shape[-1]).float()
            if target.module_kind == "conv3d_patch":
                self._identity_rows[target.name] += rows.shape[0]
                return
            if target.scope in {"llm", "vit"}:
                self._collect_llm(target.name, rows)
            else:
                self._collect_dit(target.name, rows)

        return collect

    def _collect_llm(self, name: str, rows: torch.Tensor) -> None:
        width = rows.shape[1]
        if width % self.gptq_block_size:
            raise RuntimeError(
                f"{name}: width {width} is not divisible by GPTQ block {self.gptq_block_size}"
            )
        blocks = rows.reshape(rows.shape[0], -1, self.gptq_block_size).transpose(0, 1)
        covariance = torch.bmm(blocks.transpose(1, 2), blocks)
        if name not in self._llm_hessian:
            self._llm_hessian[name] = covariance
            self._llm_rows[name] = rows.shape[0]
        else:
            self._llm_hessian[name].add_(covariance)
            self._llm_rows[name] += rows.shape[0]

    def _collect_dit(self, name: str, rows: torch.Tensor) -> None:
        step = get_dit_quant_step()
        total_steps = get_dit_quant_total_steps()
        if step is None or total_steps is None:
            raise RuntimeError(f"{name}: DiT calibration ran outside a denoising-step context")
        if total_steps != self.num_steps:
            raise RuntimeError(
                f"{name}: collector expects {self.num_steps} steps, runtime uses {total_steps}"
            )
        values = rows.abs()
        keep = min(self.topk, values.shape[0])
        candidate = torch.topk(values, k=keep, dim=0, sorted=False).values.half()
        previous = self._dit_topk[name][step]
        if previous is not None:
            candidate = torch.cat((previous, candidate), dim=0)
        if candidate.shape[0] > self.topk:
            candidate = torch.topk(
                candidate.float(), k=self.topk, dim=0, sorted=False
            ).values.half()
        self._dit_topk[name][step] = candidate
        self._dit_rows[name][step] += rows.shape[0]

    def _remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._closed = True

    def finalize(self, output_path: str | Path) -> Path:
        """Validate coverage, remove hooks, and save one suite-specific artifact."""

        if self._closed:
            raise RuntimeError("Calibration collector has already been finalized")
        self._remove_hooks()
        layers: dict[str, dict[str, Any]] = {}
        quantile = self.percentile / 100.0
        for target in self.targets:
            transform = self._transforms[target.name]
            record: dict[str, Any] = {
                "scope": target.scope,
                "module_kind": target.module_kind,
                "projection": target.projection,
            }
            if transform is None:
                record["transform"] = "identity"
                record["permutation"] = None
                record["rotation_blocks"] = None
            else:
                permutation, rotations = transform
                record["transform"] = "svd-hadamard"
                record["permutation"] = permutation.cpu()
                record["rotation_blocks"] = rotations.half().cpu()
            if target.module_kind == "conv3d_patch":
                count = self._identity_rows[target.name]
                if count == 0:
                    raise RuntimeError(f"No ViT patch activations collected for {target.name}")
                record["sample_rows"] = count
            elif target.scope in {"llm", "vit"}:
                count = self._llm_rows.get(target.name, 0)
                if count == 0:
                    raise RuntimeError(
                        f"No {target.scope.upper()} calibration activations collected for {target.name}"
                    )
                record["sample_rows"] = count
                record["hessian_blocks"] = (self._llm_hessian[target.name] / count).float().cpu()
            else:
                scales = []
                for step in range(self.num_steps):
                    count = self._dit_rows[target.name][step]
                    values = self._dit_topk[target.name][step]
                    if count == 0 or values is None:
                        raise RuntimeError(
                            f"No DiT activations collected for {target.name} at step {step}"
                        )
                    rank_from_top = max(1, count - math.ceil(quantile * count) + 1)
                    if rank_from_top > values.shape[0]:
                        raise RuntimeError(
                            f"{target.name} step {step}: q{self.percentile} requires top-k "
                            f"rank {rank_from_top}, but topk={self.topk}; recalibrate with a larger value"
                        )
                    ordered = torch.sort(values.float(), dim=0, descending=True).values
                    scales.append(ordered[rank_from_top - 1].clamp_min(1e-8).cpu() / SIGNED_QMAX)
                record["sample_rows_per_step"] = self._dit_rows[target.name]
                record["activation_scale"] = torch.stack(scales)
            layers[target.name] = record

        artifact = {
            "manifest": {
                "format_version": CALIBRATION_FORMAT_VERSION,
                "algorithm": "holoq-vla-calibration",
                "model_type": "Gr00tN1d7",
                "suite": self.suite,
                "run_id": self.run_id,
                "config_sha256": model_config_sha256(self.model),
                "num_inference_timesteps": self.num_steps,
                "scopes": list(self.scopes),
                "include_vit_mergers": self.include_vit_mergers,
                "include_vit_patch_embed": self.include_vit_patch_embed,
                "llm_layers": self.summary.llm_layers,
                "dit_layers": self.summary.dit_layers,
                "vit_layers": self.summary.vit_layers,
                "llm_linears": self.summary.llm_linears,
                "dit_linears": self.summary.dit_linears,
                "vit_linears": self.summary.vit_linears,
                "vit_patch_convs": self.summary.vit_patch_convs,
                "rotation": "blockwise-svd-randomized-hadamard",
                "rotation_block_size": self.rotation_block_size,
                "permutation": "zigzag-weight-norm",
                "gptq_block_size": self.gptq_block_size,
                "gptq_damping": 0.01,
                "dit_solver": "rtn",
                "activation_percentile": self.percentile,
                "seed": self.seed,
            },
            "layers": layers,
        }
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, output)
        return output

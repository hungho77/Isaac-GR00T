# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""CKA utilities and structural pruning support for GR00T N1.7.

The pruning manifest is deliberately stored in ``Gr00tN1d7Config``.  A
fine-tuned pruned checkpoint can therefore rebuild its reduced architecture
before Hugging Face loads the state dict, while recovery training can still
load the full NVIDIA checkpoint first and prune only after loading succeeds.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
from torch import nn


SCHEMA_VERSION = 1
SUPPORTED_MODULES = ("backbone_language", "action_dit", "vl_self_attention")
ALLOWED_TIED_WEIGHT_MISSING_KEYS = frozenset({"backbone.model.lm_head.weight"})


def load_pruning_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        return validate_pruning_manifest(json.load(file))


def validate_pruning_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize a GR00T N1.7 pruning manifest."""
    if not isinstance(manifest, dict):
        raise TypeError("CKA pruning manifest must be a JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported pruning manifest schema_version={manifest.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    if manifest.get("model_type") != "Gr00tN1d7":
        raise ValueError("Pruning manifest model_type must be 'Gr00tN1d7'")

    modules = manifest.get("modules")
    if not isinstance(modules, dict) or not modules:
        raise ValueError("Pruning manifest must contain a non-empty 'modules' object")
    unknown = sorted(set(modules) - set(SUPPORTED_MODULES))
    if unknown:
        raise ValueError(
            f"Unsupported N1.7 pruning module(s): {unknown}. Vision blocks are intentionally "
            "excluded because Qwen3-VL deep-stack taps depend on their layer positions."
        )

    normalized = deepcopy(manifest)
    for name, spec in normalized["modules"].items():
        if not isinstance(spec, dict):
            raise TypeError(f"Manifest entry {name!r} must be an object")
        original_depth = int(spec.get("original_depth", 0))
        keep = [int(index) for index in spec.get("keep_indices", [])]
        if original_depth < 1:
            raise ValueError(f"{name}.original_depth must be positive")
        if not keep:
            raise ValueError(f"{name}.keep_indices cannot be empty")
        if keep != sorted(set(keep)):
            raise ValueError(f"{name}.keep_indices must be sorted and unique")
        if keep[0] < 0 or keep[-1] >= original_depth:
            raise ValueError(f"{name}.keep_indices={keep} is outside [0, {original_depth - 1}]")
        if name == "action_dit":
            categories = {
                "text_cross": [index for index in keep if index % 4 == 0],
                "self": [index for index in keep if index % 2 == 1],
                "image_cross": [index for index in keep if index % 4 == 2],
            }
            missing = [category for category, indices in categories.items() if not indices]
            if missing:
                raise ValueError(
                    "action_dit pruning must retain text-cross, self-attention and image-cross "
                    f"blocks; missing {missing}"
                )
        spec["original_depth"] = original_depth
        spec["keep_indices"] = keep
    return normalized


def _module_lists(model: nn.Module) -> dict[str, nn.ModuleList]:
    result = {
        "backbone_language": model.backbone.model.language_model.layers,
        "action_dit": model.action_head.model.transformer_blocks,
    }
    vl_self_attention = model.action_head.vl_self_attention
    if hasattr(vl_self_attention, "transformer_blocks"):
        result["vl_self_attention"] = vl_self_attention.transformer_blocks
    return result


def get_prunable_module_lists(model: nn.Module) -> dict[str, nn.ModuleList]:
    """Public read-only discovery helper used by CKA capture tooling."""
    return _module_lists(model)


def manifest_depths(manifest: dict[str, Any]) -> dict[str, int]:
    """Return canonical retained depths without constructing a model."""
    manifest = validate_pruning_manifest(manifest)
    return {name: len(spec["keep_indices"]) for name, spec in manifest["modules"].items()}


def verify_model_matches_manifest(
    model: nn.Module, manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Fail closed unless a loaded model exactly matches its pruning manifest.

    This verifies both reduced depths and the original indices that drive
    ``AlternateVLDiT`` routing. It is intended for inference/server preflight.
    """
    if manifest is None:
        manifest = getattr(model.config, "cka_pruning_manifest", None)
    if manifest is None:
        raise ValueError("Model config does not contain cka_pruning_manifest")
    manifest = validate_pruning_manifest(manifest)
    lists = _module_lists(model)

    actual_depths: dict[str, int] = {}
    actual_original_indices: dict[str, list[int]] = {}
    for name, spec in manifest["modules"].items():
        if name not in lists:
            raise ValueError(f"Loaded model has no prunable module {name!r}")
        layers = lists[name]
        expected_indices = spec["keep_indices"]
        actual_depths[name] = len(layers)
        if len(layers) != len(expected_indices):
            raise ValueError(
                f"{name} depth mismatch: loaded={len(layers)}, expected={len(expected_indices)}"
            )
        original_indices = [getattr(layer, "_gr00t_original_index", None) for layer in layers]
        actual_original_indices[name] = original_indices
        if original_indices != expected_indices:
            raise ValueError(
                f"{name} original-index mismatch: loaded={original_indices}, "
                f"expected={expected_indices}"
            )

    configured = validate_pruning_manifest(model.config.cka_pruning_manifest)
    if configured != manifest:
        raise ValueError("Loaded model manifest differs from the requested manifest")
    return {
        "status": "pass",
        "retained_depths": actual_depths,
        "original_indices": actual_original_indices,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def validate_checkpoint_loading_info(loading_info: dict[str, Any]) -> dict[str, Any]:
    """Reject partial/random initialization when loading a pruned checkpoint."""
    missing = sorted(set(loading_info.get("missing_keys", [])) - ALLOWED_TIED_WEIGHT_MISSING_KEYS)
    unexpected = sorted(set(loading_info.get("unexpected_keys", [])))
    mismatched = list(loading_info.get("mismatched_keys", []))
    errors = list(loading_info.get("error_msgs", []))
    if missing or unexpected or mismatched or errors:
        raise RuntimeError(
            "Strict CKA checkpoint loading failed: "
            f"missing={missing}, unexpected={unexpected}, "
            f"mismatched={mismatched}, errors={errors}"
        )
    return {
        "status": "pass",
        "allowed_missing_keys": sorted(
            set(loading_info.get("missing_keys", [])) & ALLOWED_TIED_WEIGHT_MISSING_KEYS
        ),
        "unexpected_keys": [],
        "mismatched_keys": [],
    }


def apply_pruning_manifest(model: nn.Module, manifest: dict[str, Any]) -> dict[str, Any]:
    """Apply a manifest in-place and return parameter/depth statistics.

    The function is idempotent for a checkpoint whose constructor has already
    applied the exact same manifest.
    """
    manifest = validate_pruning_manifest(manifest)
    already_applied = getattr(model.config, "cka_pruning_manifest", None)
    lists = _module_lists(model)
    before_parameters = sum(parameter.numel() for parameter in model.parameters())
    depth_before = {name: len(layers) for name, layers in lists.items()}

    for name, spec in manifest["modules"].items():
        if name not in lists:
            raise ValueError(f"Model has no prunable module {name!r}")
        layers = lists[name]
        keep = spec["keep_indices"]
        original_depth = spec["original_depth"]

        if len(layers) == len(keep) and already_applied == manifest:
            current_indices = [
                getattr(layer, "_gr00t_original_index", index) for index, layer in enumerate(layers)
            ]
            if current_indices != keep:
                raise ValueError(
                    f"Already-pruned {name} indices are {current_indices}, expected {keep}"
                )
            for original_index, layer in zip(keep, layers, strict=True):
                layer._gr00t_original_index = original_index
            continue
        if len(layers) != original_depth:
            raise ValueError(
                f"Cannot apply {name} pruning: model depth is {len(layers)}, manifest expects "
                f"the full depth {original_depth}. Refusing to prune an unknown architecture."
            )

        for original_index, layer in enumerate(layers):
            # AlternateVLDiT must preserve the role assigned when the block was
            # constructed. Forward uses this metadata after ModuleList indices change.
            layer._gr00t_original_index = original_index
        replacement = nn.ModuleList([layers[index] for index in keep])

        if name == "backbone_language":
            model.backbone.model.language_model.layers = replacement
        elif name == "action_dit":
            model.action_head.model.transformer_blocks = replacement
        elif name == "vl_self_attention":
            model.action_head.vl_self_attention.transformer_blocks = replacement

    model.config.cka_pruning_manifest = deepcopy(manifest)
    after_parameters = sum(parameter.numel() for parameter in model.parameters())
    depth_after = {name: len(layers) for name, layers in _module_lists(model).items()}
    return {
        "parameters_before": before_parameters,
        "parameters_after": after_parameters,
        "parameter_reduction_fraction": 1.0 - after_parameters / before_parameters,
        "depth_before": depth_before,
        "depth_after": depth_after,
    }


def _center_gram(gram: np.ndarray) -> np.ndarray:
    count = gram.shape[0]
    centering = np.eye(count, dtype=np.float64) - np.ones((count, count), dtype=np.float64) / count
    return centering @ gram @ centering


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Compute linear CKA using centered sample Gram matrices."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"CKA expects [samples, features] arrays, got {x.shape} and {y.shape}")
    if x.shape[0] < 2:
        raise ValueError("CKA requires at least two calibration samples")
    x_gram = _center_gram(x @ x.T)
    y_gram = _center_gram(y @ y.T)
    numerator = float(np.sum(x_gram * y_gram))
    denominator = float(np.linalg.norm(x_gram) * np.linalg.norm(y_gram))
    if denominator == 0.0:
        return 0.0
    # Round-off can produce values a few ulps outside the theoretical [0, 1]
    # interval, which in turn distorts heatmap scales and ranking ties.
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def consecutive_cka(layer_activations: list[np.ndarray]) -> list[float]:
    if len(layer_activations) < 2:
        return []
    return [
        linear_cka(layer_activations[index - 1], layer_activations[index])
        for index in range(1, len(layer_activations))
    ]


def select_keep_indices(scores: list[float], target_keep: int, *, module_name: str) -> list[int]:
    """Select high-CKA layers to remove under N1.7 topology constraints."""
    depth = len(scores) + 1
    if not 1 <= target_keep <= depth:
        raise ValueError(f"target_keep must be in [1, {depth}], got {target_keep}")
    keep = set(range(depth))
    # Retain boundary transformations whenever the budget permits. Layer zero
    # consumes raw embeddings and the final layer feeds the downstream head;
    # dropping either is a much stronger architectural intervention than
    # removing a redundant interior layer.
    protected = {0}
    if target_keep >= 2:
        protected.add(depth - 1)
    candidates = sorted(
        (index for index in range(depth) if index not in protected),
        key=lambda index: scores[index - 1],
        reverse=True,
    )

    def topology_valid(indices: set[int]) -> bool:
        if module_name != "action_dit":
            return True
        return (
            any(index % 4 == 0 for index in indices)
            and any(index % 2 == 1 for index in indices)
            and any(index % 4 == 2 for index in indices)
        )

    for candidate in candidates:
        if len(keep) <= target_keep:
            break
        proposed = keep - {candidate}
        if topology_valid(proposed):
            keep = proposed
    if len(keep) != target_keep:
        raise ValueError(
            f"Could not reach target_keep={target_keep} for {module_name} without violating "
            "the N1.7 attention topology"
        )
    return sorted(keep)

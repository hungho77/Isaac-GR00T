# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in visual token hook for GR00T efficient inference experiments."""

from __future__ import annotations

from types import MethodType
from typing import Any

from gr00t.efficient.pruners.utils import gather_tokens


class VisualTokenHook:
    """Apply an efficient inference method to visual tokens when explicitly enabled."""

    def __init__(self, method: Any | None = None, enabled: bool = False) -> None:
        self.method = method
        self.enabled = enabled
        self.last_metadata: dict[str, Any] = {}

    def apply(self, visual_tokens: Any, **kwargs: Any) -> Any:
        """Apply the method to [B, N, D] tokens or return tokens unchanged."""
        if not self.enabled or self.method is None:
            self.last_metadata = {
                "method": getattr(self.method, "method_name", "none"),
                "pruning_method": "none",
                "pruned": False,
                "hook_enabled": self.enabled,
            }
            return visual_tokens

        pruned_tokens, metadata = self.method.process_visual_tokens(visual_tokens, **kwargs)
        metadata["hook_enabled"] = self.enabled
        self.last_metadata = metadata
        return pruned_tokens

    def apply_backbone_output(self, backbone_output: Any, **kwargs: Any) -> Any:
        """Apply visual-only pruning to a GR00T N1.7 backbone output BatchFeature."""
        if not self.enabled or self.method is None:
            return backbone_output

        features = _get_feature(backbone_output, "backbone_features")
        if features is None:
            self.last_metadata = {
                "method": getattr(self.method, "method_name", "none"),
                "pruning_method": "none",
                "pruned": False,
                "hook_enabled": self.enabled,
                "hook_error": "missing_backbone_features",
            }
            return backbone_output

        image_mask = _get_feature(backbone_output, "image_mask")
        attention_mask = _get_feature(backbone_output, "backbone_attention_mask")

        visual_tokens = self._extract_visual_tokens(features, image_mask)
        if visual_tokens is None:
            pruned_features = self.apply(features, **kwargs)
            _set_feature(backbone_output, "backbone_features", pruned_features)
            selected_indices = _selected_indices(self.method)
            if selected_indices is not None:
                _set_feature(backbone_output, "backbone_attention_mask", _gather_mask(attention_mask, selected_indices))
                _set_feature(backbone_output, "image_mask", _gather_mask(image_mask, selected_indices))
            self.last_metadata["hook_scope"] = "backbone_features"
            return backbone_output

        _ = self.apply(visual_tokens, **kwargs)
        selected_visual_indices = _selected_indices(self.method)
        if selected_visual_indices is None:
            self.last_metadata["hook_error"] = "missing_selected_indices"
            return backbone_output

        full_indices = self._build_full_indices(image_mask, selected_visual_indices)
        if full_indices is None:
            self.last_metadata["hook_error"] = "failed_to_build_full_indices"
            return backbone_output

        _set_feature(backbone_output, "backbone_features", gather_tokens(features, full_indices))
        _set_feature(backbone_output, "backbone_attention_mask", _gather_mask(attention_mask, full_indices))
        _set_feature(backbone_output, "image_mask", _gather_mask(image_mask, full_indices))
        self.last_metadata["hook_scope"] = "visual_tokens_only"
        self.last_metadata["full_token_count_before"] = int(features.shape[1])
        self.last_metadata["full_token_count_after"] = int(_get_feature(backbone_output, "backbone_features").shape[1])
        return backbone_output

    def _extract_visual_tokens(self, features: Any, image_mask: Any | None) -> Any | None:
        torch = _torch_or_none()
        if torch is None or image_mask is None:
            return None
        if not isinstance(features, torch.Tensor) or not isinstance(image_mask, torch.Tensor):
            return None
        if image_mask.dim() != 2 or features.dim() != 3:
            return None

        counts = image_mask.sum(dim=1)
        if counts.numel() == 0 or torch.any(counts != counts[0]):
            return None
        visual_count = int(counts[0].item())
        if visual_count < 1:
            return None
        return features[image_mask].view(features.shape[0], visual_count, features.shape[-1])

    def _build_full_indices(self, image_mask: Any, selected_visual_indices: Any) -> Any | None:
        torch = _torch_or_none()
        if torch is None or image_mask is None or not isinstance(image_mask, torch.Tensor):
            return None

        selected = torch.as_tensor(selected_visual_indices, dtype=torch.long, device=image_mask.device)
        if selected.dim() == 1:
            selected = selected.unsqueeze(0).expand(image_mask.shape[0], -1)
        if selected.dim() != 2 or selected.shape[0] != image_mask.shape[0]:
            return None

        full_indices = []
        for batch_idx in range(image_mask.shape[0]):
            visual_positions = torch.nonzero(image_mask[batch_idx], as_tuple=False).flatten()
            non_visual_positions = torch.nonzero(~image_mask[batch_idx], as_tuple=False).flatten()
            kept_visual_positions = visual_positions[selected[batch_idx]]
            merged = torch.sort(torch.cat((non_visual_positions, kept_visual_positions), dim=0)).values
            full_indices.append(merged)
        return torch.stack(full_indices, dim=0)


def attach_visual_token_hook(model: Any, hook: VisualTokenHook) -> VisualTokenHook:
    """Patch one model instance so its action head applies an opt-in visual token hook."""
    action_head = getattr(model, "action_head", None)
    if action_head is None:
        raise ValueError("Expected model.action_head for visual token hook attachment.")
    if hasattr(action_head, "_efficient_original_process_backbone_output"):
        action_head._efficient_visual_token_hook = hook
        return hook

    original = action_head.process_backbone_output
    action_head._efficient_original_process_backbone_output = original
    action_head._efficient_visual_token_hook = hook

    def process_backbone_output_with_hook(self: Any, backbone_output: Any) -> Any:
        processed = self._efficient_original_process_backbone_output(backbone_output)
        return self._efficient_visual_token_hook.apply_backbone_output(processed)

    action_head.process_backbone_output = MethodType(process_backbone_output_with_hook, action_head)
    return hook


def _get_feature(batch_feature: Any, key: str) -> Any | None:
    if isinstance(batch_feature, dict):
        return batch_feature.get(key)
    return getattr(batch_feature, key, None)


def _set_feature(batch_feature: Any, key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(batch_feature, dict):
        batch_feature[key] = value
    else:
        setattr(batch_feature, key, value)


def _selected_indices(method: Any) -> Any | None:
    for attr in ("pruner", "vlapruner"):
        pruner = getattr(method, attr, None)
        if pruner is not None and getattr(pruner, "last_selected_indices", None) is not None:
            return pruner.last_selected_indices
    if getattr(method, "last_selected_indices", None) is not None:
        return method.last_selected_indices
    return None


def _gather_mask(mask: Any | None, indices: Any) -> Any | None:
    if mask is None:
        return None
    torch = _torch_or_none()
    if torch is not None and isinstance(mask, torch.Tensor):
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=mask.device)
        if index_tensor.dim() == 1:
            return mask.index_select(1, index_tensor)
        if index_tensor.dim() == 2:
            return mask.gather(dim=1, index=index_tensor)
    return mask[:, list(indices)]


def _torch_or_none() -> Any | None:
    try:
        import torch
    except Exception:
        return None
    return torch

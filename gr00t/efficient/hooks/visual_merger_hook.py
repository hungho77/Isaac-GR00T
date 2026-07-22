# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in visual-token pruning right after Qwen3-VL's vision merger.

This hook patches ``Qwen3VLModel.get_image_features`` (the method that runs the
vision tower's patch-embed + ViT blocks + merger, and returns one merged-token
tensor per image) and applies pruning to its return value, before those tokens
are scattered into the packed text+image sequence.

Only mask-only pruning is supported here. Gather-mode (physically shortening
the token count) is NOT safe at this point: ``Qwen3VLModel.get_placeholder_mask``
raises ``ValueError`` unless ``image_features.numel()`` exactly matches the
number of image-placeholder positions already baked into ``input_ids``
(transformers/models/qwen3_vl/modeling_qwen3_vl.py, ``get_placeholder_mask``).
Satisfying that would require also editing ``input_ids`` before this call --
a materially more invasive change than gather-pruning after the full sequence
is built (see ``backbone_token_hook.py``). Mask-only avoids this entirely: it
never changes tensor shape, so ``masked_scatter`` and DeepStack's later
positional injection (which indexes by the same unchanged mask) stay valid.
"""

from __future__ import annotations

from types import MethodType
from typing import Any


class VisualMergerHook:
    """Zero out low-score visual tokens right after the Qwen3-VL merger."""

    def __init__(
        self, method: Any | None = None, enabled: bool = False, mode: str = "mask_only"
    ) -> None:
        if mode != "mask_only":
            # Gather is rejected at this hook point (see module docstring); fall
            # back to the only safe mode rather than silently doing nothing.
            mode = "mask_only"
        self.method = method
        self.enabled = enabled
        self.mode = mode
        self.last_metadata: dict[str, Any] = {}

    def apply_image_features(self, image_embeds_list: Any, split_sizes: list[int]) -> Any:
        """Mask low-score rows in each per-image embedding tensor; shape unchanged."""
        import torch

        if not self.enabled or self.method is None:
            self.last_metadata = {
                "hook_name": "after_visual_merger",
                "pruning_method": "none",
                "pruned": False,
                "hook_enabled": self.enabled,
            }
            return image_embeds_list

        combined = torch.cat(image_embeds_list, dim=0)
        if combined.dim() != 2:
            self.last_metadata = {
                "hook_name": "after_visual_merger",
                "pruning_method": "none",
                "pruned": False,
                "hook_enabled": self.enabled,
                "hook_error": "unexpected_image_embeds_shape",
            }
            return image_embeds_list

        original_tokens = combined.shape[0]
        batched = combined.unsqueeze(0)  # [1, N, D]; batch=1, matching Hook B/C's constraint
        _, metadata = self.method.process_visual_tokens(batched)

        selected = _selected_indices(self.method)
        if selected is None:
            self.last_metadata = {
                "hook_name": "after_visual_merger",
                "pruning_method": metadata.get("pruning_method", "none"),
                "pruned": False,
                "hook_enabled": self.enabled,
                "hook_error": "missing_selected_indices",
            }
            return image_embeds_list

        selected = torch.as_tensor(selected, dtype=torch.long, device=combined.device).flatten()
        keep_mask = torch.zeros(original_tokens, dtype=torch.bool, device=combined.device)
        keep_mask[selected] = True
        masked = combined * keep_mask.unsqueeze(-1).to(combined.dtype)
        masked_list = list(torch.split(masked, split_sizes))

        kept = int(selected.numel())
        effective_keep_ratio = 0.0 if original_tokens == 0 else kept / original_tokens
        self.last_metadata = {
            "hook_name": "after_visual_merger",
            "mode": "mask_only",
            "pruning_method": metadata.get(
                "pruning_method", getattr(self.method, "method_name", "none")
            ),
            "keep_ratio": getattr(self.method, "keep_ratio", None),
            "visual_token_count_before": original_tokens,
            "visual_token_count_after": original_tokens,  # shape unchanged by design
            "nonzero_token_count_after": kept,
            "effective_keep_ratio": effective_keep_ratio,
            "token_reduction_ratio": 1.0 - effective_keep_ratio,
            "pruned": kept < original_tokens,
            "hook_enabled": self.enabled,
            "per_camera_token_counts": list(split_sizes),
        }
        return masked_list


def attach_visual_merger_hook(model: Any, hook: VisualMergerHook) -> VisualMergerHook:
    """Patch one Qwen3VLModel instance's get_image_features to apply the hook."""
    qwen_model = getattr(getattr(getattr(model, "backbone", None), "model", None), "model", None)
    if qwen_model is None or not hasattr(qwen_model, "get_image_features"):
        raise ValueError(
            "Expected model.backbone.model.model (Qwen3VLModel) with get_image_features "
            "for visual merger hook attachment."
        )

    qwen_model._efficient_merger_hook = hook
    if hasattr(qwen_model, "_efficient_original_get_image_features"):
        return hook

    original = qwen_model.get_image_features
    qwen_model._efficient_original_get_image_features = original

    def get_image_features_with_hook(
        self: Any, pixel_values: Any, image_grid_thw: Any = None
    ) -> Any:
        image_embeds, deepstack_image_embeds = self._efficient_original_get_image_features(
            pixel_values, image_grid_thw
        )
        split_sizes = [t.shape[0] for t in image_embeds]
        masked = self._efficient_merger_hook.apply_image_features(list(image_embeds), split_sizes)
        # Mask-only never changes per-image shapes, so DeepStack features
        # (indexed by the same unchanged placeholder positions) stay valid.
        return tuple(masked), deepstack_image_embeds

    qwen_model.get_image_features = MethodType(get_image_features_with_hook, qwen_model)
    return hook


def _selected_indices(method: Any) -> Any | None:
    for attr in ("pruner", "vlapruner"):
        pruner = getattr(method, attr, None)
        if pruner is not None and getattr(pruner, "last_selected_indices", None) is not None:
            return pruner.last_selected_indices
    if getattr(method, "last_selected_indices", None) is not None:
        return method.last_selected_indices
    return None

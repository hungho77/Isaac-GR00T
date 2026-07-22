# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in early-layer visual token pruning inside the Qwen3-VL backbone.

Unlike :mod:`gr00t.efficient.hooks.visual_token_hook` (which prunes after the
backbone forward and therefore cannot reduce backbone compute), this hook
patches the Qwen3-VL text model so that visual tokens are dropped after an
early decoder layer. All subsequent decoder layers then attend over the
shorter sequence, which is where the latency actually lives.

Constraints (checked at runtime, safe no-op otherwise):
- batch size 1 (``cache_position`` is shared across the batch);
- pruning happens no earlier than the last DeepStack injection layer, so the
  injected visual features are never misaligned;
- kept tokens retain their original mrope position ids (FastV-style), so
  RoPE stays aligned with the unpruned computation.
"""

from __future__ import annotations

import sys
from types import MethodType
from typing import Any


class BackboneVisualTokenHook:
    """Prune visual tokens after an early Qwen3-VL decoder layer when enabled."""

    def __init__(
        self, method: Any | None = None, enabled: bool = False, prune_layer: int = 3
    ) -> None:
        if prune_layer < 0:
            raise ValueError("prune_layer must be >= 0.")
        self.method = method
        self.enabled = enabled
        self.prune_layer = prune_layer
        self.last_metadata: dict[str, Any] = {}
        self.last_full_indices: Any | None = None

    def begin_forward(self) -> None:
        """Reset per-forward state before the backbone runs."""
        self.last_full_indices = None

    def resolve_prune_layer(self, num_deepstack_layers: int, num_layers: int) -> int:
        """Clamp the prune layer after DeepStack injections and inside the stack."""
        effective = max(self.prune_layer, num_deepstack_layers - 1)
        if effective >= num_layers:
            return -1
        return effective

    def prune_intermediate(
        self,
        *,
        hidden_states: Any,
        visual_pos_masks: Any,
        attention_mask: Any,
        text_position_ids: Any,
        position_ids: Any,
        cache_position: Any,
    ) -> dict[str, Any] | None:
        """Prune visual positions from mid-stack decoder state; None means no-op."""
        import torch

        if not self.enabled or self.method is None:
            return None
        if visual_pos_masks is None or not isinstance(visual_pos_masks, torch.Tensor):
            self._record_noop("missing_visual_pos_masks")
            return None
        if hidden_states.shape[0] != 1:
            self._record_noop("backbone_prune_requires_batch_size_1")
            return None
        if attention_mask is not None and (
            not isinstance(attention_mask, torch.Tensor) or attention_mask.dim() != 4
        ):
            self._record_noop("unsupported_attention_mask_format")
            return None

        mask = visual_pos_masks[0].bool()
        visual_positions = torch.nonzero(mask, as_tuple=False).flatten()
        if visual_positions.numel() < 1:
            self._record_noop("no_visual_positions")
            return None

        visual_tokens = hidden_states[:, visual_positions, :]
        _, metadata = self.method.process_visual_tokens(visual_tokens)
        selected = _selected_indices(self.method)
        if selected is None:
            self._record_noop("missing_selected_indices")
            return None
        selected = torch.as_tensor(
            selected, dtype=torch.long, device=hidden_states.device
        ).flatten()
        if selected.numel() == visual_positions.numel():
            # keep_ratio 1.0: nothing to drop, skip the gathers entirely.
            self.last_metadata = dict(metadata)
            self.last_metadata.update({"hook_scope": "backbone_layer", "pruned": False})
            return None

        kept_visual_positions = visual_positions[selected]
        non_visual_positions = torch.nonzero(~mask, as_tuple=False).flatten()
        keep = torch.sort(torch.cat((non_visual_positions, kept_visual_positions), dim=0)).values

        pruned: dict[str, Any] = {
            "hidden_states": hidden_states.index_select(1, keep),
            "visual_pos_masks": visual_pos_masks.index_select(1, keep),
            "text_position_ids": (
                text_position_ids.index_select(1, keep) if text_position_ids is not None else None
            ),
            "position_ids": (
                position_ids.index_select(-1, keep) if position_ids is not None else None
            ),
            "cache_position": (
                cache_position.index_select(0, keep) if cache_position is not None else None
            ),
            "attention_mask": (
                attention_mask.index_select(2, keep).index_select(3, keep)
                if attention_mask is not None
                else None
            ),
        }

        # Keep on-device: a .cpu() here would force a GPU sync inside every backbone forward.
        self.last_full_indices = keep
        self.last_metadata = dict(metadata)
        self.last_metadata.update(
            {
                "hook_scope": "backbone_layer",
                "full_token_count_before": int(hidden_states.shape[1]),
                "full_token_count_after": int(keep.numel()),
            }
        )
        return pruned

    def _record_noop(self, reason: str) -> None:
        self.last_metadata = {
            "method": getattr(self.method, "method_name", "none"),
            "pruning_method": "none",
            "pruned": False,
            "hook_enabled": self.enabled,
            "hook_error": reason,
        }


def attach_backbone_visual_token_hook(
    model: Any, hook: BackboneVisualTokenHook
) -> BackboneVisualTokenHook:
    """Patch one Gr00tN1d7 instance to prune visual tokens inside the backbone."""
    backbone = getattr(model, "backbone", None)
    qwen_vl = getattr(backbone, "model", None)
    inner_model = getattr(qwen_vl, "model", None)
    text_model = getattr(inner_model, "language_model", None)
    if backbone is None or text_model is None:
        raise ValueError(
            "Expected model.backbone.model.model.language_model (Qwen3-VL text model) "
            "for backbone visual token hook attachment."
        )

    patch_text_model(text_model, hook)

    backbone._efficient_backbone_hook = hook
    if not hasattr(backbone, "_efficient_original_forward"):
        backbone._efficient_original_forward = backbone.forward

        def backbone_forward_with_hook(self: Any, vl_input: Any) -> Any:
            active_hook = self._efficient_backbone_hook
            active_hook.begin_forward()
            output = self._efficient_original_forward(vl_input)
            keep = active_hook.last_full_indices
            if keep is not None:
                image_mask = output["image_mask"]
                keep_dev = keep.to(image_mask.device)
                output["image_mask"] = image_mask.index_select(1, keep_dev)
                output["backbone_attention_mask"] = output["backbone_attention_mask"].index_select(
                    1, keep_dev
                )
            return output

        backbone.forward = MethodType(backbone_forward_with_hook, backbone)
    return hook


def patch_text_model(text_model: Any, hook: BackboneVisualTokenHook) -> None:
    """Replace one Qwen3VLTextModel instance's forward with a pruning-aware copy."""
    text_model._efficient_backbone_hook = hook
    if hasattr(text_model, "_efficient_original_forward"):
        return
    text_model._efficient_original_forward = text_model.forward
    text_model.forward = MethodType(_text_forward_with_pruning, text_model)


def _text_forward_with_pruning(
    self: Any,
    input_ids: Any = None,
    attention_mask: Any = None,
    position_ids: Any = None,
    past_key_values: Any = None,
    inputs_embeds: Any = None,
    use_cache: Any = None,
    cache_position: Any = None,
    visual_pos_masks: Any = None,
    deepstack_visual_embeds: Any = None,
    **kwargs: Any,
) -> Any:
    """Replicate Qwen3VLTextModel.forward (transformers 4.57) with mid-stack pruning."""
    import torch

    mod = sys.modules[type(self).__module__]
    hook: BackboneVisualTokenHook = self._efficient_backbone_hook

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = mod.DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = position_ids[0]

    attention_mask = mod.create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    num_deepstack = len(deepstack_visual_embeds) if deepstack_visual_embeds is not None else 0
    prune_after = hook.resolve_prune_layer(num_deepstack, len(self.layers))

    for layer_idx, decoder_layer in enumerate(self.layers):
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = layer_outputs

        if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
            hidden_states = self._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )

        if layer_idx == prune_after:
            pruned = hook.prune_intermediate(
                hidden_states=hidden_states,
                visual_pos_masks=visual_pos_masks,
                attention_mask=attention_mask,
                text_position_ids=text_position_ids,
                position_ids=position_ids,
                cache_position=cache_position,
            )
            if pruned is not None:
                hidden_states = pruned["hidden_states"]
                attention_mask = pruned["attention_mask"]
                text_position_ids = pruned["text_position_ids"]
                position_ids = pruned["position_ids"]
                cache_position = pruned["cache_position"]
                visual_pos_masks = pruned["visual_pos_masks"]
                position_embeddings = self.rotary_emb(hidden_states, position_ids)

    hidden_states = self.norm(hidden_states)

    return mod.BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


def _selected_indices(method: Any) -> Any | None:
    for attr in ("pruner", "vlapruner"):
        pruner = getattr(method, attr, None)
        if pruner is not None and getattr(pruner, "last_selected_indices", None) is not None:
            return pruner.last_selected_indices
    if getattr(method, "last_selected_indices", None) is not None:
        return method.last_selected_indices
    return None

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training-free VLA-Pruner MVP for visual-token pruning experiments."""

from __future__ import annotations

import math
from typing import Any

from gr00t.efficient.pruners.base import VisualTokenPruner
from gr00t.efficient.pruners.utils import first_k_indices, gather_tokens


class VLAPruner(VisualTokenPruner):
    """MVP visual-token pruner with simple semantic/action scoring."""

    method_name = "vlapruner"
    _VALID_SCORE_MODES = {"norm", "mean_abs", "attention", "action"}

    def __init__(
        self,
        keep_ratio: float = 0.75,
        alpha: float = 0.5,
        beta: float = 0.5,
        temporal_momentum: float = 0.8,
        score_mode: str = "norm",
        enabled: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__(keep_ratio=keep_ratio, enabled=enabled, method_name=self.method_name)
        if score_mode not in self._VALID_SCORE_MODES:
            raise ValueError(
                f"Unsupported score_mode {score_mode!r}. Use one of {sorted(self._VALID_SCORE_MODES)}."
            )
        if not 0.0 <= temporal_momentum <= 1.0:
            raise ValueError("temporal_momentum must be in the interval [0, 1].")
        self.alpha = alpha
        self.beta = beta
        self.temporal_momentum = temporal_momentum
        self.score_mode = score_mode
        self.seed = seed
        self.prev_score: Any | None = None
        self.last_selected_indices: Any | None = None

    def reset(self) -> None:
        super().reset()
        self.prev_score = None
        self.last_selected_indices = None

    def prune(
        self,
        visual_tokens: Any,
        attention: Any | None = None,
        robot_state: Any | None = None,
        action_state: Any | None = None,
        timestep: int | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Prune [B, N, D] visual tokens by top-k MVP scores."""
        input_shape = self._validate_visual_tokens(visual_tokens)
        original_tokens = input_shape[1]
        keep = self._compute_keep_count(original_tokens)

        if not self.enabled or self.keep_ratio >= 1.0 or keep == original_tokens:
            indices = first_k_indices(original_tokens, original_tokens)
            self.last_selected_indices = indices
            metadata = self._metadata(
                original_tokens=original_tokens,
                kept_tokens=original_tokens,
                input_shape=input_shape,
                output_shape=input_shape,
                pruned=False,
                timestep=timestep,
                used_attention=False,
                used_action_score=False,
                used_temporal_smoothing=False,
                selected_indices_shape=[original_tokens],
            )
            self.last_metadata = metadata
            return visual_tokens, metadata

        torch = _torch_or_none()
        if torch is None or not isinstance(visual_tokens, torch.Tensor):
            return self._fallback_prune(visual_tokens, input_shape, keep, timestep)

        score, score_metadata = self._compute_score(
            visual_tokens=visual_tokens,
            attention=attention,
            action_state=action_state,
            torch=torch,
        )

        used_temporal_smoothing = False
        if self.prev_score is not None and tuple(self.prev_score.shape) == tuple(score.shape):
            score = self.temporal_momentum * self.prev_score + (1.0 - self.temporal_momentum) * score
            used_temporal_smoothing = True

        topk_indices = torch.topk(score, k=keep, dim=1, largest=True, sorted=False).indices
        selected_indices = torch.sort(topk_indices, dim=1).values
        pruned_tokens = gather_tokens(visual_tokens, selected_indices)
        output_shape = self._validate_visual_tokens(pruned_tokens)

        self.prev_score = score.detach()
        self.last_selected_indices = selected_indices.detach().cpu()
        metadata = self._metadata(
            original_tokens=original_tokens,
            kept_tokens=keep,
            input_shape=input_shape,
            output_shape=output_shape,
            pruned=True,
            timestep=timestep,
            used_attention=score_metadata["used_attention"],
            used_action_score=score_metadata["used_action_score"],
            used_temporal_smoothing=used_temporal_smoothing,
            selected_indices_shape=list(selected_indices.shape),
            selected_indices_preview=selected_indices[0, :16].detach().cpu().tolist(),
        )
        self.last_metadata = metadata
        return pruned_tokens, metadata

    def _fallback_prune(
        self,
        visual_tokens: Any,
        input_shape: tuple[int, int, int],
        keep: int,
        timestep: int | None,
    ) -> tuple[Any, dict[str, Any]]:
        """Dependency-safe fallback when torch tensors are unavailable."""
        original_tokens = input_shape[1]
        selected_indices = first_k_indices(original_tokens, keep)
        pruned_tokens = gather_tokens(visual_tokens, selected_indices)
        output_shape = self._validate_visual_tokens(pruned_tokens)
        self.last_selected_indices = selected_indices
        metadata = self._metadata(
            original_tokens=original_tokens,
            kept_tokens=keep,
            input_shape=input_shape,
            output_shape=output_shape,
            pruned=True,
            timestep=timestep,
            used_attention=False,
            used_action_score=False,
            used_temporal_smoothing=False,
            selected_indices_shape=[len(selected_indices)],
            selected_indices_preview=selected_indices[:16],
            fallback_reason="torch_tensor_unavailable",
        )
        self.last_metadata = metadata
        return pruned_tokens, metadata

    def _compute_score(
        self,
        visual_tokens: Any,
        attention: Any | None,
        action_state: Any | None,
        torch: Any,
    ) -> tuple[Any, dict[str, bool]]:
        semantic_score, used_attention = self._semantic_score(visual_tokens, attention, torch)
        action_score = self._action_score(visual_tokens, action_state, torch)
        used_action_score = action_score is not None

        if used_action_score:
            score = self.alpha * semantic_score + self.beta * action_score
        else:
            score = semantic_score
        return score, {
            "used_attention": used_attention,
            "used_action_score": used_action_score,
        }

    def _semantic_score(self, visual_tokens: Any, attention: Any | None, torch: Any) -> tuple[Any, bool]:
        if self.score_mode == "mean_abs":
            return torch.mean(torch.abs(visual_tokens.float()), dim=-1), False

        if self.score_mode == "attention":
            attention_score = self._attention_score(
                attention=attention,
                batch_size=visual_tokens.shape[0],
                num_tokens=visual_tokens.shape[1],
                torch=torch,
            )
            if attention_score is not None:
                return attention_score, True

        return torch.norm(visual_tokens.float(), dim=-1), False

    def _attention_score(
        self,
        attention: Any | None,
        batch_size: int,
        num_tokens: int,
        torch: Any,
    ) -> Any | None:
        if attention is None or not isinstance(attention, torch.Tensor):
            return None
        if attention.dim() < 1 or attention.shape[-1] != num_tokens:
            return None

        score = torch.abs(attention.float())
        if score.dim() == 1:
            score = score.view(1, num_tokens)
        else:
            reduce_dims = tuple(range(1, score.dim() - 1))
            if reduce_dims:
                score = score.mean(dim=reduce_dims)

        if score.shape[0] == 1 and batch_size > 1:
            score = score.expand(batch_size, -1)
        if tuple(score.shape) != (batch_size, num_tokens):
            return None
        return score

    def _action_score(self, visual_tokens: Any, action_state: Any | None, torch: Any) -> Any | None:
        if action_state is None or not isinstance(action_state, torch.Tensor):
            return None

        batch_size, num_tokens, hidden_dim = visual_tokens.shape
        state = action_state.float()

        if state.dim() == 1 and state.shape[0] == hidden_dim:
            state = state.view(1, 1, hidden_dim).expand(batch_size, num_tokens, hidden_dim)
        elif state.dim() == 2 and state.shape[-1] == hidden_dim:
            if state.shape[0] not in (1, batch_size):
                return None
            state = state.view(state.shape[0], 1, hidden_dim).expand(batch_size, num_tokens, hidden_dim)
        elif state.dim() == 3 and state.shape[-1] == hidden_dim:
            if state.shape[0] not in (1, batch_size):
                return None
            if state.shape[1] not in (1, num_tokens):
                return None
            state = state.expand(batch_size, num_tokens, hidden_dim)
        else:
            return None

        return torch.sum(visual_tokens.float() * state, dim=-1).abs() / math.sqrt(hidden_dim)

    def _metadata(
        self,
        original_tokens: int,
        kept_tokens: int,
        input_shape: tuple[int, int, int],
        output_shape: tuple[int, int, int],
        pruned: bool,
        timestep: int | None,
        used_attention: bool,
        used_action_score: bool,
        used_temporal_smoothing: bool,
        selected_indices_shape: list[int],
        selected_indices_preview: list[int] | None = None,
        fallback_reason: str | None = None,
    ) -> dict[str, Any]:
        token_reduction_ratio = 0.0 if original_tokens == 0 else 1.0 - kept_tokens / original_tokens
        metadata = self._build_metadata(
            original_tokens=original_tokens,
            kept_tokens=kept_tokens,
            input_shape=input_shape,
            output_shape=output_shape,
            pruned=pruned,
            timestep=timestep,
            score_mode=self.score_mode,
            alpha=self.alpha,
            beta=self.beta,
            temporal_momentum=self.temporal_momentum,
            token_reduction_ratio=token_reduction_ratio,
            pruning_method=self.method_name,
            used_attention=used_attention,
            used_action_score=used_action_score,
            used_temporal_smoothing=used_temporal_smoothing,
            selected_indices_shape=selected_indices_shape,
        )
        if selected_indices_preview is not None:
            metadata["selected_indices_preview"] = selected_indices_preview
        if fallback_reason is not None:
            metadata["fallback_reason"] = fallback_reason
        return metadata


def _torch_or_none() -> Any | None:
    try:
        import torch
    except Exception:
        return None
    return torch


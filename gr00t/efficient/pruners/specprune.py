# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training-free SpecPrune-VLA MVP for visual-token pruning experiments."""

from __future__ import annotations

from typing import Any

from gr00t.efficient.pruners.base import VisualTokenPruner
from gr00t.efficient.pruners.utils import first_k_indices, gather_tokens, random_indices


class SpecPruneVLA(VisualTokenPruner):
    """MVP top-k visual-token pruner with cached index reuse."""

    method_name = "specprune"
    _VALID_SCORE_MODES = {"norm", "mean_abs", "random"}

    def __init__(
        self,
        keep_ratio: float = 0.75,
        reuse_steps: int = 2,
        score_mode: str = "norm",
        temporal_momentum: float = 0.8,
        enabled: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__(keep_ratio=keep_ratio, enabled=enabled, method_name=self.method_name)
        if reuse_steps < 1:
            raise ValueError("reuse_steps must be >= 1.")
        if score_mode not in self._VALID_SCORE_MODES:
            raise ValueError(
                f"Unsupported score_mode {score_mode!r}. Use one of {sorted(self._VALID_SCORE_MODES)}."
            )
        if not 0.0 <= temporal_momentum <= 1.0:
            raise ValueError("temporal_momentum must be in the interval [0, 1].")
        self.reuse_steps = reuse_steps
        self.score_mode = score_mode
        self.temporal_momentum = temporal_momentum
        self.seed = seed
        self.prev_score: Any | None = None
        self.cached_indices: Any | None = None

    def reset(self) -> None:
        super().reset()
        self.prev_score = None
        self.cached_indices = None

    def prune(
        self,
        visual_tokens: Any,
        attention: Any | None = None,
        robot_state: Any | None = None,
        action_state: Any | None = None,
        timestep: int | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Prune [B, N, D] tokens, reusing cached indices between scoring steps."""
        input_shape = self._validate_visual_tokens(visual_tokens)
        original_tokens = input_shape[1]
        keep = self._compute_keep_count(original_tokens)

        if not self.enabled or self.keep_ratio >= 1.0 or keep == original_tokens:
            indices = first_k_indices(original_tokens, original_tokens)
            self.cached_indices = indices
            metadata = self._metadata(
                original_tokens=original_tokens,
                kept_tokens=original_tokens,
                input_shape=input_shape,
                output_shape=input_shape,
                pruned=False,
                timestep=timestep,
                reused_indices=False,
                selected_indices_shape=[original_tokens],
            )
            self.last_metadata = metadata
            return visual_tokens, metadata

        should_reuse = self._should_reuse(timestep, keep, input_shape[0])
        if should_reuse:
            selected_indices = self.cached_indices
            reused_indices = True
        else:
            selected_indices = self._select_indices(visual_tokens, input_shape, keep, timestep)
            self.cached_indices = selected_indices
            reused_indices = False

        pruned_tokens = gather_tokens(visual_tokens, selected_indices)
        output_shape = self._validate_visual_tokens(pruned_tokens)
        metadata = self._metadata(
            original_tokens=original_tokens,
            kept_tokens=keep,
            input_shape=input_shape,
            output_shape=output_shape,
            pruned=True,
            timestep=timestep,
            reused_indices=reused_indices,
            selected_indices_shape=_indices_shape(selected_indices),
            selected_indices_preview=_indices_preview(selected_indices),
        )
        self.last_metadata = metadata
        return pruned_tokens, metadata

    def _should_reuse(
        self,
        timestep: int | None,
        keep: int,
        batch_size: int,
    ) -> bool:
        if timestep is None or timestep % self.reuse_steps == 0 or self.cached_indices is None:
            return False

        shape = _indices_shape(self.cached_indices)
        return shape in ([keep], [batch_size, keep])

    def _select_indices(
        self,
        visual_tokens: Any,
        input_shape: tuple[int, int, int],
        keep: int,
        timestep: int | None,
    ) -> Any:
        torch = _torch_or_none()
        if torch is None or not isinstance(visual_tokens, torch.Tensor):
            if self.score_mode == "random":
                return random_indices(input_shape[1], keep, seed=self.seed + int(timestep or 0))
            return first_k_indices(input_shape[1], keep)

        if self.score_mode == "random":
            generator = torch.Generator(device=visual_tokens.device)
            generator.manual_seed(self.seed + int(timestep or 0))
            score = torch.rand(
                (input_shape[0], input_shape[1]),
                generator=generator,
                device=visual_tokens.device,
            )
        elif self.score_mode == "mean_abs":
            score = torch.mean(torch.abs(visual_tokens.float()), dim=-1)
        else:
            score = torch.norm(visual_tokens.float(), dim=-1)

        if self.prev_score is not None and tuple(self.prev_score.shape) == tuple(score.shape):
            score = self.temporal_momentum * self.prev_score + (1.0 - self.temporal_momentum) * score
        self.prev_score = score.detach()

        topk_indices = torch.topk(score, k=keep, dim=1, largest=True, sorted=False).indices
        return torch.sort(topk_indices, dim=1).values

    def _metadata(
        self,
        original_tokens: int,
        kept_tokens: int,
        input_shape: tuple[int, int, int],
        output_shape: tuple[int, int, int],
        pruned: bool,
        timestep: int | None,
        reused_indices: bool,
        selected_indices_shape: list[int],
        selected_indices_preview: list[int] | None = None,
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
            reuse_steps=self.reuse_steps,
            reused_indices=reused_indices,
            temporal_momentum=self.temporal_momentum,
            token_reduction_ratio=token_reduction_ratio,
            pruning_method=self.method_name,
            selected_indices_shape=selected_indices_shape,
        )
        if selected_indices_preview is not None:
            metadata["selected_indices_preview"] = selected_indices_preview
        return metadata


def _torch_or_none() -> Any | None:
    try:
        import torch
    except Exception:
        return None
    return torch


def _indices_shape(indices: Any) -> list[int]:
    shape = getattr(indices, "shape", None)
    if shape is not None:
        return [int(dim) for dim in shape]
    return [len(indices)]


def _indices_preview(indices: Any) -> list[int]:
    if hasattr(indices, "detach"):
        value = indices.detach().cpu()
        if value.dim() == 2:
            value = value[0]
        return [int(idx) for idx in value[:16].tolist()]
    return [int(idx) for idx in list(indices)[:16]]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dummy visual token pruner for pipeline testing only."""

from __future__ import annotations

from typing import Any

from gr00t.efficient.pruners.base import VisualTokenPruner
from gr00t.efficient.pruners.utils import (
    first_k_indices,
    gather_tokens,
    random_indices,
    uniform_indices,
)


class DummyVisualTokenPruner(VisualTokenPruner):
    """Select a subset of visual tokens for pipeline testing.

    This is intentionally simple and should only be used to test benchmark
    plumbing. It is not VLA-Pruner, SpecPrune-VLA, ADP, or a model-quality
    pruning method.
    """

    method_name = "dummy"
    _VALID_MODES = {"first", "uniform", "random"}

    def __init__(
        self,
        keep_ratio: float = 1.0,
        mode: str = "first",
        seed: int = 0,
        enabled: bool = True,
    ) -> None:
        super().__init__(keep_ratio=keep_ratio, enabled=enabled, method_name=self.method_name)
        if mode not in self._VALID_MODES:
            raise ValueError(f"Unsupported dummy pruning mode {mode!r}. Use one of {sorted(self._VALID_MODES)}.")
        self.mode = mode
        self.seed = seed
        self.last_selected_indices: list[int] | None = None

    def prune(
        self,
        visual_tokens: Any,
        attention: Any | None = None,
        robot_state: Any | None = None,
        action_state: Any | None = None,
        timestep: int | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Select visual tokens along the token axis of [B, N, D]."""
        input_shape = self._validate_visual_tokens(visual_tokens)
        original_tokens = input_shape[1]
        keep = self._compute_keep_count(original_tokens)

        if not self.enabled or self.keep_ratio >= 1.0 or keep == original_tokens:
            self.last_selected_indices = list(range(original_tokens))
            metadata = self._build_metadata(
                original_tokens=original_tokens,
                kept_tokens=original_tokens,
                input_shape=input_shape,
                output_shape=input_shape,
                pruned=False,
                timestep=timestep,
                mode=self.mode,
                selected_indices_shape=[original_tokens],
                pruning_method=f"dummy_{self.mode}",
            )
            self.last_metadata = metadata
            return visual_tokens, metadata

        selected_indices = self._select_indices(original_tokens, keep)
        pruned_tokens = gather_tokens(visual_tokens, selected_indices)
        output_shape = self._validate_visual_tokens(pruned_tokens)
        self.last_selected_indices = selected_indices

        metadata = self._build_metadata(
            original_tokens=original_tokens,
            kept_tokens=keep,
            input_shape=input_shape,
            output_shape=output_shape,
            pruned=True,
            timestep=timestep,
            mode=self.mode,
            selected_indices_shape=[len(selected_indices)],
            selected_indices_preview=selected_indices[:16],
            pruning_method=f"dummy_{self.mode}",
        )
        self.last_metadata = metadata
        return pruned_tokens, metadata

    def _select_indices(self, original_tokens: int, keep: int) -> list[int]:
        if self.mode == "first":
            return first_k_indices(original_tokens, keep)
        if self.mode == "uniform":
            return uniform_indices(original_tokens, keep)
        if self.mode == "random":
            return random_indices(original_tokens, keep, seed=self.seed)
        raise ValueError(f"Unsupported dummy pruning mode {self.mode!r}.")

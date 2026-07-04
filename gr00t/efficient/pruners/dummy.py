# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dummy visual token pruner for pipeline testing only."""

from __future__ import annotations

import math
from typing import Any

from gr00t.efficient.profiler.token import get_visual_token_count
from gr00t.efficient.pruners.base import VisualTokenPruner


class DummyVisualTokenPruner(VisualTokenPruner):
    """Keep the first K visual tokens.

    This is intentionally simple and should only be used to test benchmark
    plumbing. It is not VLA-Pruner, SpecPrune-VLA, ADP, or a model-quality
    pruning method.
    """

    method = "dummy_first_k"

    def __init__(self, keep_ratio: float = 1.0) -> None:
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in the interval (0, 1].")
        self.keep_ratio = keep_ratio

    def prune(
        self,
        visual_tokens: Any,
        attention: Any | None = None,
        robot_state: Any | None = None,
        action_state: Any | None = None,
        timestep: int | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Slice the token dimension to the first K tokens for pipeline tests."""
        original_tokens = get_visual_token_count(visual_tokens)
        if original_tokens is None:
            kept_tokens = None
            pruned_tokens = visual_tokens
        else:
            kept_tokens = self._kept_token_count(original_tokens)
            pruned_tokens = self._slice_first_k(visual_tokens, kept_tokens)

        metadata = {
            "method": self.method,
            "keep_ratio": self.keep_ratio,
            "original_tokens": original_tokens,
            "kept_tokens": kept_tokens,
            "timestep": timestep,
        }
        return pruned_tokens, metadata

    def _kept_token_count(self, original_tokens: int) -> int:
        if original_tokens <= 0:
            return 0
        return max(1, min(original_tokens, int(math.ceil(original_tokens * self.keep_ratio))))

    @staticmethod
    def _slice_first_k(visual_tokens: Any, kept_tokens: int) -> Any:
        shape = getattr(visual_tokens, "shape", None)
        if shape is not None:
            try:
                ndim = len(shape)
                if ndim == 0:
                    return visual_tokens
                token_axis = -2 if ndim >= 2 else 0
                slices = [slice(None)] * ndim
                slices[token_axis] = slice(0, kept_tokens)
                return visual_tokens[tuple(slices)]
            except Exception:
                return visual_tokens

        try:
            return visual_tokens[:kept_tokens]
        except Exception:
            return visual_tokens


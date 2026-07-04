# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base interface for visual token pruning experiments."""

from __future__ import annotations

from typing import Any

from gr00t.efficient.profiler.token import get_token_shape
from gr00t.efficient.pruners.utils import compute_keep_count


class VisualTokenPruner:
    """Base visual-token pruner with safe no-op behavior."""

    method_name = "none"

    def __init__(
        self,
        keep_ratio: float = 1.0,
        enabled: bool = True,
        method_name: str | None = None,
    ) -> None:
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in the interval (0, 1].")
        self.keep_ratio = keep_ratio
        self.enabled = enabled
        self.method_name = method_name or self.method_name
        self.method = self.method_name
        self.last_metadata: dict[str, Any] | None = None

    def reset(self) -> None:
        """Reset per-episode state."""
        self.last_metadata = None
        return None

    def prune(
        self,
        visual_tokens: Any,
        attention: Any | None = None,
        robot_state: Any | None = None,
        action_state: Any | None = None,
        timestep: int | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Return visual tokens unchanged with standardized metadata."""
        shape = self._validate_visual_tokens(visual_tokens)
        metadata = self._build_metadata(
            original_tokens=shape[1],
            kept_tokens=shape[1],
            input_shape=shape,
            output_shape=shape,
            pruned=False,
            timestep=timestep,
        )
        self.last_metadata = metadata
        return visual_tokens, metadata

    def _validate_visual_tokens(self, visual_tokens: Any) -> tuple[int, int, int]:
        """Validate that visual tokens look like [B, N, D]."""
        shape = get_token_shape(visual_tokens)
        if shape is None:
            raise ValueError("visual_tokens must expose a tensor-like shape [B, N, D].")
        if len(shape) != 3:
            raise ValueError(f"Expected visual_tokens shape [B, N, D], got {shape}.")
        batch_size, num_tokens, hidden_dim = shape
        if batch_size < 1:
            raise ValueError(f"Expected batch size >= 1, got {batch_size}.")
        if num_tokens < 0:
            raise ValueError(f"Expected token count >= 0, got {num_tokens}.")
        if hidden_dim < 1:
            raise ValueError(f"Expected hidden dim >= 1, got {hidden_dim}.")
        return shape

    def _compute_keep_count(self, num_tokens: int) -> int:
        """Compute how many visual tokens this pruner should keep."""
        return compute_keep_count(num_tokens, self.keep_ratio)

    def _build_metadata(
        self,
        original_tokens: int,
        kept_tokens: int,
        input_shape: tuple[int, int, int],
        output_shape: tuple[int, int, int],
        pruned: bool,
        timestep: int | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        effective_keep_ratio = 0.0 if original_tokens == 0 else kept_tokens / original_tokens
        metadata = {
            "method": self.method_name,
            "original_tokens": original_tokens,
            "kept_tokens": kept_tokens,
            "keep_ratio": self.keep_ratio,
            "effective_keep_ratio": effective_keep_ratio,
            "enabled": self.enabled,
            "pruned": pruned,
            "input_shape": list(input_shape),
            "output_shape": list(output_shape),
            "token_order_preserved": True,
            "timestep": timestep,
        }
        metadata.update(extra)
        return metadata

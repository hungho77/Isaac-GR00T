# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base interface for visual token pruning experiments."""

from __future__ import annotations

from typing import Any

from gr00t.efficient.profiler.token import get_visual_token_count


class VisualTokenPruner:
    """No-op visual token pruner used as the safe default."""

    method = "none"

    def reset(self) -> None:
        """Reset per-episode state."""
        return None

    def prune(
        self,
        visual_tokens: Any,
        attention: Any | None = None,
        robot_state: Any | None = None,
        action_state: Any | None = None,
        timestep: int | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Return visual tokens unchanged with lightweight metadata."""
        token_count = get_visual_token_count(visual_tokens)
        metadata = {
            "method": self.method,
            "original_tokens": token_count,
            "kept_tokens": token_count,
            "timestep": timestep,
            "pruned": False,
        }
        return visual_tokens, metadata


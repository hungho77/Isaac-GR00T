# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base scheduler interface for future adaptive efficient inference methods."""

from __future__ import annotations

from typing import Any


class EfficientInferenceScheduler:
    """No-op scheduler placeholder for adaptive pruning experiments."""

    method = "none"

    def reset(self) -> None:
        """Reset per-episode scheduler state."""
        return None

    def step(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return unchanged scheduling metadata for scaffolded benchmarks."""
        return {"method": self.method, "metadata": metadata or {}}


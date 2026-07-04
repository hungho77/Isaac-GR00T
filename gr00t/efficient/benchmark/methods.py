# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Method wrappers for efficient inference benchmark runs."""

from __future__ import annotations

from typing import Any

from gr00t.efficient.profiler.token import get_visual_token_count
from gr00t.efficient.pruners.dummy import DummyVisualTokenPruner


class EfficientInferenceMethod:
    """Base no-op method interface used by benchmark runners."""

    method_name = "base"
    keep_ratio = 1.0

    def reset(self) -> None:
        """Reset any method-level state."""
        return None

    def before_episode(self, episode_id: int | None = None, task: str | None = None) -> None:
        """Prepare method state before an episode."""
        return None

    def after_episode(self, episode_id: int | None = None, task: str | None = None) -> None:
        """Finalize method state after an episode."""
        return None

    def process_visual_tokens(
        self,
        visual_tokens: Any,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """Return visual tokens unchanged plus method metadata."""
        token_count = get_visual_token_count(visual_tokens)
        metadata = {
            "method": self.method_name,
            "pruning_method": "none",
            "keep_ratio": self.keep_ratio,
            "original_tokens": token_count,
            "kept_tokens": token_count,
            "pruning_enabled": False,
        }
        metadata.update({key: value for key, value in kwargs.items() if value is not None})
        return visual_tokens, metadata

    def update_metrics(self, record: Any) -> Any:
        """Attach method identity to a benchmark record."""
        if isinstance(record, dict):
            record["method"] = self.method_name
            record["keep_ratio"] = self.keep_ratio
            return record

        if hasattr(record, "method"):
            record.method = self.method_name
        if hasattr(record, "keep_ratio"):
            record.keep_ratio = self.keep_ratio
        return record

    def metadata(self) -> dict[str, Any]:
        """Return JSON-safe method metadata."""
        return {
            "method": self.method_name,
            "keep_ratio": self.keep_ratio,
            "pruning_enabled": False,
        }


class BaselineMethod(EfficientInferenceMethod):
    """No-op baseline method that preserves GR00T behavior."""

    method_name = "baseline"

    def __init__(self, keep_ratio: float = 1.0, **_: Any) -> None:
        if keep_ratio != 1.0:
            raise ValueError("BaselineMethod requires keep_ratio=1.0.")
        self.keep_ratio = 1.0


class DummyPruningMethod(EfficientInferenceMethod):
    """Dummy visual-token pruning wrapper for explicit pipeline tests only."""

    method_name = "dummy"

    def __init__(self, keep_ratio: float = 1.0, **_: Any) -> None:
        self.keep_ratio = keep_ratio
        self.pruner = DummyVisualTokenPruner(keep_ratio=keep_ratio)

    def reset(self) -> None:
        self.pruner.reset()

    def process_visual_tokens(
        self,
        visual_tokens: Any,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        pruned_tokens, pruner_metadata = self.pruner.prune(
            visual_tokens,
            attention=kwargs.get("attention"),
            robot_state=kwargs.get("robot_state"),
            action_state=kwargs.get("action_state"),
            timestep=kwargs.get("timestep"),
        )
        metadata = {
            "method": self.method_name,
            "pruning_method": pruner_metadata.get("method", "dummy"),
            "keep_ratio": self.keep_ratio,
            "original_tokens": pruner_metadata.get("original_tokens"),
            "kept_tokens": pruner_metadata.get("kept_tokens"),
            "pruning_enabled": True,
        }
        metadata.update(
            {
                key: value
                for key, value in kwargs.items()
                if key not in {"attention", "robot_state", "action_state"} and value is not None
            }
        )
        return pruned_tokens, metadata

    def metadata(self) -> dict[str, Any]:
        return {
            "method": self.method_name,
            "keep_ratio": self.keep_ratio,
            "pruning_enabled": True,
            "pruning_method": self.pruner.method,
            "notes": "Dummy first-K slicing is only for benchmark pipeline testing.",
        }

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
        self.last_pruning_metadata = None
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
            "effective_keep_ratio": 1.0,
            "original_tokens": token_count,
            "kept_tokens": token_count,
            "pruning_enabled": False,
            "pruned": False,
        }
        metadata.update({key: value for key, value in kwargs.items() if value is not None})
        self.last_pruning_metadata = metadata
        return visual_tokens, metadata

    def update_metrics(self, record: Any) -> Any:
        """Attach method identity and pruning metadata to a benchmark record."""
        metadata = getattr(self, "last_pruning_metadata", None) or {}
        before = metadata.get("original_tokens")
        after = metadata.get("kept_tokens", before)
        reduction = 0.0 if not before else 1.0 - (float(after) / float(before))
        pruning_method = metadata.get("pruning_method", "none")

        if isinstance(record, dict):
            record["method"] = self.method_name
            record["keep_ratio"] = self.keep_ratio
            record["visual_token_count_before"] = before
            record["visual_token_count_after"] = after
            record["token_reduction_ratio"] = reduction
            record["pruning_method"] = pruning_method
            return record

        if hasattr(record, "method"):
            record.method = self.method_name
        if hasattr(record, "keep_ratio"):
            record.keep_ratio = self.keep_ratio
        if hasattr(record, "visual_token_count_before"):
            record.visual_token_count_before = before
        if hasattr(record, "visual_token_count_after"):
            record.visual_token_count_after = after
        if hasattr(record, "token_reduction_ratio"):
            record.token_reduction_ratio = reduction
        if hasattr(record, "pruning_method"):
            record.pruning_method = pruning_method
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
        self.last_pruning_metadata: dict[str, Any] | None = None


class DummyPruningMethod(EfficientInferenceMethod):
    """Dummy visual-token pruning wrapper for explicit pipeline tests only."""

    method_name = "dummy"

    def __init__(
        self,
        keep_ratio: float = 1.0,
        mode: str = "first",
        seed: int = 0,
        enabled: bool = True,
        **_: Any,
    ) -> None:
        self.keep_ratio = keep_ratio
        self.mode = mode
        self.pruner = DummyVisualTokenPruner(
            keep_ratio=keep_ratio,
            mode=mode,
            seed=seed,
            enabled=enabled,
        )
        self.last_pruning_metadata: dict[str, Any] | None = None

    def reset(self) -> None:
        self.pruner.reset()
        self.last_pruning_metadata = None

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
            "pruning_method": pruner_metadata.get("pruning_method", "dummy"),
            "mode": self.mode,
            "keep_ratio": self.keep_ratio,
            "effective_keep_ratio": pruner_metadata.get("effective_keep_ratio"),
            "original_tokens": pruner_metadata.get("original_tokens"),
            "kept_tokens": pruner_metadata.get("kept_tokens"),
            "pruning_enabled": True,
            "pruned": pruner_metadata.get("pruned", False),
        }
        metadata.update(
            {
                key: value
                for key, value in kwargs.items()
                if key not in {"attention", "robot_state", "action_state"} and value is not None
            }
        )
        metadata["pruner_metadata"] = pruner_metadata
        self.last_pruning_metadata = metadata
        return pruned_tokens, metadata

    def metadata(self) -> dict[str, Any]:
        return {
            "method": self.method_name,
            "keep_ratio": self.keep_ratio,
            "pruning_enabled": True,
            "mode": self.mode,
            "pruning_method": f"dummy_{self.mode}",
            "notes": "Dummy token selection is only for benchmark pipeline testing.",
        }

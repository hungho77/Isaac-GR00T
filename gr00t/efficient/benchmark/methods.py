# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Method wrappers for efficient inference benchmark runs."""

from __future__ import annotations

from typing import Any

from gr00t.efficient.profiler.token import get_visual_token_count
from gr00t.efficient.pruners.dummy import DummyVisualTokenPruner
from gr00t.efficient.pruners.specprune import SpecPruneVLA
from gr00t.efficient.pruners.vlapruner import VLAPruner
from gr00t.efficient.schedulers.adp import ADPScheduler


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
        effective_keep_ratio = metadata.get("effective_keep_ratio")
        score_mode = metadata.get("score_mode", "none")
        scheduler = metadata.get("scheduler", "none")
        scheduler_reason = metadata.get("scheduler_reason", "")
        dynamic_keep_ratio = metadata.get("dynamic_keep_ratio")
        avg_keep_ratio = metadata.get("avg_keep_ratio")
        reuse_steps = metadata.get("reuse_steps")
        reused_indices = metadata.get("reused_indices")

        if isinstance(record, dict):
            record["method"] = self.method_name
            record["keep_ratio"] = self.keep_ratio
            record["visual_token_count_before"] = before
            record["visual_token_count_after"] = after
            record["token_reduction_ratio"] = reduction
            record["pruning_method"] = pruning_method
            record["effective_keep_ratio"] = effective_keep_ratio
            record["score_mode"] = score_mode
            record["scheduler"] = scheduler
            record["scheduler_reason"] = scheduler_reason
            record["dynamic_keep_ratio"] = dynamic_keep_ratio
            record["avg_keep_ratio"] = avg_keep_ratio
            record["reuse_steps"] = reuse_steps
            record["reused_indices"] = reused_indices
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
        if hasattr(record, "effective_keep_ratio"):
            record.effective_keep_ratio = effective_keep_ratio
        if hasattr(record, "score_mode"):
            record.score_mode = score_mode
        if hasattr(record, "scheduler"):
            record.scheduler = scheduler
        if hasattr(record, "scheduler_reason"):
            record.scheduler_reason = scheduler_reason
        if hasattr(record, "dynamic_keep_ratio"):
            record.dynamic_keep_ratio = dynamic_keep_ratio
        if hasattr(record, "avg_keep_ratio"):
            record.avg_keep_ratio = avg_keep_ratio
        if hasattr(record, "reuse_steps"):
            record.reuse_steps = reuse_steps
        if hasattr(record, "reused_indices"):
            record.reused_indices = reused_indices
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


class VLAPrunerMethod(EfficientInferenceMethod):
    """Training-free VLA-Pruner MVP method wrapper."""

    method_name = "vlapruner"

    def __init__(
        self,
        keep_ratio: float = 0.75,
        score_mode: str = "norm",
        alpha: float = 0.5,
        beta: float = 0.5,
        temporal_momentum: float = 0.8,
        enabled: bool = True,
        seed: int = 0,
        **_: Any,
    ) -> None:
        self.keep_ratio = keep_ratio
        self.score_mode = score_mode
        self.alpha = alpha
        self.beta = beta
        self.temporal_momentum = temporal_momentum
        self.pruner = VLAPruner(
            keep_ratio=keep_ratio,
            alpha=alpha,
            beta=beta,
            temporal_momentum=temporal_momentum,
            score_mode=score_mode,
            enabled=enabled,
            seed=seed,
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
            "pruning_method": pruner_metadata.get("pruning_method", self.method_name),
            "score_mode": self.score_mode,
            "keep_ratio": self.keep_ratio,
            "effective_keep_ratio": pruner_metadata.get("effective_keep_ratio"),
            "token_reduction_ratio": pruner_metadata.get("token_reduction_ratio"),
            "original_tokens": pruner_metadata.get("original_tokens"),
            "kept_tokens": pruner_metadata.get("kept_tokens"),
            "pruning_enabled": True,
            "pruned": pruner_metadata.get("pruned", False),
            "used_attention": pruner_metadata.get("used_attention", False),
            "used_action_score": pruner_metadata.get("used_action_score", False),
            "used_temporal_smoothing": pruner_metadata.get("used_temporal_smoothing", False),
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
            "pruning_method": self.method_name,
            "score_mode": self.score_mode,
            "alpha": self.alpha,
            "beta": self.beta,
            "temporal_momentum": self.temporal_momentum,
            "notes": "Training-free VLA-Pruner MVP; real GR00T hooks are not connected yet.",
        }


class SpecPruneMethod(EfficientInferenceMethod):
    """Training-free SpecPrune-VLA MVP method wrapper."""

    method_name = "specprune"

    def __init__(
        self,
        keep_ratio: float = 0.75,
        score_mode: str = "norm",
        reuse_steps: int = 2,
        temporal_momentum: float = 0.8,
        enabled: bool = True,
        seed: int = 0,
        **_: Any,
    ) -> None:
        self.keep_ratio = keep_ratio
        self.score_mode = score_mode
        self.reuse_steps = reuse_steps
        self.temporal_momentum = temporal_momentum
        self.pruner = SpecPruneVLA(
            keep_ratio=keep_ratio,
            reuse_steps=reuse_steps,
            score_mode=score_mode,
            temporal_momentum=temporal_momentum,
            enabled=enabled,
            seed=seed,
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
            "pruning_method": pruner_metadata.get("pruning_method", self.method_name),
            "score_mode": self.score_mode,
            "keep_ratio": self.keep_ratio,
            "effective_keep_ratio": pruner_metadata.get("effective_keep_ratio"),
            "token_reduction_ratio": pruner_metadata.get("token_reduction_ratio"),
            "original_tokens": pruner_metadata.get("original_tokens"),
            "kept_tokens": pruner_metadata.get("kept_tokens"),
            "reuse_steps": pruner_metadata.get("reuse_steps"),
            "reused_indices": pruner_metadata.get("reused_indices"),
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
            "pruning_method": self.method_name,
            "score_mode": self.score_mode,
            "reuse_steps": self.reuse_steps,
            "temporal_momentum": self.temporal_momentum,
            "notes": "Training-free SpecPrune-VLA MVP with token-index reuse.",
        }


class ADPMethod(EfficientInferenceMethod):
    """Adaptive dynamic pruning MVP using a simple scheduler and dummy selector."""

    method_name = "adp"

    def __init__(
        self,
        default_keep_ratio: float = 0.5,
        contact_keep_ratio: float = 1.0,
        move_keep_ratio: float = 0.6,
        idle_keep_ratio: float = 0.5,
        action_delta_threshold: float = 0.05,
        mode: str = "first",
        enabled: bool = True,
        **_: Any,
    ) -> None:
        self.default_keep_ratio = default_keep_ratio
        self.keep_ratio = default_keep_ratio
        self.mode = mode
        self.enabled = enabled
        self.scheduler = ADPScheduler(
            default_keep_ratio=default_keep_ratio,
            contact_keep_ratio=contact_keep_ratio,
            move_keep_ratio=move_keep_ratio,
            idle_keep_ratio=idle_keep_ratio,
            action_delta_threshold=action_delta_threshold,
        )
        self.last_pruning_metadata: dict[str, Any] | None = None

    def reset(self) -> None:
        self.scheduler.reset()
        self.last_pruning_metadata = None

    def process_visual_tokens(
        self,
        visual_tokens: Any,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        dynamic_keep_ratio, scheduler_metadata = self.scheduler.get_keep_ratio(
            robot_state=kwargs.get("robot_state"),
            action_state=kwargs.get("action_state"),
            prev_action_state=kwargs.get("prev_action_state"),
            timestep=kwargs.get("timestep"),
        )
        self.keep_ratio = dynamic_keep_ratio
        pruner = DummyVisualTokenPruner(
            keep_ratio=dynamic_keep_ratio,
            mode=self.mode,
            enabled=self.enabled,
        )
        pruned_tokens, pruner_metadata = pruner.prune(
            visual_tokens,
            timestep=kwargs.get("timestep"),
        )
        metadata = self._build_adp_metadata(
            pruner_metadata=pruner_metadata,
            scheduler_metadata=scheduler_metadata,
            pruning_method=self.method_name,
        )
        metadata.update(
            {
                key: value
                for key, value in kwargs.items()
                if key not in {"attention", "robot_state", "action_state", "prev_action_state"}
                and value is not None
            }
        )
        self.last_pruning_metadata = metadata
        return pruned_tokens, metadata

    def _build_adp_metadata(
        self,
        pruner_metadata: dict[str, Any],
        scheduler_metadata: dict[str, Any],
        pruning_method: str,
    ) -> dict[str, Any]:
        original_tokens = pruner_metadata.get("original_tokens")
        kept_tokens = pruner_metadata.get("kept_tokens")
        token_reduction_ratio = (
            0.0
            if not original_tokens
            else 1.0 - (float(kept_tokens) / float(original_tokens))
        )
        return {
            "method": self.method_name,
            "pruning_method": pruning_method,
            "keep_ratio": scheduler_metadata["keep_ratio"],
            "dynamic_keep_ratio": scheduler_metadata["dynamic_keep_ratio"],
            "effective_keep_ratio": pruner_metadata.get("effective_keep_ratio"),
            "token_reduction_ratio": token_reduction_ratio,
            "original_tokens": original_tokens,
            "kept_tokens": kept_tokens,
            "scheduler": scheduler_metadata["scheduler"],
            "scheduler_reason": scheduler_metadata["scheduler_reason"],
            "action_delta": scheduler_metadata.get("action_delta"),
            "pruning_enabled": True,
            "pruned": pruner_metadata.get("pruned", False),
            "pruner_metadata": pruner_metadata,
            "scheduler_metadata": scheduler_metadata,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "method": self.method_name,
            "keep_ratio": self.default_keep_ratio,
            "pruning_enabled": True,
            "pruning_method": self.method_name,
            "scheduler": "adp",
            "notes": "ADP MVP uses mock action-state heuristics for dynamic keep ratios.",
        }


class ADPVLAPrunerMethod(ADPMethod):
    """Hybrid ADP scheduler plus VLA-Pruner MVP selector."""

    method_name = "adp_vlapruner"

    def __init__(
        self,
        score_mode: str = "norm",
        alpha: float = 0.5,
        beta: float = 0.5,
        temporal_momentum: float = 0.8,
        default_keep_ratio: float = 0.5,
        contact_keep_ratio: float = 1.0,
        move_keep_ratio: float = 0.6,
        idle_keep_ratio: float = 0.5,
        action_delta_threshold: float = 0.05,
        enabled: bool = True,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            default_keep_ratio=default_keep_ratio,
            contact_keep_ratio=contact_keep_ratio,
            move_keep_ratio=move_keep_ratio,
            idle_keep_ratio=idle_keep_ratio,
            action_delta_threshold=action_delta_threshold,
            enabled=enabled,
            **kwargs,
        )
        self.method_name = "adp_vlapruner"
        self.score_mode = score_mode
        self.alpha = alpha
        self.beta = beta
        self.temporal_momentum = temporal_momentum
        self.vlapruner = VLAPruner(
            keep_ratio=default_keep_ratio,
            alpha=alpha,
            beta=beta,
            temporal_momentum=temporal_momentum,
            score_mode=score_mode,
            enabled=enabled,
            seed=seed,
        )

    def reset(self) -> None:
        self.scheduler.reset()
        self.vlapruner.reset()
        self.last_pruning_metadata = None

    def process_visual_tokens(
        self,
        visual_tokens: Any,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        dynamic_keep_ratio, scheduler_metadata = self.scheduler.get_keep_ratio(
            robot_state=kwargs.get("robot_state"),
            action_state=kwargs.get("action_state"),
            prev_action_state=kwargs.get("prev_action_state"),
            timestep=kwargs.get("timestep"),
        )
        self.keep_ratio = dynamic_keep_ratio
        self.vlapruner.keep_ratio = dynamic_keep_ratio
        pruned_tokens, pruner_metadata = self.vlapruner.prune(
            visual_tokens,
            attention=kwargs.get("attention"),
            robot_state=kwargs.get("robot_state"),
            action_state=kwargs.get("action_state"),
            timestep=kwargs.get("timestep"),
        )
        metadata = self._build_adp_metadata(
            pruner_metadata=pruner_metadata,
            scheduler_metadata=scheduler_metadata,
            pruning_method=self.method_name,
        )
        metadata["score_mode"] = self.score_mode
        metadata["used_temporal_smoothing"] = pruner_metadata.get("used_temporal_smoothing", False)
        metadata.update(
            {
                key: value
                for key, value in kwargs.items()
                if key not in {"attention", "robot_state", "action_state", "prev_action_state"}
                and value is not None
            }
        )
        self.last_pruning_metadata = metadata
        return pruned_tokens, metadata

    def metadata(self) -> dict[str, Any]:
        return {
            "method": self.method_name,
            "keep_ratio": self.default_keep_ratio,
            "pruning_enabled": True,
            "pruning_method": self.method_name,
            "scheduler": "adp",
            "score_mode": self.score_mode,
            "alpha": self.alpha,
            "beta": self.beta,
            "temporal_momentum": self.temporal_momentum,
            "notes": "Hybrid ADP + VLA-Pruner MVP; real action hooks are not connected yet.",
        }

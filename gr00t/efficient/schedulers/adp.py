# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adaptive dynamic pruning scheduler MVP."""

from __future__ import annotations

import math
from numbers import Number
from typing import Any


class ADPScheduler:
    """Choose a keep ratio from lightweight action-state heuristics."""

    def __init__(
        self,
        default_keep_ratio: float = 0.5,
        contact_keep_ratio: float = 1.0,
        move_keep_ratio: float = 0.6,
        idle_keep_ratio: float = 0.5,
        action_delta_threshold: float = 0.05,
        gripper_index: int = -1,
    ) -> None:
        self.default_keep_ratio = _validate_keep_ratio(default_keep_ratio, "default_keep_ratio")
        self.contact_keep_ratio = _validate_keep_ratio(contact_keep_ratio, "contact_keep_ratio")
        self.move_keep_ratio = _validate_keep_ratio(move_keep_ratio, "move_keep_ratio")
        self.idle_keep_ratio = _validate_keep_ratio(idle_keep_ratio, "idle_keep_ratio")
        if action_delta_threshold < 0.0:
            raise ValueError("action_delta_threshold must be >= 0.")
        self.action_delta_threshold = action_delta_threshold
        self.gripper_index = gripper_index
        self.last_metadata: dict[str, Any] | None = None

    def reset(self) -> None:
        self.last_metadata = None

    def get_keep_ratio(
        self,
        robot_state: Any | None = None,
        action_state: Any | None = None,
        prev_action_state: Any | None = None,
        timestep: int | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Return a dynamic keep ratio and scheduler metadata."""
        del robot_state

        action_values = _flatten_numeric(action_state)
        if action_values is None:
            return self._metadata(
                keep_ratio=self.default_keep_ratio,
                reason="no_action_state",
                timestep=timestep,
                action_delta=None,
            )

        if _is_contact_or_gripper(action_values, self.gripper_index):
            return self._metadata(
                keep_ratio=self.contact_keep_ratio,
                reason="contact_or_gripper",
                timestep=timestep,
                action_delta=None,
            )

        action_delta = _action_delta(action_values, _flatten_numeric(prev_action_state))
        if action_delta is not None and action_delta > self.action_delta_threshold:
            return self._metadata(
                keep_ratio=self.move_keep_ratio,
                reason="moving",
                timestep=timestep,
                action_delta=action_delta,
            )

        return self._metadata(
            keep_ratio=self.idle_keep_ratio,
            reason="idle",
            timestep=timestep,
            action_delta=action_delta,
        )

    def _metadata(
        self,
        keep_ratio: float,
        reason: str,
        timestep: int | None,
        action_delta: float | None,
    ) -> tuple[float, dict[str, Any]]:
        metadata = {
            "scheduler": "adp",
            "reason": reason,
            "scheduler_reason": reason,
            "keep_ratio": keep_ratio,
            "dynamic_keep_ratio": keep_ratio,
            "timestep": timestep,
            "action_delta": action_delta,
        }
        self.last_metadata = metadata
        return keep_ratio, metadata


def _validate_keep_ratio(value: float, name: str) -> float:
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in the interval (0, 1].")
    return value


def _flatten_numeric(value: Any | None) -> list[float] | None:
    if value is None:
        return None

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, Number) and not isinstance(value, bool):
        return [float(value)]

    if isinstance(value, (str, bytes)):
        return None

    try:
        iterator = iter(value)
    except TypeError:
        return None

    flattened: list[float] = []
    for item in iterator:
        item_values = _flatten_numeric(item)
        if item_values is not None:
            flattened.extend(item_values)
    return flattened


def _is_contact_or_gripper(action_values: list[float], gripper_index: int) -> bool:
    if not action_values:
        return False
    index = gripper_index if gripper_index >= 0 else len(action_values) + gripper_index
    if index < 0 or index >= len(action_values):
        return False
    return action_values[index] > 0.5


def _action_delta(
    action_values: list[float],
    prev_action_values: list[float] | None,
) -> float | None:
    if prev_action_values is None or len(action_values) != len(prev_action_values):
        return None
    return math.sqrt(
        sum((value - prev_value) ** 2 for value, prev_value in zip(action_values, prev_action_values))
    )


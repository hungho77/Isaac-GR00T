# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight metric helpers for efficient inference benchmarks."""

from __future__ import annotations

import math
from numbers import Number
from typing import Any, Iterable


def _flatten_numeric(value: Any) -> Iterable[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, Number) and not isinstance(value, bool):
        yield float(value)
        return

    if isinstance(value, (str, bytes)):
        raise TypeError("Expected numeric action values, got string-like data.")

    try:
        iterator = iter(value)
    except TypeError as exc:
        raise TypeError(f"Expected numeric action values, got {type(value).__name__}.") from exc

    for item in iterator:
        yield from _flatten_numeric(item)


def compute_action_l2(action: Any, baseline_action: Any) -> float:
    """Compute L2 distance between an action and its baseline action."""
    action_values = list(_flatten_numeric(action))
    baseline_values = list(_flatten_numeric(baseline_action))
    if len(action_values) != len(baseline_values):
        raise ValueError(
            f"Action shapes differ after flattening: {len(action_values)} vs {len(baseline_values)}."
        )
    return math.sqrt(sum((value - base) ** 2 for value, base in zip(action_values, baseline_values)))


def summarize_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize numeric record fields with simple means."""
    summary: dict[str, Any] = {"num_records": len(records)}
    numeric_values: dict[str, list[float]] = {}
    failure_counts: dict[str, int] = {}

    for record in records:
        for key, value in record.items():
            if isinstance(value, Number) and not isinstance(value, bool):
                numeric_values.setdefault(key, []).append(float(value))

        failure_type = record.get("failure_type")
        if failure_type:
            failure_counts[str(failure_type)] = failure_counts.get(str(failure_type), 0) + 1

    for key, values in numeric_values.items():
        summary[f"mean_{key}"] = sum(values) / len(values)

    if failure_counts:
        summary["failure_counts"] = failure_counts

    return summary


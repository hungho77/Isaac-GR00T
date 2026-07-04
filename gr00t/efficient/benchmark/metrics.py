# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight metric helpers for efficient inference benchmarks."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, is_dataclass
import json
import math
from numbers import Number
from pathlib import Path
from typing import Any, Iterable


@dataclass
class BenchmarkRecord:
    """One episode-level efficient inference benchmark record."""

    benchmark: str
    method: str
    task: str
    episode_id: int
    success: bool
    success_rate: float
    latency_per_action_ms: float
    episode_time_s: float
    gpu_memory_mb: float
    peak_gpu_memory_mb: float
    visual_token_count: int | None
    keep_ratio: float
    action_l2_vs_baseline: float
    failure_type: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


RECORD_FIELDS = [
    "benchmark",
    "method",
    "task",
    "episode_id",
    "success",
    "success_rate",
    "latency_per_action_ms",
    "episode_time_s",
    "gpu_memory_mb",
    "peak_gpu_memory_mb",
    "visual_token_count",
    "keep_ratio",
    "action_l2_vs_baseline",
    "failure_type",
    "notes",
]

SUMMARY_NUMERIC_FIELDS = [
    "success_rate",
    "latency_per_action_ms",
    "episode_time_s",
    "gpu_memory_mb",
    "peak_gpu_memory_mb",
    "visual_token_count",
    "keep_ratio",
    "action_l2_vs_baseline",
]


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


def _record_to_dict(record: BenchmarkRecord | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, BenchmarkRecord):
        return record.to_dict()
    if is_dataclass(record):
        return asdict(record)
    return dict(record)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BenchmarkRecord):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: str | Path, data: Any) -> None:
    """Write JSON data, creating parent directories as needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(data), indent=2) + "\n", encoding="utf-8")


def write_csv(path: str | Path, records: Iterable[BenchmarkRecord | dict[str, Any]]) -> None:
    """Write benchmark records to CSV using the baseline metrics schema."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_record_to_dict(record) for record in records]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORD_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in RECORD_FIELDS})


def summarize_metrics(records: Iterable[BenchmarkRecord | dict[str, Any]]) -> dict[str, Any]:
    """Summarize episode records with means for the Day 2 target metrics."""
    rows = [_record_to_dict(record) for record in records]
    summary: dict[str, Any] = {"num_records": len(rows)}
    numeric_values: dict[str, list[float]] = {field: [] for field in SUMMARY_NUMERIC_FIELDS}
    failure_counts: dict[str, int] = {}

    for record in rows:
        for key in SUMMARY_NUMERIC_FIELDS:
            value = record.get(key)
            if isinstance(value, Number) and not isinstance(value, bool):
                numeric_values[key].append(float(value))

        failure_type = record.get("failure_type")
        if failure_type:
            failure_counts[str(failure_type)] = failure_counts.get(str(failure_type), 0) + 1

    for key, values in numeric_values.items():
        if values:
            summary[key] = sum(values) / len(values)

    if failure_counts:
        summary["failure_counts"] = failure_counts

    return summary

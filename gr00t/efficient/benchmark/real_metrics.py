# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metric helpers for real LIBERO efficient-inference runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

from gr00t.efficient.profiler.memory import get_cuda_memory_stats


def start_episode_record(
    *,
    benchmark: str,
    method: str,
    task: str,
    episode_id: int,
    keep_ratio: float,
    notes: str = "",
) -> dict[str, Any]:
    """Create a real-eval episode record with CSV-safe defaults."""
    return {
        "benchmark": benchmark,
        "method": method,
        "task": task,
        "episode_id": episode_id,
        "success": False,
        "success_rate": 0.0,
        "latency_per_action_ms": 0.0,
        "episode_time_s": 0.0,
        "gpu_memory_mb": 0.0,
        "peak_gpu_memory_mb": 0.0,
        "visual_token_count": None,
        "visual_token_count_before": None,
        "visual_token_count_after": None,
        "keep_ratio": keep_ratio,
        "effective_keep_ratio": keep_ratio,
        "token_reduction_ratio": 0.0,
        "action_l2_vs_baseline": 0.0,
        "failure_type": "",
        "notes": notes,
        "_episode_start_time": time.perf_counter(),
    }


def record_action_latency(latencies_ms: list[float], elapsed_ms: float) -> None:
    """Append one action latency sample in milliseconds."""
    latencies_ms.append(float(elapsed_ms))


def finish_episode_record(
    record: dict[str, Any],
    *,
    success: bool,
    success_rate: float | None = None,
    episode_time_s: float | None = None,
    action_latencies_ms: Iterable[float] | None = None,
    visual_metadata: dict[str, Any] | None = None,
    action_l2_vs_baseline: float | None = None,
    failure_type: str = "",
    notes: str | None = None,
) -> dict[str, Any]:
    """Finalize one real-eval episode record."""
    record["success"] = bool(success)
    record["success_rate"] = float(success_rate if success_rate is not None else int(success))
    if episode_time_s is None:
        start_time = record.get("_episode_start_time")
        if isinstance(start_time, (int, float)):
            episode_time_s = time.perf_counter() - float(start_time)
        else:
            episode_time_s = 0.0
    record["episode_time_s"] = float(episode_time_s)

    latencies = list(action_latencies_ms or [])
    if latencies:
        record["latency_per_action_ms"] = sum(latencies) / len(latencies)
        # Median is robust to one-time warmup outliers (e.g. torch.compile on first actions).
        ordered = sorted(latencies)
        mid = len(ordered) // 2
        record["latency_per_action_ms_median"] = (
            ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
        )

    memory = get_cuda_memory_stats()
    record["gpu_memory_mb"] = float(memory.get("gpu_memory_mb", memory.get("allocated_mb", 0.0)))
    record["peak_gpu_memory_mb"] = float(
        memory.get("peak_gpu_memory_mb", memory.get("peak_allocated_mb", 0.0))
    )

    if visual_metadata:
        before = visual_metadata.get("original_tokens")
        after = visual_metadata.get("kept_tokens", before)
        record["visual_token_count_before"] = before
        record["visual_token_count_after"] = after
        record["visual_token_count"] = after
        record["effective_keep_ratio"] = visual_metadata.get(
            "effective_keep_ratio", record["effective_keep_ratio"]
        )
        record["token_reduction_ratio"] = visual_metadata.get(
            "token_reduction_ratio",
            0.0 if not before else 1.0 - (float(after) / float(before)),
        )
        record["pruning_method"] = visual_metadata.get(
            "pruning_method", visual_metadata.get("method", "none")
        )
        record["score_mode"] = visual_metadata.get("score_mode", "none")
        record["score_mode_effective"] = visual_metadata.get(
            "score_mode_effective", record["score_mode"]
        )
        record["used_attention"] = visual_metadata.get("used_attention", False)
        record["used_action_score"] = visual_metadata.get("used_action_score", False)
        if "hook_error" in visual_metadata:
            record["hook_error"] = visual_metadata["hook_error"]
    elif record["keep_ratio"] == 1.0:
        record["effective_keep_ratio"] = 1.0
        record["token_reduction_ratio"] = 0.0

    if action_l2_vs_baseline is not None:
        record["action_l2_vs_baseline"] = float(action_l2_vs_baseline)
    record["failure_type"] = failure_type
    if notes is not None:
        record["notes"] = notes
    record.pop("_episode_start_time", None)
    return record


def compute_episode_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute a compact summary for real episode records."""
    rows = list(records)
    if not rows:
        return {"num_records": 0}

    numeric_fields = [
        "success_rate",
        "latency_per_action_ms",
        "latency_per_action_ms_median",
        "episode_time_s",
        "gpu_memory_mb",
        "peak_gpu_memory_mb",
        "visual_token_count_before",
        "visual_token_count_after",
        "keep_ratio",
        "effective_keep_ratio",
        "token_reduction_ratio",
        "action_l2_vs_baseline",
    ]
    summary: dict[str, Any] = {"num_records": len(rows)}
    for field in numeric_fields:
        values = [
            float(record[field])
            for record in rows
            if isinstance(record.get(field), (int, float))
            and not isinstance(record.get(field), bool)
        ]
        if values:
            summary[field] = sum(values) / len(values)
    return summary


def save_action_trace(
    path: str | Path, actions: Iterable[Any], metadata: dict[str, Any] | None = None
) -> None:
    """Save an action trace as NPZ plus small JSON metadata sidecar."""
    import numpy as np

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flattened = [_flatten_numeric_or_repr(action) for action in actions]
    np.savez_compressed(output_path, actions=np.array(flattened, dtype=object))
    if metadata is not None:
        output_path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )


def load_action_trace(path: str | Path) -> list[Any]:
    """Load an action trace saved by :func:`save_action_trace`."""
    import numpy as np

    with np.load(Path(path), allow_pickle=True) as data:
        return data["actions"].tolist()


def compute_action_l2_trace(current_trace: Iterable[Any], baseline_trace: Iterable[Any]) -> float:
    """Compute mean L2 distance between two action traces."""
    current = [_flatten_numeric(action) for action in current_trace]
    baseline = [_flatten_numeric(action) for action in baseline_trace]
    pairs = list(zip(current, baseline))
    if not pairs:
        return 0.0

    distances = []
    for current_values, baseline_values in pairs:
        length = min(len(current_values), len(baseline_values))
        if length == 0:
            continue
        distances.append(
            math.sqrt(
                sum((current_values[idx] - baseline_values[idx]) ** 2 for idx in range(length))
            )
        )
    if not distances:
        return 0.0
    return sum(distances) / len(distances)


def _flatten_numeric_or_repr(value: Any) -> Any:
    try:
        return _flatten_numeric(value)
    except TypeError:
        return repr(value)


def _flatten_numeric(value: Any) -> list[float]:
    if isinstance(value, dict):
        flattened: list[float] = []
        for key in sorted(value):
            flattened.extend(_flatten_numeric(value[key]))
        return flattened
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (str, bytes)):
        raise TypeError("Action trace contains non-numeric string data.")
    try:
        iterator = iter(value)
    except TypeError as exc:
        raise TypeError(f"Unsupported action trace value: {type(value).__name__}") from exc

    flattened = []
    for item in iterator:
        flattened.extend(_flatten_numeric(item))
    return flattened

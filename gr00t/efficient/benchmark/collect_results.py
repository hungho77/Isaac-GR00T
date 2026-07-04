# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for collecting and summarizing efficient benchmark result files."""

from __future__ import annotations

import argparse
import csv
import json
from numbers import Number
from pathlib import Path
from typing import Any, Sequence

from gr00t.efficient.benchmark.metrics import RECORD_FIELDS, write_csv, write_json


AGGREGATE_FILENAMES = {
    "aggregated_records.csv",
    "comparison_summary.csv",
    "comparison_summary.json",
    "method_summary.csv",
    "method_summary.json",
}

SUMMARY_FIELDS = [
    "method",
    "num_records",
    "mean_success_rate",
    "mean_latency_per_action_ms",
    "mean_episode_time_s",
    "mean_gpu_memory_mb",
    "mean_peak_gpu_memory_mb",
    "mean_visual_token_count_after",
    "mean_keep_ratio",
    "mean_token_reduction_ratio",
    "mean_action_l2_vs_baseline",
    "speedup_vs_baseline",
]


def load_json_results(results_dir: str | Path) -> list[dict[str, Any]]:
    """Load JSON result files from a directory without enforcing a schema yet."""
    root = Path(results_dir)
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            records.append(json.load(handle))
    return records


def _coerce_csv_value(value: str) -> Any:
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer() and "." not in value:
        return int(number)
    return number


def _is_record(candidate: Any) -> bool:
    return isinstance(candidate, dict) and "method" in candidate and "episode_id" in candidate


def _records_from_json_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if _is_record(item)]
    if isinstance(payload, dict):
        records = payload.get("records")
        if isinstance(records, list):
            return [dict(item) for item in records if _is_record(item)]
        if _is_record(payload):
            return [dict(payload)]
    return []


def _load_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {key: _coerce_csv_value(value) for key, value in row.items()}
            for row in reader
            if row.get("method") and row.get("episode_id") not in {None, ""}
        ]


def load_records(results_dir: str | Path) -> list[dict[str, Any]]:
    """Load benchmark episode records from JSON/CSV files in a directory."""
    root = Path(results_dir)
    records: list[dict[str, Any]] = []
    json_stems_with_records: set[str] = set()

    for path in sorted(root.glob("*.json")):
        if path.name in AGGREGATE_FILENAMES:
            continue
        with path.open("r", encoding="utf-8") as handle:
            loaded = _records_from_json_payload(json.load(handle))
        if loaded:
            json_stems_with_records.add(path.stem)
            records.extend(loaded)

    for path in sorted(root.glob("*.csv")):
        if path.name in AGGREGATE_FILENAMES or path.stem in json_stems_with_records:
            continue
        loaded = _load_csv_records(path)
        records.extend(loaded)

    return records


def _numeric_mean(records: list[dict[str, Any]], field: str) -> float | None:
    values = [
        float(record[field])
        for record in records
        if isinstance(record.get(field), Number) and not isinstance(record.get(field), bool)
    ]
    if not values:
        return None
    return sum(values) / len(values)


def summarize_by_method(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute method-level aggregate metrics and baseline-relative speedup."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        method = str(record.get("method", "unknown"))
        grouped.setdefault(method, []).append(record)

    baseline_latency = _numeric_mean(grouped.get("baseline", []), "latency_per_action_ms")
    summaries: list[dict[str, Any]] = []
    for method in sorted(grouped):
        rows = grouped[method]
        mean_latency = _numeric_mean(rows, "latency_per_action_ms")
        speedup = None
        if baseline_latency and mean_latency:
            speedup = baseline_latency / mean_latency

        summary = {
            "method": method,
            "num_records": len(rows),
            "mean_success_rate": _numeric_mean(rows, "success_rate"),
            "mean_latency_per_action_ms": mean_latency,
            "mean_episode_time_s": _numeric_mean(rows, "episode_time_s"),
            "mean_gpu_memory_mb": _numeric_mean(rows, "gpu_memory_mb"),
            "mean_peak_gpu_memory_mb": _numeric_mean(rows, "peak_gpu_memory_mb"),
            "mean_visual_token_count_after": _numeric_mean(rows, "visual_token_count_after"),
            "mean_keep_ratio": _numeric_mean(rows, "keep_ratio"),
            "mean_token_reduction_ratio": _numeric_mean(rows, "token_reduction_ratio"),
            "mean_action_l2_vs_baseline": _numeric_mean(rows, "action_l2_vs_baseline"),
            "speedup_vs_baseline": speedup,
        }
        summaries.append(summary)
    return summaries


def _write_summary_csv(path: str | Path, summaries: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in SUMMARY_FIELDS})


def collect_results(input_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Aggregate benchmark result files and write record and method summaries."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    records = load_records(input_dir)
    method_summary = summarize_by_method(records)

    aggregated_records_path = output_root / "aggregated_records.csv"
    method_summary_csv = output_root / "method_summary.csv"
    method_summary_json = output_root / "method_summary.json"

    normalized_records = [{field: record.get(field) for field in RECORD_FIELDS} for record in records]
    write_csv(aggregated_records_path, normalized_records)
    _write_summary_csv(method_summary_csv, method_summary)
    write_json(method_summary_json, {"records": method_summary})

    return {
        "status": "ok",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_records": len(records),
        "num_methods": len(method_summary),
        "aggregated_records_csv": str(aggregated_records_path),
        "method_summary_csv": str(method_summary_csv),
        "method_summary_json": str(method_summary_json),
        "methods": [row["method"] for row in method_summary],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect efficient benchmark result files.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = collect_results(input_dir=args.input_dir, output_dir=args.output_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

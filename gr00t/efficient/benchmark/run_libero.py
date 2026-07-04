# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LIBERO baseline profiling scaffold for efficient inference benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gr00t.efficient.benchmark.metrics import (
    BenchmarkRecord,
    summarize_metrics,
    write_csv,
    write_json,
)
from gr00t.efficient.profiler.memory import reset_peak_memory_stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LIBERO efficient inference benchmark scaffold.")
    parser.add_argument("--config", default="gr00t/efficient/configs/libero_baseline.yaml")
    parser.add_argument("--output", default="results/efficient_benchmark/libero_result.json")
    parser.add_argument("--method", default="baseline")
    parser.add_argument("--keep-ratio", type=float, default=1.0)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--task", default="debug")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock", action="store_true")
    return parser


def _validate_day2_args(args: argparse.Namespace) -> None:
    if args.method != "baseline":
        raise SystemExit("Day 2 run_libero supports only --method baseline.")
    if args.keep_ratio != 1.0:
        raise SystemExit("Baseline profiling must use --keep-ratio 1.0.")
    if args.num_episodes < 1:
        raise SystemExit("--num-episodes must be >= 1.")


def _csv_path_for_json(output_path: Path) -> Path | None:
    if output_path.suffix == ".json":
        return output_path.with_suffix(".csv")
    return None


def _dry_run_result(args: argparse.Namespace) -> dict[str, object]:
    return {
        "benchmark": "LIBERO",
        "config": args.config,
        "method": args.method,
        "keep_ratio": args.keep_ratio,
        "num_episodes": args.num_episodes,
        "task": args.task,
        "status": "dry_run_ok",
    }


def run_mock_libero_baseline(args: argparse.Namespace) -> dict[str, object]:
    """Create deterministic baseline records without importing LIBERO."""
    reset_peak_memory_stats()
    records: list[BenchmarkRecord] = []
    for episode_id in range(args.num_episodes):
        records.append(
            BenchmarkRecord(
                benchmark="LIBERO",
                method="baseline",
                task=args.task,
                episode_id=episode_id,
                success=True,
                success_rate=1.0,
                latency_per_action_ms=42.0 + (episode_id * 1.5),
                episode_time_s=12.0 + (episode_id * 0.25),
                gpu_memory_mb=4096.0,
                peak_gpu_memory_mb=5120.0,
                visual_token_count=576,
                keep_ratio=1.0,
                action_l2_vs_baseline=0.0,
                failure_type="",
                notes="mock baseline record; no LIBERO environment was run",
            )
        )

    summary = summarize_metrics(records)
    return {
        "status": "mock_ok",
        "benchmark": "LIBERO",
        "config": args.config,
        "method": args.method,
        "keep_ratio": args.keep_ratio,
        "num_episodes": args.num_episodes,
        "task": args.task,
        "summary": summary,
        "records": records,
    }


def run_real_libero_baseline() -> dict[str, object]:
    """Placeholder for the real LIBERO baseline integration."""
    raise NotImplementedError(
        "Real LIBERO baseline profiling is not wired into gr00t.efficient yet. "
        "The repo entrypoints to connect are gr00t/eval/run_gr00t_server.py "
        "for GR00T policy serving and gr00t/eval/rollout_policy.py for "
        "LIBERO rollouts. Add metric/profiler collection around that path "
        "after gr00t/eval/sim/LIBERO/setup_libero.sh and the LIBERO checkpoint "
        "are available."
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_day2_args(args)
    output_path = Path(args.output)

    if args.dry_run:
        result = _dry_run_result(args)
        write_json(output_path, result)
        print(json.dumps(result, indent=2))
        return 0

    if args.mock:
        result = run_mock_libero_baseline(args)
        write_json(output_path, result)
        records = result["records"]
        csv_path = _csv_path_for_json(output_path)
        if csv_path is not None:
            write_csv(csv_path, records)
            result["csv_output"] = str(csv_path)
            write_json(output_path, result)
        print(json.dumps(result, indent=2, default=lambda value: value.to_dict()))
        return 0

    try:
        result = run_real_libero_baseline()
    except NotImplementedError as exc:
        raise SystemExit(str(exc)) from exc
    write_json(output_path, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

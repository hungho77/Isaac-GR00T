# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LIBERO baseline profiling scaffold for efficient inference benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gr00t.efficient.benchmark.metrics import write_csv, write_json
from gr00t.efficient.benchmark.registry import build_method, list_methods
from gr00t.efficient.benchmark.runner import BenchmarkRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LIBERO efficient inference benchmark scaffold.")
    parser.add_argument("--config", default="gr00t/efficient/configs/libero_baseline.yaml")
    parser.add_argument("--output", default="results/efficient_benchmark/libero_result.json")
    parser.add_argument("--method", default="baseline")
    parser.add_argument("--keep-ratio", type=float, default=1.0)
    parser.add_argument("--dummy-mode", default="first", choices=["first", "uniform", "random"])
    parser.add_argument("--score-mode", default="norm", choices=["norm", "mean_abs", "attention", "action"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--temporal-momentum", type=float, default=0.8)
    parser.add_argument("--visual-token-count", type=int, default=256)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--task", default="debug")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_episodes < 1:
        raise SystemExit("--num-episodes must be >= 1.")
    if args.visual_token_count < 1:
        raise SystemExit("--visual-token-count must be >= 1.")


def _csv_path_for_json(output_path: Path) -> Path | None:
    if output_path.suffix == ".json":
        return output_path.with_suffix(".csv")
    return None


def _dry_run_result(args: argparse.Namespace, method_metadata: dict[str, object]) -> dict[str, object]:
    return {
        "benchmark": "LIBERO",
        "config": args.config,
        "method": args.method,
        "keep_ratio": args.keep_ratio,
        "dummy_mode": args.dummy_mode,
        "score_mode": args.score_mode,
        "alpha": args.alpha,
        "beta": args.beta,
        "temporal_momentum": args.temporal_momentum,
        "visual_token_count": args.visual_token_count,
        "num_episodes": args.num_episodes,
        "task": args.task,
        "status": "dry_run_ok",
        "available_methods": list_methods(),
        "method_metadata": method_metadata,
    }


def run_mock_libero_baseline(args: argparse.Namespace, runner: BenchmarkRunner) -> dict[str, object]:
    """Create deterministic baseline records without importing LIBERO."""
    records = runner.run_mock(
        num_episodes=args.num_episodes,
        task=args.task,
        visual_token_count=args.visual_token_count,
    )
    summary = runner.summarize(records)
    return {
        "status": "mock_ok",
        "benchmark": "LIBERO",
        "config": args.config,
        "method": args.method,
        "keep_ratio": args.keep_ratio,
        "dummy_mode": args.dummy_mode,
        "score_mode": args.score_mode,
        "alpha": args.alpha,
        "beta": args.beta,
        "temporal_momentum": args.temporal_momentum,
        "visual_token_count": args.visual_token_count,
        "num_episodes": args.num_episodes,
        "task": args.task,
        "method_metadata": runner.method.metadata(),
        "summary": summary,
        "records": records,
    }


def run_real_libero_baseline(method: object, args: argparse.Namespace) -> dict[str, object]:
    """Placeholder for the real LIBERO baseline integration."""
    raise NotImplementedError(
        "Real LIBERO baseline profiling is not wired into gr00t.efficient yet. "
        "The repo entrypoints to connect are gr00t/eval/run_gr00t_server.py "
        "for GR00T policy serving and gr00t/eval/rollout_policy.py for "
        "LIBERO rollouts. Add metric/profiler collection around that path "
        "after gr00t/eval/sim/LIBERO/setup_libero.sh and the LIBERO checkpoint "
        f"are available. Requested method={getattr(method, 'method_name', args.method)!r}."
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    try:
        method = build_method(
            args.method,
            keep_ratio=args.keep_ratio,
            mode=args.dummy_mode,
            score_mode=args.score_mode,
            alpha=args.alpha,
            beta=args.beta,
            temporal_momentum=args.temporal_momentum,
        )
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    runner = BenchmarkRunner(benchmark_name="LIBERO", method=method)
    output_path = Path(args.output)

    if args.dry_run:
        result = _dry_run_result(args, method.metadata())
        write_json(output_path, result)
        print(json.dumps(result, indent=2))
        return 0

    if args.mock:
        result = run_mock_libero_baseline(args, runner)
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
        result = run_real_libero_baseline(method=method, args=args)
    except NotImplementedError as exc:
        raise SystemExit(str(exc)) from exc
    write_json(output_path, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a unified mock comparison across efficient inference methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from gr00t.efficient.benchmark.metrics import (
    BenchmarkRecord,
    summarize_metrics,
    write_csv,
    write_json,
)
from gr00t.efficient.benchmark.registry import build_method
from gr00t.efficient.benchmark.runner import BenchmarkRunner


DEFAULT_METHODS = "baseline,dummy,vlapruner,specprune,adp,adp_vlapruner"
DEFAULT_KEEP_RATIOS = "1.0,0.75,0.6,0.5"
DYNAMIC_METHODS = {"adp", "adp_vlapruner"}
FIXED_PRUNING_METHODS = {"dummy", "vlapruner", "specprune"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a full efficient inference comparison.")
    parser.add_argument("--benchmark", default="LIBERO")
    parser.add_argument("--output-dir", default="results/efficient_benchmark/day7_comparison")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--visual-token-count", type=int, default=256)
    parser.add_argument("--task", default="debug")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--keep-ratios", default=DEFAULT_KEEP_RATIOS)
    parser.add_argument("--dummy-mode", default="first", choices=["first", "uniform", "random"])
    parser.add_argument("--score-mode", default="norm", choices=["norm", "mean_abs", "attention", "action"])
    parser.add_argument("--reuse-steps", type=int, default=2)
    parser.add_argument("--default-keep-ratio", type=float, default=0.5)
    parser.add_argument("--contact-keep-ratio", type=float, default=1.0)
    parser.add_argument("--move-keep-ratio", type=float, default=0.6)
    parser.add_argument("--idle-keep-ratio", type=float, default=0.5)
    parser.add_argument("--action-delta-threshold", type=float, default=0.05)
    return parser


def _parse_csv_names(value: str) -> list[str]:
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("Expected at least one method.")
    return names


def _parse_keep_ratios(value: str) -> list[float]:
    ratios = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not ratios:
        raise ValueError("Expected at least one keep ratio.")
    for ratio in ratios:
        if ratio <= 0:
            raise ValueError("Keep ratios must be > 0.")
    return ratios


def _ratio_suffix(keep_ratio: float) -> str:
    return f"{keep_ratio:.3g}".replace(".", "p")


def _planned_runs(methods: list[str], keep_ratios: list[float], explicit_ratios: bool) -> list[tuple[str, float]]:
    runs: list[tuple[str, float]] = []
    for method in methods:
        if method == "baseline":
            runs.append((method, 1.0))
        elif method in DYNAMIC_METHODS:
            runs.append((method, 1.0))
        elif method in FIXED_PRUNING_METHODS:
            for keep_ratio in keep_ratios:
                if keep_ratio >= 1.0 and not explicit_ratios:
                    continue
                runs.append((method, keep_ratio))
        else:
            raise ValueError(f"Unknown comparison method {method!r}.")
    return runs


def _build_method_kwargs(args: argparse.Namespace, method_name: str, keep_ratio: float) -> dict[str, Any]:
    kwargs = {
        "keep_ratio": keep_ratio,
        "mode": args.dummy_mode,
        "score_mode": args.score_mode,
        "reuse_steps": args.reuse_steps,
        "default_keep_ratio": args.default_keep_ratio,
        "contact_keep_ratio": args.contact_keep_ratio,
        "move_keep_ratio": args.move_keep_ratio,
        "idle_keep_ratio": args.idle_keep_ratio,
        "action_delta_threshold": args.action_delta_threshold,
    }
    if method_name in DYNAMIC_METHODS:
        kwargs["keep_ratio"] = args.default_keep_ratio
    return kwargs


def _run_one_mock(
    args: argparse.Namespace,
    method_name: str,
    keep_ratio: float,
    output_dir: Path,
) -> dict[str, Any]:
    method = build_method(method_name, **_build_method_kwargs(args, method_name, keep_ratio))
    runner = BenchmarkRunner(benchmark_name=args.benchmark, method=method)
    records = runner.run_mock(
        num_episodes=args.num_episodes,
        task=args.task,
        visual_token_count=args.visual_token_count,
    )
    summary = runner.summarize(records)
    if "dynamic_keep_ratio" in summary:
        summary["avg_keep_ratio"] = summary["dynamic_keep_ratio"]

    if method_name in DYNAMIC_METHODS:
        run_id = f"{args.benchmark.lower()}_{method_name}_dynamic_mock"
    else:
        run_id = f"{args.benchmark.lower()}_{method_name}_kr{_ratio_suffix(keep_ratio)}_mock"

    json_path = output_dir / f"{run_id}.json"
    csv_path = output_dir / f"{run_id}.csv"
    result = {
        "status": "mock_ok",
        "benchmark": args.benchmark,
        "method": method_name,
        "keep_ratio": keep_ratio,
        "visual_token_count": args.visual_token_count,
        "num_episodes": args.num_episodes,
        "task": args.task,
        "method_metadata": method.metadata(),
        "summary": summary,
        "records": records,
        "csv_output": str(csv_path),
    }
    write_json(json_path, result)
    write_csv(csv_path, records)
    return {
        "run_id": run_id,
        "json_output": str(json_path),
        "csv_output": str(csv_path),
        "method": method_name,
        "keep_ratio": keep_ratio,
        "summary": summary,
        "records": records,
    }


def _record_dicts(records: list[BenchmarkRecord]) -> list[dict[str, Any]]:
    return [record.to_dict() if isinstance(record, BenchmarkRecord) else dict(record) for record in records]


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    if not args.mock:
        raise NotImplementedError(
            "run_comparison currently supports --mock only. Real LIBERO/LIBERO-Plus execution "
            "must connect run_libero.py or run_libero_plus.py to the evaluation entrypoint first."
        )
    if args.num_episodes < 1:
        raise ValueError("--num-episodes must be >= 1.")
    if args.visual_token_count < 1:
        raise ValueError("--visual-token-count must be >= 1.")

    methods = _parse_csv_names(args.methods)
    keep_ratios = _parse_keep_ratios(args.keep_ratios)
    explicit_ratios = args.keep_ratios != DEFAULT_KEEP_RATIOS
    runs = _planned_runs(methods, keep_ratios, explicit_ratios=explicit_ratios)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_results: list[dict[str, Any]] = []
    combined_records: list[BenchmarkRecord] = []
    for method_name, keep_ratio in runs:
        run_result = _run_one_mock(args, method_name, keep_ratio, output_dir)
        run_results.append({key: value for key, value in run_result.items() if key != "records"})
        combined_records.extend(run_result["records"])

    combined_summary = summarize_metrics(combined_records)
    if "dynamic_keep_ratio" in combined_summary:
        combined_summary["avg_keep_ratio"] = combined_summary["dynamic_keep_ratio"]

    combined_csv = output_dir / "comparison_summary.csv"
    combined_json = output_dir / "comparison_summary.json"
    write_csv(combined_csv, combined_records)
    result = {
        "status": "mock_ok",
        "benchmark": args.benchmark,
        "task": args.task,
        "num_episodes": args.num_episodes,
        "visual_token_count": args.visual_token_count,
        "methods": methods,
        "keep_ratios": keep_ratios,
        "num_runs": len(run_results),
        "num_records": len(combined_records),
        "summary": combined_summary,
        "runs": run_results,
        "records": _record_dicts(combined_records),
        "csv_output": str(combined_csv),
    }
    write_json(combined_json, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_comparison(args)
    except (KeyError, ValueError, NotImplementedError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

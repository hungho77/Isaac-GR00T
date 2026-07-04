# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LIBERO baseline profiling scaffold for efficient inference benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gr00t.efficient.benchmark.metrics import write_csv, write_json
from gr00t.efficient.benchmark.real_libero_adapter import RealLiberoAdapter
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
    parser.add_argument("--reuse-steps", type=int, default=2)
    parser.add_argument("--default-keep-ratio", type=float, default=0.5)
    parser.add_argument("--contact-keep-ratio", type=float, default=1.0)
    parser.add_argument("--move-keep-ratio", type=float, default=0.6)
    parser.add_argument("--idle-keep-ratio", type=float, default=0.5)
    parser.add_argument("--action-delta-threshold", type=float, default=0.05)
    parser.add_argument("--visual-token-count", type=int, default=256)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--task", default="debug")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--max-steps", type=int, default=720)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-actions", action="store_true")
    parser.add_argument("--save-traces", action="store_true")
    parser.add_argument("--real-output-dir", default="results/efficient_benchmark/real_libero")
    parser.add_argument("--real-command", default="")
    parser.add_argument("--eval-config", default="")
    parser.add_argument("--policy-client-host", default="127.0.0.1")
    parser.add_argument("--policy-client-port", type=int, default=5555)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--libero-python", default="")
    parser.add_argument("--hook-visual-tokens", action="store_true")
    parser.add_argument("--disable-visual-token-hook", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_episodes < 1:
        raise SystemExit("--num-episodes must be >= 1.")
    if args.visual_token_count < 1:
        raise SystemExit("--visual-token-count must be >= 1.")
    if args.reuse_steps < 1:
        raise SystemExit("--reuse-steps must be >= 1.")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be >= 1.")
    if args.n_envs < 1:
        raise SystemExit("--n-envs must be >= 1.")
    if args.n_action_steps < 1:
        raise SystemExit("--n-action-steps must be >= 1.")


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
        "reuse_steps": args.reuse_steps,
        "default_keep_ratio": args.default_keep_ratio,
        "contact_keep_ratio": args.contact_keep_ratio,
        "move_keep_ratio": args.move_keep_ratio,
        "idle_keep_ratio": args.idle_keep_ratio,
        "action_delta_threshold": args.action_delta_threshold,
        "visual_token_count": args.visual_token_count,
        "num_episodes": args.num_episodes,
        "task": args.task,
        "suite": args.suite,
        "checkpoint_path": args.checkpoint_path,
        "model_path": args.model_path,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "device": args.device,
        "real_output_dir": args.real_output_dir,
        "hook_visual_tokens": args.hook_visual_tokens,
        "disable_visual_token_hook": args.disable_visual_token_hook,
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
    if "dynamic_keep_ratio" in summary:
        summary["avg_keep_ratio"] = summary["dynamic_keep_ratio"]
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
        "reuse_steps": args.reuse_steps,
        "default_keep_ratio": args.default_keep_ratio,
        "contact_keep_ratio": args.contact_keep_ratio,
        "move_keep_ratio": args.move_keep_ratio,
        "idle_keep_ratio": args.idle_keep_ratio,
        "action_delta_threshold": args.action_delta_threshold,
        "visual_token_count": args.visual_token_count,
        "num_episodes": args.num_episodes,
        "task": args.task,
        "method_metadata": runner.method.metadata(),
        "summary": summary,
        "records": records,
    }


def run_real_libero_baseline(method: object, args: argparse.Namespace) -> dict[str, object]:
    """Run real LIBERO through the efficient benchmark adapter."""
    adapter = RealLiberoAdapter(args=args, method=method)
    records = adapter.run()
    runner = BenchmarkRunner(benchmark_name="LIBERO", method=method)
    summary = runner.summarize(records)
    return {
        "status": "real_ok",
        "benchmark": "LIBERO",
        "config": args.config,
        "method": getattr(method, "method_name", args.method),
        "keep_ratio": getattr(method, "keep_ratio", args.keep_ratio),
        "suite": args.suite,
        "task": args.task,
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "device": args.device,
        "method_metadata": method.metadata() if hasattr(method, "metadata") else {},
        "summary": summary,
        "records": records,
    }


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
            reuse_steps=args.reuse_steps,
            default_keep_ratio=args.default_keep_ratio,
            contact_keep_ratio=args.contact_keep_ratio,
            move_keep_ratio=args.move_keep_ratio,
            idle_keep_ratio=args.idle_keep_ratio,
            action_delta_threshold=args.action_delta_threshold,
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
    except (RuntimeError, ImportError, FileNotFoundError, NotImplementedError) as exc:
        raise SystemExit(str(exc)) from exc
    write_json(output_path, result)
    records = result.get("records", [])
    csv_path = _csv_path_for_json(output_path)
    if csv_path is not None:
        write_csv(csv_path, records)
        result["csv_output"] = str(csv_path)
        write_json(output_path, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

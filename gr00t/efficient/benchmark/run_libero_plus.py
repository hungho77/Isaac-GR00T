# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Day 1 CLI scaffold for LIBERO-Plus efficient inference benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LIBERO-Plus efficient inference benchmark scaffold."
    )
    parser.add_argument("--config", default="gr00t/efficient/configs/libero_plus_baseline.yaml")
    parser.add_argument("--output", default="results/efficient_benchmark/libero_plus_result.json")
    parser.add_argument("--method", default="baseline")
    parser.add_argument("--keep-ratio", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def write_result(path: str | Path, result: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.dry_run:
        result = {
            "benchmark": "LIBERO-Plus",
            "config": args.config,
            "method": args.method,
            "keep_ratio": args.keep_ratio,
            "status": "dry_run_ok",
        }
        write_result(args.output, result)
        print(json.dumps(result, indent=2))
        return 0

    raise SystemExit("Only --dry-run is implemented in the Day 1 scaffold.")


if __name__ == "__main__":
    raise SystemExit(main())


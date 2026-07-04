# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate visual-token pruner behavior with real torch tensors when available."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gr00t.efficient.benchmark.metrics import write_json
from gr00t.efficient.pruners.dummy import DummyVisualTokenPruner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate visual-token pruner tensor behavior.")
    parser.add_argument("--output", default="results/efficient_benchmark/day4_pruner_validation.json")
    return parser


def _shape(value: object) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def run_validation() -> dict[str, object]:
    try:
        import torch
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": f"torch unavailable: {exc}",
            "cases": [],
        }

    visual_tokens = torch.randn(2, 100, 64)
    cases = [
        {"keep_ratio": 1.0, "mode": "first"},
        {"keep_ratio": 0.75, "mode": "first"},
        {"keep_ratio": 0.5, "mode": "uniform"},
        {"keep_ratio": 0.25, "mode": "random"},
    ]

    results: list[dict[str, object]] = []
    for case in cases:
        pruner = DummyVisualTokenPruner(
            keep_ratio=float(case["keep_ratio"]),
            mode=str(case["mode"]),
            seed=0,
        )
        pruned_tokens, metadata = pruner.prune(visual_tokens)
        expected_tokens = metadata["kept_tokens"]
        status = "ok" if _shape(pruned_tokens) == [2, expected_tokens, 64] else "failed"
        results.append(
            {
                "case": f"keep={case['keep_ratio']} mode={case['mode']}",
                "input_shape": _shape(visual_tokens),
                "output_shape": _shape(pruned_tokens),
                "keep_ratio": case["keep_ratio"],
                "mode": case["mode"],
                "status": status,
                "metadata": metadata,
            }
        )

    overall_status = "ok" if all(case["status"] == "ok" for case in results) else "failed"
    return {
        "status": overall_status,
        "cases": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_validation()
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"ok", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

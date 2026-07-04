# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate VLA-Pruner MVP tensor behavior when torch is available."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from gr00t.efficient.benchmark.metrics import write_json
from gr00t.efficient.pruners.vlapruner import VLAPruner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate VLA-Pruner MVP behavior.")
    parser.add_argument("--output", default="results/efficient_benchmark/day5_vlapruner_validation.json")
    return parser


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def run_validation() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "status": "skipped_no_torch",
            "reason": f"torch unavailable: {exc}",
            "cases": [],
        }

    generator = torch.Generator()
    generator.manual_seed(17)
    visual_tokens = torch.randn((2, 256, 128), generator=generator)

    cases: list[dict[str, Any]] = []
    for keep_ratio, score_mode, temporal in [
        (1.0, "norm", False),
        (0.75, "norm", False),
        (0.5, "mean_abs", False),
    ]:
        pruner = VLAPruner(keep_ratio=keep_ratio, score_mode=score_mode)
        pruned_tokens, metadata = pruner.prune(visual_tokens, timestep=0)
        cases.append(
            _case_result(
                name=f"{score_mode} keep={keep_ratio}",
                input_shape=_shape(visual_tokens),
                output_shape=_shape(pruned_tokens),
                keep_ratio=keep_ratio,
                score_mode=score_mode,
                temporal=temporal,
                metadata=metadata,
            )
        )

    temporal_pruner = VLAPruner(keep_ratio=0.5, score_mode="norm", temporal_momentum=0.8)
    first_tokens, first_metadata = temporal_pruner.prune(visual_tokens, timestep=0)
    second_tokens, second_metadata = temporal_pruner.prune(visual_tokens + 0.01, timestep=1)
    temporal_status = (
        "ok"
        if _shape(first_tokens) == [2, 128, 128]
        and _shape(second_tokens) == [2, 128, 128]
        and second_metadata.get("used_temporal_smoothing") is True
        else "failed"
    )
    cases.append(
        {
            "case": "norm keep=0.5 temporal",
            "input_shape": _shape(visual_tokens),
            "output_shape": _shape(second_tokens),
            "warmup_output_shape": _shape(first_tokens),
            "keep_ratio": 0.5,
            "score_mode": "norm",
            "temporal": True,
            "status": temporal_status,
            "warmup_metadata": first_metadata,
            "metadata": second_metadata,
        }
    )

    status = "ok" if all(case["status"] == "ok" for case in cases) else "failed"
    return {
        "status": status,
        "cases": cases,
    }


def _case_result(
    name: str,
    input_shape: list[int] | None,
    output_shape: list[int] | None,
    keep_ratio: float,
    score_mode: str,
    temporal: bool,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    expected_tokens = metadata.get("kept_tokens")
    expected_shape = None if input_shape is None else [input_shape[0], expected_tokens, input_shape[2]]
    return {
        "case": name,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "keep_ratio": keep_ratio,
        "score_mode": score_mode,
        "temporal": temporal,
        "status": "ok" if output_shape == expected_shape else "failed",
        "metadata": metadata,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_validation()
    if result["status"] == "skipped_no_torch":
        print("torch unavailable; skipping VLA-Pruner validation")
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"ok", "skipped_no_torch"} else 1


if __name__ == "__main__":
    raise SystemExit(main())


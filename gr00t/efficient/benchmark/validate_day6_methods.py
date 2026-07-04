# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate SpecPrune and ADP MVP behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from gr00t.efficient.benchmark.methods import ADPVLAPrunerMethod
from gr00t.efficient.benchmark.metrics import write_json
from gr00t.efficient.pruners.specprune import SpecPruneVLA
from gr00t.efficient.schedulers.adp import ADPScheduler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Day 6 efficient inference methods.")
    parser.add_argument("--output", default="results/efficient_benchmark/day6_methods_validation.json")
    return parser


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def run_validation() -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "ok",
        "specprune": {"status": "skipped_no_torch"},
        "adp": _validate_adp(),
        "adp_vlapruner": {"status": "skipped_no_torch"},
    }

    try:
        import torch
    except Exception as exc:
        result["tensor_pruning_status"] = "skipped_no_torch"
        result["tensor_pruning_reason"] = str(exc)
        return result

    generator = torch.Generator()
    generator.manual_seed(23)
    visual_tokens = torch.randn((2, 256, 128), generator=generator)
    result["specprune"] = _validate_specprune(visual_tokens)
    result["adp_vlapruner"] = _validate_adp_vlapruner()
    result["status"] = (
        "ok"
        if result["specprune"]["status"] == "ok"
        and result["adp"]["status"] == "ok"
        and result["adp_vlapruner"]["status"] == "ok"
        else "failed"
    )
    return result


def _validate_specprune(visual_tokens: Any) -> dict[str, Any]:
    pruner = SpecPruneVLA(keep_ratio=0.75, reuse_steps=2, score_mode="norm")
    first_tokens, first_metadata = pruner.prune(visual_tokens, timestep=0)
    second_tokens, second_metadata = pruner.prune(visual_tokens + 0.01, timestep=1)
    status = (
        "ok"
        if _shape(first_tokens) == [2, 192, 128]
        and _shape(second_tokens) == [2, 192, 128]
        and first_metadata.get("reused_indices") is False
        and second_metadata.get("reused_indices") is True
        else "failed"
    )
    return {
        "status": status,
        "first_output_shape": _shape(first_tokens),
        "second_output_shape": _shape(second_tokens),
        "first_metadata": first_metadata,
        "second_metadata": second_metadata,
    }


def _validate_adp() -> dict[str, Any]:
    scheduler = ADPScheduler()
    default_ratio, default_metadata = scheduler.get_keep_ratio(timestep=0)
    move_ratio, move_metadata = scheduler.get_keep_ratio(
        action_state=[0.2, 0.0, 0.0, 0.0],
        prev_action_state=[0.0, 0.0, 0.0, 0.0],
        timestep=1,
    )
    contact_ratio, contact_metadata = scheduler.get_keep_ratio(
        action_state=[0.0, 0.0, 0.0, 1.0],
        prev_action_state=[0.0, 0.0, 0.0, 0.0],
        timestep=2,
    )
    status = (
        "ok"
        if default_ratio == 0.5
        and default_metadata["reason"] == "no_action_state"
        and move_ratio == 0.6
        and move_metadata["reason"] == "moving"
        and contact_ratio == 1.0
        and contact_metadata["reason"] == "contact_or_gripper"
        else "failed"
    )
    return {
        "status": status,
        "default": default_metadata,
        "moving": move_metadata,
        "contact": contact_metadata,
    }


def _validate_adp_vlapruner() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"status": "skipped_no_torch", "reason": str(exc)}

    generator = torch.Generator()
    generator.manual_seed(29)
    visual_tokens = torch.randn((1, 256, 64), generator=generator)
    method = ADPVLAPrunerMethod(score_mode="norm")

    default_tokens, default_metadata = method.process_visual_tokens(
        visual_tokens,
        action_state=None,
        prev_action_state=None,
        timestep=0,
    )
    move_tokens, move_metadata = method.process_visual_tokens(
        visual_tokens,
        action_state=[0.2, 0.0, 0.0, 0.0],
        prev_action_state=[0.0, 0.0, 0.0, 0.0],
        timestep=1,
    )
    contact_tokens, contact_metadata = method.process_visual_tokens(
        visual_tokens,
        action_state=[0.0, 0.0, 0.0, 1.0],
        prev_action_state=[0.0, 0.0, 0.0, 0.0],
        timestep=2,
    )
    status = (
        "ok"
        if _shape(default_tokens) == [1, 128, 64]
        and _shape(move_tokens) == [1, 154, 64]
        and _shape(contact_tokens) == [1, 256, 64]
        else "failed"
    )
    return {
        "status": status,
        "default_output_shape": _shape(default_tokens),
        "move_output_shape": _shape(move_tokens),
        "contact_output_shape": _shape(contact_tokens),
        "default_metadata": default_metadata,
        "move_metadata": move_metadata,
        "contact_metadata": contact_metadata,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_validation()
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())


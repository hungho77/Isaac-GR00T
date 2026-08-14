# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Offline fail-closed verification for a native CKA N1.7 checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import gr00t.model  # noqa: F401
from gr00t.model.cka_pruning import (
    manifest_depths,
    validate_checkpoint_loading_info,
    verify_model_matches_manifest,
)
from transformers import AutoModel, AutoProcessor
import tyro


@dataclass
class Args:
    model_path: str
    expected_action_dit: int | None = None
    expected_backbone_language: int | None = None
    expected_vl_self_attention: int | None = None
    expected_action_horizon: int | None = 16


def main(args: Args) -> None:
    model_path = Path(args.model_path).expanduser().resolve()
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = raw_config.get("cka_pruning_manifest")
    if manifest is None:
        raise ValueError("Checkpoint does not embed cka_pruning_manifest")

    model, loading_info = AutoModel.from_pretrained(model_path, output_loading_info=True)
    loading_report = validate_checkpoint_loading_info(loading_info)
    model_report = verify_model_matches_manifest(model, manifest)
    depths = manifest_depths(manifest)
    expected = {
        "action_dit": args.expected_action_dit,
        "backbone_language": args.expected_backbone_language,
        "vl_self_attention": args.expected_vl_self_attention,
    }
    for name, expected_depth in expected.items():
        if expected_depth is not None and depths.get(name) != expected_depth:
            raise ValueError(f"{name} retained depth={depths.get(name)}, expected={expected_depth}")

    processor_dir = (
        model_path / "processor"
        if (model_path / "processor").is_dir()
        and not (model_path / "processor_config.json").exists()
        else model_path
    )
    processor = AutoProcessor.from_pretrained(processor_dir)
    modality_configs = processor.get_modality_configs()
    horizons = sorted(
        {
            len(modalities["action"].delta_indices)
            for modalities in modality_configs.values()
            if "action" in modalities
        }
    )
    if args.expected_action_horizon is not None and args.expected_action_horizon not in horizons:
        raise ValueError(
            f"Processor action horizons={horizons}, expected={args.expected_action_horizon}"
        )

    print(
        json.dumps(
            {
                "status": "pass",
                "model_path": str(model_path),
                "retained_depths": depths,
                "model": model_report,
                "checkpoint_loading": loading_report,
                "processor_action_horizons": horizons,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main(tyro.cli(Args))

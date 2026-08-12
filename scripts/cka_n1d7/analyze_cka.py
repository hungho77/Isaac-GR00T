#!/usr/bin/env python
"""Compute N1.7 CKA matrices and emit a validated pruning manifest."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from gr00t.model.cka_pruning import (
    SCHEMA_VERSION,
    consecutive_cka,
    linear_cka,
    select_keep_indices,
    validate_pruning_manifest,
)
from matplotlib import pyplot as plt
import numpy as np
import tyro


@dataclass
class Args:
    calibration_dir: str = "outputs/cka_n1d7/calibration"
    output_dir: str = "outputs/cka_n1d7/analysis"
    backbone_language_prune_ratio: float = 0.40
    action_dit_prune_ratio: float = 0.50
    vl_self_attention_prune_ratio: float = 0.25
    minimum_calibration_samples: int = 8


def _load_layers(archive, module_name: str, *, expected_samples: int) -> list[np.ndarray]:
    prefix = f"{module_name}__layer_"
    keys = sorted(key for key in archive.files if key.startswith(prefix))
    if not keys:
        raise ValueError(f"No activations found for {module_name!r}")
    layers = [archive[key] for key in keys]
    invalid = {
        key: list(value.shape)
        for key, value in zip(keys, layers, strict=True)
        if value.ndim != 2 or value.shape[0] != expected_samples or not np.isfinite(value).all()
    }
    if invalid:
        raise ValueError(
            f"Invalid {module_name} activation arrays; expected finite "
            f"[{expected_samples}, features] tensors, got {invalid}"
        )
    return layers


def _plot_matrix(matrix: np.ndarray, module_name: str, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    axis.set_title(f"GR00T N1.7 CKA: {module_name}")
    axis.set_xlabel("Layer")
    axis.set_ylabel("Layer")
    figure.colorbar(image, ax=axis, label="linear CKA")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main(args: Args) -> None:
    if args.minimum_calibration_samples < 3:
        raise ValueError(
            "minimum_calibration_samples must be at least 3; with two samples, "
            "centered linear CKA is degenerate and typically equals 1 for every layer pair"
        )
    ratios = {
        "backbone_language": args.backbone_language_prune_ratio,
        "action_dit": args.action_dit_prune_ratio,
        "vl_self_attention": args.vl_self_attention_prune_ratio,
    }
    if any(not 0.0 <= ratio < 1.0 for ratio in ratios.values()):
        raise ValueError(f"Prune ratios must be in [0, 1), got {ratios}")

    calibration_dir = Path(args.calibration_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((calibration_dir / "metadata.json").read_text(encoding="utf-8"))
    archive = np.load(calibration_dir / "activations.npz")
    if metadata.get("activation_schema_version") != 2:
        raise ValueError(
            "CKA analysis requires activation schema v2 (one pooled row per observation). "
            "Recapture activations with the current capture_activations.py."
        )
    sampled_steps = metadata.get("sampled_steps")
    if not isinstance(sampled_steps, list) or len(sampled_steps) < args.minimum_calibration_samples:
        observed = len(sampled_steps) if isinstance(sampled_steps, list) else 0
        raise ValueError(
            "CKA calibration is too small for a meaningful layer ranking: "
            f"observed {observed}, required {args.minimum_calibration_samples}. "
            "Use task-diverse observations; 32 or more is recommended for a report."
        )
    expected_samples = len(sampled_steps)

    modules = {}
    score_report = {"args": asdict(args), "calibration_metadata": metadata, "modules": {}}
    for module_name, prune_ratio in ratios.items():
        layers = _load_layers(archive, module_name, expected_samples=expected_samples)
        depth = len(layers)
        target_keep = max(1, int(round(depth * (1.0 - prune_ratio))))
        if module_name == "action_dit":
            target_keep = max(3, target_keep)
        adjacent_scores = consecutive_cka(layers)
        keep_indices = select_keep_indices(adjacent_scores, target_keep, module_name=module_name)
        matrix = np.eye(depth, dtype=np.float64)
        for row in range(depth):
            for column in range(row + 1, depth):
                matrix[row, column] = matrix[column, row] = linear_cka(layers[row], layers[column])
        _plot_matrix(matrix, module_name, output_dir / f"cka_{module_name}.png")
        modules[module_name] = {
            "original_depth": depth,
            "keep_indices": keep_indices,
        }
        score_report["modules"][module_name] = {
            "original_depth": depth,
            "target_keep": target_keep,
            "prune_ratio_requested": prune_ratio,
            "prune_ratio_actual": 1.0 - len(keep_indices) / depth,
            "adjacent_cka": adjacent_scores,
            "keep_indices": keep_indices,
            "remove_indices": sorted(set(range(depth)) - set(keep_indices)),
            "cka_matrix": matrix.tolist(),
        }

    manifest = validate_pruning_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "model_type": "Gr00tN1d7",
            "method": "adjacent_linear_cka_topk",
            "source_model": metadata["args"]["model_path"],
            "calibration_dataset": metadata["args"]["dataset_path"],
            "calibration_dataset_fingerprint": metadata.get("dataset_fingerprint"),
            "modules": modules,
        }
    )
    (output_dir / "pruning_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "cka_report.json").write_text(
        json.dumps(score_report, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(f"Saved analysis to {output_dir.resolve()}")


if __name__ == "__main__":
    main(tyro.cli(Args))

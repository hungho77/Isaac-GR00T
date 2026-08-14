# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed GR00T server for native CKA-pruned N1.7 checkpoints.

Unlike the generic entry point, this server refuses to start unless the
checkpoint embeds ``cka_pruning_manifest``, reconstructs exactly the retained
architecture, and loads without unexplained missing or unexpected keys.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.cka_pruning import manifest_depths, verify_model_matches_manifest
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyServer
import tyro


@dataclass
class NativeCkaServerConfig:
    model_path: str
    """Local merged/root checkpoint containing config.json and model weights."""

    embodiment_tag: str = "new_embodiment"
    device: str = "cuda"
    host: str = "0.0.0.0"
    port: int = 8791
    strict: bool = True

    modality_config_path: str | None = "examples/UR10eCup/ur10e_cup_config.py"
    """Optional Python config imported for a deployment-side consistency check."""

    expected_action_dit: int | None = None
    expected_backbone_language: int | None = None
    expected_vl_self_attention: int | None = None
    expected_action_horizon: int = 16


def _import_modality_config(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Modality config does not exist: {path}")
    module_name = f"gr00t_native_cka_modality_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import modality config: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def _expected_depths(config: NativeCkaServerConfig) -> dict[str, int]:
    mapping = {
        "action_dit": config.expected_action_dit,
        "backbone_language": config.expected_backbone_language,
        "vl_self_attention": config.expected_vl_self_attention,
    }
    return {name: value for name, value in mapping.items() if value is not None}


def main(config: NativeCkaServerConfig) -> None:
    model_path = Path(config.model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {model_path}")
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint config.json does not exist: {config_path}")
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = raw_config.get("cka_pruning_manifest")
    if manifest is None:
        raise ValueError(
            "Native CKA server requires config.json['cka_pruning_manifest']; "
            "refusing to load an unpruned or legacy-schema checkpoint"
        )

    if config.modality_config_path is not None:
        _import_modality_config(Path(config.modality_config_path))

    embodiment_tag = EmbodimentTag.resolve(config.embodiment_tag)
    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=str(model_path),
        device=config.device,
        strict=config.strict,
    )
    if policy.cka_loading_report is None:
        raise RuntimeError("CKA strict loading report was not produced")

    model_report = verify_model_matches_manifest(policy.model, manifest)
    actual_depths = manifest_depths(manifest)
    for name, expected in _expected_depths(config).items():
        actual = actual_depths.get(name)
        if actual != expected:
            raise ValueError(
                f"Wrong checkpoint variant: {name} depth={actual}, expected={expected}"
            )

    action_config = policy.modality_configs.get("action")
    if action_config is None:
        raise ValueError("Checkpoint processor has no action modality config")
    action_horizon = len(action_config.delta_indices)
    if action_horizon != config.expected_action_horizon:
        raise ValueError(
            f"Action horizon mismatch: checkpoint={action_horizon}, "
            f"expected={config.expected_action_horizon}"
        )

    deployment_report = {
        "status": "native_cka_preflight_pass",
        "model_path": str(model_path),
        "embodiment_tag": embodiment_tag.value,
        "retained_depths": actual_depths,
        "action_horizon": action_horizon,
        "action_keys": list(action_config.modality_keys),
        "parameters": model_report["parameters"],
        "strict_loading": policy.cka_loading_report,
        "host": config.host,
        "port": config.port,
    }
    print(json.dumps(deployment_report, indent=2))
    print("Starting native CKA GR00T inference server...")

    with PolicyServer(policy=policy, host=config.host, port=config.port) as server:
        try:
            server.run()
        except KeyboardInterrupt:
            print("\nShutting down native CKA server...")


if __name__ == "__main__":
    main(tyro.cli(NativeCkaServerConfig))

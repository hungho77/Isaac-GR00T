# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import sys

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.replay_policy import ReplayPolicy
from gr00t.policy.server_client import PolicyServer
import tyro


DEFAULT_MODEL_SERVER_PORT = 5555


def _load_json_modality_configs(config_path: Path) -> dict[str, ModalityConfig]:
    """Load a JSON file whose values are ModalityConfig field dicts.

    A dataset's ``meta/modality.json`` is a different (data-layout) schema and is
    not accepted here — point such users at a .py config instead of letting the
    ``ModalityConfig(**v)`` unpack raise a bare ``TypeError``.
    """
    with open(config_path, "r") as f:
        raw = json.load(f)
    try:
        return {k: ModalityConfig(**v) for k, v in raw.items()}
    except TypeError as exc:
        raise ValueError(
            f"{config_path} is not a ModalityConfig JSON: each value must hold ModalityConfig "
            f"fields (delta_indices, modality_keys, ...). A dataset's meta/modality.json uses a "
            f"different schema; pass a .py modality config (e.g. examples/SO100/so100_config.py) instead."
        ) from exc


@dataclass
class ServerConfig:
    """Configuration for running the GR00T inference server."""

    # Gr00t policy configs
    model_path: str | None = None
    """Path to the model checkpoint directory"""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag (name or value, case-insensitive). Run with --help to see known tags."""

    device: str = "cuda"
    """Device to run the model on"""

    # Replay policy configs
    dataset_path: str | None = None
    """Path to the dataset for replay trajectory"""

    modality_config_path: str | None = None
    """Path to the modality configuration file"""

    execution_horizon: int | None = None
    """Policy execution horizon during inference. Required when --dataset-path is set (ReplayPolicy)."""

    # Server configs
    host: str = "0.0.0.0"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper"""

    # Efficient inference configs (opt-in; no effect when efficient_method is unset)
    efficient_method: str = ""
    """Visual-token pruning method from gr00t.efficient (e.g. 'vlapruner', 'dummy'). Empty disables it."""

    efficient_keep_ratio: float = 1.0
    """Fraction of visual tokens to keep when efficient_method is set."""

    efficient_score_mode: str = "norm"
    """VLA-Pruner scoring mode: 'norm', 'mean_abs', 'attention', or 'action'."""

    efficient_reuse_steps: int = 2
    """SpecPrune: recompute pruning indices every N calls, reusing cached indices in between."""

    efficient_temporal_momentum: float = 0.8
    """EMA blend factor for temporal score smoothing (VLA-Pruner and SpecPrune)."""

    efficient_alpha: float = 0.5
    """VLA-Pruner semantic/action score blend weight."""

    efficient_beta: float = 0.5
    """VLA-Pruner semantic/action score blend weight (complements alpha)."""

    efficient_default_keep_ratio: float = 0.5
    """ADP: keep ratio used when the scheduler has no action-state signal to react to.
    This is the operative keep ratio for 'adp'/'adp_vlapruner' -- their constructors
    ignore --efficient-keep-ratio and read this instead."""

    efficient_contact_keep_ratio: float = 1.0
    """ADP: keep ratio when the scheduler detects contact/gripper activity."""

    efficient_move_keep_ratio: float = 0.6
    """ADP: keep ratio while the scheduler detects the arm is moving."""

    efficient_idle_keep_ratio: float = 0.5
    """ADP: keep ratio when the scheduler detects no significant action delta."""

    efficient_action_delta_threshold: float = 0.05
    """ADP: action-delta magnitude above which the scheduler considers the arm 'moving'."""

    efficient_prune_stage: str = "backbone"
    """Where to prune: 'backbone' (early Qwen3-VL layer, reduces latency) or 'action_head' (post-backbone)."""

    efficient_prune_layer: int = 3
    """Decoder layer index for backbone-stage pruning (clamped after DeepStack injection layers)."""

    efficient_torch_compile: bool = False
    """torch.compile the DiT action head (CUDA graphs) to remove kernel-launch overhead."""


def _attach_efficient_inference(model, config: "ServerConfig"):
    """Attach an opt-in gr00t.efficient visual-token pruning hook to a loaded model.

    No-op unless --efficient-method is set; existing server behavior is unchanged.
    Returns the pruning method instance so the caller can wire its ``reset()``
    into the server's "reset" endpoint -- without this, temporal-momentum state
    (e.g. VLAPruner.prev_score) would never be cleared between episodes/clients,
    since Gr00tPolicy.reset() has no knowledge of this externally-attached hook.
    """
    from gr00t.efficient.benchmark.registry import build_method

    method = build_method(
        config.efficient_method,
        keep_ratio=config.efficient_keep_ratio,
        score_mode=config.efficient_score_mode,
        reuse_steps=config.efficient_reuse_steps,
        temporal_momentum=config.efficient_temporal_momentum,
        alpha=config.efficient_alpha,
        beta=config.efficient_beta,
        default_keep_ratio=config.efficient_default_keep_ratio,
        contact_keep_ratio=config.efficient_contact_keep_ratio,
        move_keep_ratio=config.efficient_move_keep_ratio,
        idle_keep_ratio=config.efficient_idle_keep_ratio,
        action_delta_threshold=config.efficient_action_delta_threshold,
    )

    if config.efficient_prune_stage == "backbone":
        from gr00t.efficient.hooks.backbone_token_hook import (
            BackboneVisualTokenHook,
            attach_backbone_visual_token_hook,
        )

        hook = BackboneVisualTokenHook(
            method=method, enabled=True, prune_layer=config.efficient_prune_layer
        )
        attach_backbone_visual_token_hook(model, hook)
    else:
        from gr00t.efficient.hooks.visual_token_hook import (
            VisualTokenHook,
            attach_visual_token_hook,
        )

        hook = VisualTokenHook(method=method, enabled=True)
        attach_visual_token_hook(model, hook)

    if config.efficient_torch_compile:
        import torch

        action_head = getattr(model, "action_head", None)
        if action_head is not None and hasattr(action_head, "model"):
            action_head.model = torch.compile(
                action_head.model, mode="reduce-overhead", dynamic=False
            )

    print(
        f"  Efficient inference: method={config.efficient_method} "
        f"keep_ratio={config.efficient_keep_ratio} score_mode={config.efficient_score_mode} "
        f"reuse_steps={config.efficient_reuse_steps} "
        f"temporal_momentum={config.efficient_temporal_momentum} "
        f"stage={config.efficient_prune_stage} torch_compile={config.efficient_torch_compile}"
    )
    if config.efficient_method in ("adp", "adp_vlapruner"):
        print(
            f"    ADP scheduler: default_keep_ratio={config.efficient_default_keep_ratio} "
            f"contact_keep_ratio={config.efficient_contact_keep_ratio} "
            f"move_keep_ratio={config.efficient_move_keep_ratio} "
            f"idle_keep_ratio={config.efficient_idle_keep_ratio} "
            f"action_delta_threshold={config.efficient_action_delta_threshold} "
            f"(note: --efficient-keep-ratio is ignored for this method; "
            f"--efficient-default-keep-ratio is the operative value)"
        )
    return method


def main(config: ServerConfig):
    config.embodiment_tag = EmbodimentTag.resolve(config.embodiment_tag)
    print("Starting GR00T inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    # Create and start the server
    efficient_method = None
    if config.model_path is not None:
        # check if the model path exists
        if config.model_path.startswith("/") and not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model path {config.model_path} does not exist")
        policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
        )
        if config.efficient_method:
            efficient_method = _attach_efficient_inference(policy.model, config)
    elif config.dataset_path is not None:
        if config.execution_horizon is None:
            raise ValueError(
                "--execution-horizon is required when --dataset-path is set "
                "(ReplayPolicy needs a positive integer to advance episodes)."
            )
        if config.execution_horizon <= 0:
            raise ValueError(
                f"--execution-horizon must be positive; got {config.execution_horizon}."
            )

        modality_configs: dict[str, ModalityConfig] | None = None
        if config.modality_config_path is not None:
            config_path = Path(config.modality_config_path)
            if config_path.suffix == ".py":
                # The .py file is expected to call register_modality_config()
                # as an import side-effect; resolution falls through to
                # MODALITY_CONFIGS below.
                sys.path.append(str(config_path.parent))
                importlib.import_module(config_path.stem)
                print(f"Loaded modality config: {config_path}")
            elif config_path.suffix == ".json":
                modality_configs = _load_json_modality_configs(config_path)
            else:
                raise ValueError(
                    f"Unsupported modality config format: {config_path.suffix}. Use .py or .json"
                )

        # For .py configs (or no config path), look up from the registry
        if modality_configs is None:
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            modality_configs = MODALITY_CONFIGS.get(config.embodiment_tag.value)
            if modality_configs is None:
                raise ValueError(
                    f"No built-in modality config for embodiment tag "
                    f"'{config.embodiment_tag.name}' (value='{config.embodiment_tag.value}'). "
                    f"Available tags: {sorted(MODALITY_CONFIGS.keys())}. "
                    f"Please provide --modality-config-path (JSON or .py) "
                    f"when using this tag with ReplayPolicy."
                )
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    # Apply sim policy wrapper if needed
    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    if efficient_method is not None and hasattr(efficient_method, "reset"):
        # Gr00tPolicy.reset() / Gr00tSimPolicyWrapper.reset() have no knowledge
        # of the externally-attached pruning hook, so the "reset" endpoint would
        # otherwise be a complete no-op for VLA-Pruner's temporal-momentum state.
        # Wrap it so a client calling reset between episodes actually clears it.
        original_reset = policy.reset

        def reset_with_efficient_state(options: dict | None = None):
            efficient_method.reset()
            return original_reset(options)

        policy.reset = reset_with_efficient_state

    with PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    ) as server:
        try:
            server.run()
        except KeyboardInterrupt:
            print("\nShutting down server...")


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)

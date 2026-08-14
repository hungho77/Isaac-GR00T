#!/usr/bin/env python
"""Capture pooled N1.7 hidden states for CKA calibration.

Authentication is read by huggingface_hub from ``HF_TOKEN``. The token is
never accepted as a CLI argument, so it cannot leak through process listings.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.utils import parse_observation_gr00t
from gr00t.model.cka_pruning import get_prunable_module_lists
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import torch
import tyro


@dataclass
class Args:
    model_path: str = "nvidia/GR00T-N1.7-3B"
    dataset_path: str = "demo_data/droid_sample"
    embodiment_tag: str = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
    output_dir: str = "outputs/cka_n1d7/calibration"
    trajectory_ids: list[int] = field(default_factory=lambda: [0])
    samples_per_trajectory: int = 16
    sample_stride: int = 4
    denoising_steps: int = 1
    seed: int = 42
    device: str = "cuda:0"
    modules: list[str] = field(
        default_factory=lambda: ["backbone_language", "action_dit", "vl_self_attention"]
    )
    require_hf_token: bool = False


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _dataset_fingerprint(dataset_path: str) -> dict:
    root = Path(dataset_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
        if relative.startswith("meta/"):
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
    return {
        "scheme": "metadata-content-plus-file-inventory-v1",
        "sha256": digest.hexdigest(),
        "file_count": len(files),
    }


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state
    raise TypeError(f"Forward hook output has no tensor: {type(value)!r}")


class ActivationRecorder:
    def __init__(self, model, requested_modules: list[str]):
        available = get_prunable_module_lists(model)
        unknown = sorted(set(requested_modules) - set(available))
        if unknown:
            raise ValueError(f"Unavailable CKA module(s) {unknown}; available={sorted(available)}")
        self.activations: dict[str, list[list[np.ndarray]]] = {}
        self._pending: dict[str, list[list[np.ndarray]]] | None = None
        self.hook_calls_per_sample: list[dict[str, list[int]]] = []
        self.handles = []
        for module_name in requested_modules:
            layers = available[module_name]
            self.activations[module_name] = [[] for _ in layers]
            for layer_index, layer in enumerate(layers):
                self.handles.append(
                    layer.register_forward_hook(self._hook(module_name, layer_index))
                )

    def _hook(self, module_name: str, layer_index: int):
        def record(_module, _inputs, output):
            if self._pending is None:
                raise RuntimeError("Activation hook fired outside begin_sample()/end_sample()")
            tensor = _first_tensor(output).detach().float()
            if tensor.ndim < 2:
                tensor = tensor.reshape(1, -1)
            elif tensor.ndim > 2:
                tensor = tensor.mean(dim=tuple(range(1, tensor.ndim - 1)))
            rows = tensor.cpu().numpy()
            if rows.shape[0] != 1:
                raise RuntimeError(
                    "CKA capture processes one observation at a time; "
                    f"{module_name}[{layer_index}] produced batch size {rows.shape[0]}"
                )
            self._pending[module_name][layer_index].append(rows[0])

        return record

    def begin_sample(self) -> None:
        if self._pending is not None:
            raise RuntimeError("Previous CKA sample was not finalized")
        self._pending = {
            module_name: [[] for _ in layer_values]
            for module_name, layer_values in self.activations.items()
        }

    def abort_sample(self) -> None:
        self._pending = None

    def end_sample(self) -> None:
        if self._pending is None:
            raise RuntimeError("No active CKA sample")
        call_counts: dict[str, list[int]] = {}
        for module_name, layer_values in self._pending.items():
            counts = [len(values) for values in layer_values]
            if not counts or any(count < 1 for count in counts):
                self._pending = None
                raise RuntimeError(f"Missing hook calls for {module_name}: {counts}")
            if len(set(counts)) != 1:
                self._pending = None
                raise RuntimeError(f"Inconsistent hook calls within {module_name}: {counts}")
            call_counts[module_name] = counts
            for layer_index, values in enumerate(layer_values):
                # Action-DiT blocks execute once per denoising step while the
                # language backbone generally executes once. Store exactly one
                # representation per calibration observation by averaging the
                # repeated calls. CKA sample axes are therefore aligned across
                # modules and across checkpoints.
                pooled = np.mean(np.stack(values, axis=0), axis=0)
                self.activations[module_name][layer_index].append(pooled)
        self.hook_calls_per_sample.append(call_counts)
        self._pending = None

    def close(self) -> None:
        self.abort_sample()
        for handle in self.handles:
            handle.remove()

    def save(self, output_dir: Path, metadata: dict[str, Any]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for module_name, layer_values in self.activations.items():
            counts = {len(values) for values in layer_values}
            if len(counts) != 1 or not counts or next(iter(counts)) < 2:
                raise RuntimeError(
                    f"Inconsistent/insufficient hook calls for {module_name}: "
                    f"{[len(values) for values in layer_values]}"
                )
            for layer_index, values in enumerate(layer_values):
                arrays[f"{module_name}__layer_{layer_index:03d}"] = np.stack(values)
        np.savez_compressed(output_dir / "activations.npz", **arrays)
        metadata["activation_schema_version"] = 2
        metadata["sample_aggregation"] = "mean_over_forward_calls"
        metadata["hook_calls_per_sample"] = self.hook_calls_per_sample
        metadata["activation_shapes"] = {key: list(value.shape) for key, value in arrays.items()}
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _prepare_observation(traj, step, modality_configs, embodiment_tag, loader):
    data_point = extract_step_data(traj, step, modality_configs, embodiment_tag)
    observation = {}
    for key, value in data_point.states.items():
        observation[f"state.{key}"] = value
    for key, value in data_point.images.items():
        observation[f"video.{key}"] = np.asarray(value)
    for language_key in loader.modality_configs["language"].modality_keys:
        observation[language_key] = data_point.text
    return parse_observation_gr00t(observation, loader.modality_configs)


def main(args: Args) -> None:
    if args.require_hf_token:
        from huggingface_hub import get_token

        if get_token() is None:
            raise RuntimeError(
                "No Hugging Face token was found in HF_TOKEN or the huggingface_hub cache"
            )
    if not torch.cuda.is_available():
        raise RuntimeError("N1.7 CKA capture requires a CUDA GPU")
    if args.samples_per_trajectory < 1 or args.sample_stride < 1:
        raise ValueError("samples_per_trajectory and sample_stride must be positive")
    if len(set(args.trajectory_ids)) != len(args.trajectory_ids):
        raise ValueError(f"trajectory_ids contains duplicates: {args.trajectory_ids}")
    _seed_everything(args.seed)

    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_path,
        device=args.device,
    )
    policy.model.action_head.num_inference_timesteps = args.denoising_steps
    loader = LeRobotEpisodeLoader(args.dataset_path, policy.get_modality_config())
    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    inference_modalities = deepcopy(loader.modality_configs)
    inference_modalities.pop("action")
    recorder = ActivationRecorder(policy.model, args.modules)
    prunable_layers = get_prunable_module_lists(policy.model)
    module_layer_indices = {
        name: [
            int(getattr(layer, "_gr00t_original_index", layer_index))
            for layer_index, layer in enumerate(prunable_layers[name])
        ]
        for name in args.modules
    }

    sampled_steps = []
    try:
        with torch.inference_mode():
            for trajectory_id in args.trajectory_ids:
                traj = loader[trajectory_id]
                steps = list(range(0, len(traj), args.sample_stride))[: args.samples_per_trajectory]
                for step in steps:
                    sample_seed = args.seed + trajectory_id * 1_000_000 + step
                    _seed_everything(sample_seed)
                    observation = _prepare_observation(
                        traj, step, inference_modalities, embodiment_tag, loader
                    )
                    recorder.begin_sample()
                    try:
                        policy.get_action(observation)
                        recorder.end_sample()
                    except Exception:
                        recorder.abort_sample()
                        raise
                    sampled_steps.append(
                        {"trajectory_id": trajectory_id, "step": step, "seed": sample_seed}
                    )
    finally:
        recorder.close()

    recorder.save(
        Path(args.output_dir),
        {
            "args": asdict(args),
            "sampled_steps": sampled_steps,
            "dataset_fingerprint": _dataset_fingerprint(args.dataset_path),
            "gpu": torch.cuda.get_device_name(torch.device(args.device)),
            "torch_version": torch.__version__,
            "module_layer_indices": module_layer_indices,
            "cka_pruning_manifest": getattr(policy.model.config, "cka_pruning_manifest", None),
        },
    )
    print(f"Saved CKA calibration activations to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main(tyro.cli(Args))

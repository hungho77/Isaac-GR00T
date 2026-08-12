"""Memory-bounded Action-DiT LoRA recovery for GR00T N1.7.

The integration is opt-in through ``GR00T_ACTION_DIT_LORA_RANK``. Full base
weights are loaded and structurally pruned before adapters are installed.
Intermediate checkpoints can contain only trainable tensors; the final export
merges the adapters and remains loadable by the unmodified GR00T architecture.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import logging
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


logger = logging.getLogger(__name__)
ADAPTER_FORMAT = "adapter_only_v1"
ADAPTER_METADATA_NAME = "gr00t_adapter_checkpoint.json"


def _manifest_sha256(manifest: dict | None) -> str | None:
    if manifest is None:
        return None
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _update_file_sample(digest, path: Path, *, include_content: bool) -> None:
    size = path.stat().st_size
    digest.update(str(size).encode("ascii"))
    if not include_content:
        return
    sample_size = 1024 * 1024
    with path.open("rb") as file:
        digest.update(file.read(sample_size))
        if size > sample_size:
            file.seek(max(0, size - sample_size))
            digest.update(file.read(sample_size))


def _directory_fingerprint(path: str | Path, *, checkpoint: bool) -> dict:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Fingerprint directory does not exist: {root}")
    digest = hashlib.sha256()
    files = sorted(
        file
        for file in root.rglob("*")
        if file.is_file()
        and not {".cache", "__pycache__"}.intersection(file.relative_to(root).parts)
    )
    for file in files:
        relative = file.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        if checkpoint:
            include_content = (
                file.suffix == ".safetensors" or file.stat().st_size <= 8 * 1024 * 1024
            )
        else:
            include_content = relative.startswith("meta/")
        _update_file_sample(digest, file, include_content=include_content)
    return {
        "scheme": "checkpoint-sampled-content-v1"
        if checkpoint
        else "dataset-metadata-inventory-v1",
        "sha256": digest.hexdigest(),
        "file_count": len(files),
    }


def _checkpoint_fingerprint(source: str | Path | None) -> dict | None:
    if source is None:
        return None
    path = Path(source).expanduser()
    if path.is_dir():
        return _directory_fingerprint(path, checkpoint=True)
    return {"scheme": "hugging-face-id-v1", "identifier": str(source)}


class LoRALinear(nn.Module):
    """Linear layer with a frozen base and trainable low-rank delta."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.weight = linear.weight
        self.bias = linear.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                self.in_features,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                self.out_features,
                self.rank,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.merged = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = F.linear(inputs, self.weight, self.bias)
        if self.lora_A is None or self.lora_B is None:
            return result
        hidden = F.linear(self.dropout(inputs), self.lora_A)
        delta = F.linear(hidden, self.lora_B)
        return result + delta * self.scaling

    @torch.no_grad()
    def merge_for_export_(self) -> None:
        if self.merged or self.lora_A is None or self.lora_B is None:
            return
        delta = torch.matmul(self.lora_B.float(), self.lora_A.float())
        self.weight.add_(delta.to(dtype=self.weight.dtype) * self.scaling)
        self.register_parameter("lora_A", None)
        self.register_parameter("lora_B", None)
        self.dropout = nn.Identity()
        self.merged = True


def _replace_linear_children(module: nn.Module, rank: int, alpha: float, dropout: float) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            count += 1
        else:
            count += _replace_linear_children(child, rank, alpha, dropout)
    return count


def configure_action_dit_lora_from_env(
    model: nn.Module,
    *,
    checkpoint_source: str | Path | None = None,
    dataset_paths: list[str | Path] | None = None,
) -> dict:
    """Freeze pretrained weights and install matched LoRA adapters."""
    rank = int(os.environ.get("GR00T_ACTION_DIT_LORA_RANK", "0") or 0)
    if rank <= 0:
        return {"enabled": False, "linear_modules": 0, "trainable_parameters": 0}
    alpha = float(os.environ.get("GR00T_ACTION_DIT_LORA_ALPHA", str(2 * rank)))
    dropout = float(os.environ.get("GR00T_ACTION_DIT_LORA_DROPOUT", "0.05"))
    if alpha <= 0 or not 0.0 <= dropout < 1.0:
        raise ValueError(f"Invalid LoRA alpha/dropout: alpha={alpha}, dropout={dropout}")

    root = model.module if hasattr(model, "module") else model
    root.requires_grad_(False)
    diffusion = root.action_head.model
    blocks = diffusion.transformer_blocks
    replaced = sum(
        _replace_linear_children(block, rank=rank, alpha=alpha, dropout=dropout) for block in blocks
    )
    if replaced == 0:
        raise RuntimeError("No Action-DiT Linear modules were found for LoRA injection")

    for name in ("timestep_encoder", "proj_out_1", "proj_out_2"):
        module = getattr(diffusion, name, None)
        if module is not None:
            module.requires_grad_(True)

    trainable = sum(parameter.numel() for parameter in root.parameters() if parameter.requires_grad)
    max_trainable = int(
        os.environ.get("GR00T_ACTION_DIT_LORA_MAX_TRAINABLE_PARAMETERS", "64000000")
    )
    if trainable > max_trainable:
        raise RuntimeError(
            f"LoRA recovery budget violated: {trainable:,} trainable parameters; "
            f"expected <= {max_trainable:,}"
        )
    lora_trainable = sum(
        parameter.numel()
        for module in root.modules()
        if isinstance(module, LoRALinear)
        for parameter in (module.lora_A, module.lora_B)
        if parameter is not None
    )
    metadata = {
        "enabled": True,
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "retained_action_dit_blocks": len(blocks),
        "linear_modules": replaced,
        "lora_trainable_parameters": lora_trainable,
        "total_trainable_parameters": trainable,
        "max_trainable_parameters": max_trainable,
        "cka_pruning_manifest_sha256": _manifest_sha256(
            getattr(root.config, "cka_pruning_manifest", None)
        ),
        "base_checkpoint_fingerprint": _checkpoint_fingerprint(checkpoint_source),
        "training_dataset_fingerprints": [
            _directory_fingerprint(path, checkpoint=False) for path in (dataset_paths or [])
        ],
    }
    root._gr00t_action_dit_lora_metadata = metadata
    logger.warning("Configured Action-DiT LoRA recovery: %s", metadata)
    return metadata


def adapter_checkpoint_metadata(model: nn.Module) -> dict:
    root = model.module if hasattr(model, "module") else model
    lora = deepcopy(getattr(root, "_gr00t_action_dit_lora_metadata", None))
    if not lora or not lora.get("enabled"):
        raise RuntimeError("Adapter-only checkpoint requested without active Action-DiT LoRA")
    trainable_names = sorted(
        name for name, parameter in root.named_parameters() if parameter.requires_grad
    )
    return {
        "format": ADAPTER_FORMAT,
        "lora": lora,
        "trainable_parameter_names": trainable_names,
    }


def validate_adapter_checkpoint(
    model: nn.Module,
    checkpoint: str | Path,
    *,
    require_training_state: bool = True,
) -> None:
    checkpoint = Path(checkpoint)
    path = checkpoint / ADAPTER_METADATA_NAME
    if not path.is_file():
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    current = adapter_checkpoint_metadata(model)
    mismatches = {
        key: (saved.get(key), current.get(key))
        for key in ("format", "lora", "trainable_parameter_names")
        if saved.get(key) != current.get(key)
    }
    if mismatches:
        raise ValueError(f"Adapter checkpoint is incompatible with the current model: {mismatches}")
    if not require_training_state:
        return

    required = (
        "model.safetensors",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if not list(checkpoint.glob("rng_state*.pth")):
        missing.append("rng_state*.pth")
    incomplete = [
        file.name
        for file in checkpoint.iterdir()
        if file.is_file() and (file.name.endswith(".incomplete") or file.stat().st_size == 0)
    ]
    if missing or incomplete:
        raise ValueError(
            f"Adapter checkpoint is not safely resumable: missing={missing}, "
            f"incomplete_or_empty={incomplete}"
        )

    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    directory_step = checkpoint.name.removeprefix("checkpoint-")
    if directory_step.isdigit() and int(state.get("global_step", -1)) != int(directory_step):
        raise ValueError(
            f"Adapter checkpoint step mismatch: directory={directory_step}, "
            f"trainer_state={state.get('global_step')}"
        )
    from safetensors import safe_open

    with safe_open(str(checkpoint / "model.safetensors"), framework="pt", device="cpu") as file:
        saved_tensor_names = set(file.keys())
    expected_tensor_names = set(saved["trainable_parameter_names"])
    if saved_tensor_names != expected_tensor_names:
        raise ValueError(
            "Adapter checkpoint tensor inventory differs from metadata: "
            f"missing={sorted(expected_tensor_names - saved_tensor_names)}, "
            f"unexpected={sorted(saved_tensor_names - expected_tensor_names)}"
        )


def merge_action_dit_lora_for_export(model: nn.Module) -> dict:
    root = model.module if hasattr(model, "module") else model
    merged = 0
    for module in root.modules():
        if isinstance(module, LoRALinear) and not module.merged:
            module.merge_for_export_()
            merged += 1
    metadata = dict(getattr(root, "_gr00t_action_dit_lora_metadata", {}))
    metadata.update({"merged_for_export": True, "merged_linear_modules": merged})
    root._gr00t_action_dit_lora_metadata = metadata
    logger.warning("Merged Action-DiT LoRA adapters for final export: %s", metadata)
    return metadata

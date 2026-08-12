import json

from gr00t.model.action_dit_lora import (
    ADAPTER_METADATA_NAME,
    LoRALinear,
    adapter_checkpoint_metadata,
    configure_action_dit_lora_from_env,
    merge_action_dit_lora_for_export,
    validate_adapter_checkpoint,
)
import pytest
import torch
from torch import nn


class _Config:
    cka_pruning_manifest = {"modules": {"action_dit": {"keep_indices": [0, 2]}}}

    def save_pretrained(self, output_dir):
        del output_dir


class _Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(4, 4), nn.SiLU(), nn.Linear(4, 4)) for _ in range(2)]
        )
        self.timestep_encoder = nn.Linear(1, 4)
        self.proj_out_1 = nn.Linear(4, 4)
        self.proj_out_2 = nn.Linear(4, 2)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.action_head = nn.Module()
        self.action_head.model = _Diffusion()
        self.config = _Config()


@pytest.fixture
def lora_env(monkeypatch):
    monkeypatch.setenv("GR00T_ACTION_DIT_LORA_RANK", "2")
    monkeypatch.setenv("GR00T_ACTION_DIT_LORA_ALPHA", "4")
    monkeypatch.setenv("GR00T_ACTION_DIT_LORA_DROPOUT", "0")
    monkeypatch.setenv("GR00T_ACTION_DIT_LORA_MAX_TRAINABLE_PARAMETERS", "10000")


def test_lora_freezes_base_and_merges_for_export(lora_env):
    model = _Model()
    metadata = configure_action_dit_lora_from_env(model)

    assert metadata["linear_modules"] == 4
    assert metadata["retained_action_dit_blocks"] == 2
    assert not model.backbone.weight.requires_grad
    assert model.action_head.model.timestep_encoder.weight.requires_grad
    assert all(
        not layer.weight.requires_grad for layer in model.modules() if isinstance(layer, LoRALinear)
    )

    first = next(layer for layer in model.modules() if isinstance(layer, LoRALinear))
    with torch.no_grad():
        first.lora_B.fill_(0.25)
    before = first.weight.detach().clone()
    merged = merge_action_dit_lora_for_export(model)

    assert merged["merged_linear_modules"] == 4
    assert first.lora_A is None
    assert first.lora_B is None
    assert not torch.equal(before, first.weight)


def test_adapter_checkpoint_rejects_different_manifest(tmp_path, lora_env):
    model = _Model()
    configure_action_dit_lora_from_env(model)
    metadata = adapter_checkpoint_metadata(model)
    (tmp_path / ADAPTER_METADATA_NAME).write_text(json.dumps(metadata), encoding="utf-8")

    validate_adapter_checkpoint(model, tmp_path, require_training_state=False)
    model._gr00t_action_dit_lora_metadata["cka_pruning_manifest_sha256"] = "different"
    with pytest.raises(ValueError, match="incompatible"):
        validate_adapter_checkpoint(model, tmp_path, require_training_state=False)


def test_lora_is_disabled_without_rank(monkeypatch):
    monkeypatch.delenv("GR00T_ACTION_DIT_LORA_RANK", raising=False)
    model = _Model()
    result = configure_action_dit_lora_from_env(model)

    assert result == {"enabled": False, "linear_modules": 0, "trainable_parameters": 0}
    assert all(not isinstance(module, LoRALinear) for module in model.modules())

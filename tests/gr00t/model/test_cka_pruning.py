from types import SimpleNamespace

from gr00t.model.cka_pruning import (
    apply_pruning_manifest,
    linear_cka,
    select_keep_indices,
    validate_pruning_manifest,
)
import numpy as np
import pytest
from torch import nn


class DummyStack(nn.Module):
    pass


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(cka_pruning_manifest=None)
        self.backbone = DummyStack()
        self.backbone.model = DummyStack()
        self.backbone.model.language_model = DummyStack()
        self.backbone.model.language_model.layers = nn.ModuleList(
            [nn.Linear(4, 4) for _ in range(12)]
        )
        self.action_head = DummyStack()
        self.action_head.model = DummyStack()
        self.action_head.model.transformer_blocks = nn.ModuleList(
            [nn.Linear(4, 4) for _ in range(16)]
        )
        self.action_head.vl_self_attention = DummyStack()
        self.action_head.vl_self_attention.transformer_blocks = nn.ModuleList(
            [nn.Linear(4, 4) for _ in range(4)]
        )


@pytest.fixture
def manifest():
    return {
        "schema_version": 1,
        "model_type": "Gr00tN1d7",
        "modules": {
            "backbone_language": {"original_depth": 12, "keep_indices": [0, 2, 4, 7, 11]},
            "action_dit": {
                "original_depth": 16,
                "keep_indices": [0, 3, 6, 8, 11, 12, 14, 15],
            },
            "vl_self_attention": {"original_depth": 4, "keep_indices": [0, 1, 3]},
        },
    }


def test_apply_manifest_is_structural_and_idempotent(manifest):
    model = DummyModel()
    stats = apply_pruning_manifest(model, manifest)
    assert stats["depth_after"] == {
        "backbone_language": 5,
        "action_dit": 8,
        "vl_self_attention": 3,
    }
    assert stats["parameters_after"] < stats["parameters_before"]
    action_blocks = model.action_head.model.transformer_blocks
    assert [layer._gr00t_original_index for layer in action_blocks] == [
        0,
        3,
        6,
        8,
        11,
        12,
        14,
        15,
    ]
    second = apply_pruning_manifest(model, manifest)
    assert second["parameters_before"] == second["parameters_after"]


def test_manifest_rejects_action_topology_without_image_cross(manifest):
    manifest["modules"]["action_dit"]["keep_indices"] = [0, 1, 4, 5]
    with pytest.raises(ValueError, match="image-cross"):
        validate_pruning_manifest(manifest)


def test_linear_cka_identical_representations():
    rng = np.random.default_rng(4)
    activation = rng.normal(size=(16, 32))
    assert linear_cka(activation, activation) == pytest.approx(1.0)


def test_action_selection_preserves_all_attention_categories():
    scores = [0.95 - index * 0.01 for index in range(15)]
    keep = select_keep_indices(scores, 8, module_name="action_dit")
    assert any(index % 4 == 0 for index in keep)
    assert any(index % 4 == 2 for index in keep)
    assert any(index % 2 == 1 for index in keep)
    assert keep[0] == 0
    assert keep[-1] == 15


def test_non_action_selection_preserves_boundaries_when_budget_allows():
    scores = [0.99, 0.98, 0.97, 0.96, 0.95]
    keep = select_keep_indices(scores, 3, module_name="backbone_language")
    assert keep[0] == 0
    assert keep[-1] == 5


def test_linear_cka_is_numerically_bounded():
    rng = np.random.default_rng(8)
    left = rng.normal(size=(32, 17))
    right = left + rng.normal(scale=1e-12, size=left.shape)
    assert 0.0 <= linear_cka(left, right) <= 1.0

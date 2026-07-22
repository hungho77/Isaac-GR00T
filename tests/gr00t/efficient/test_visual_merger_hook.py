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

"""
Test VisualMergerHook (Hook A: after_visual_merger) with synthetic tensors --
no real Qwen3-VL model or checkpoint required. Covers mask-only correctness,
shape preservation (the whole safety argument for this hook), metadata, and
the attach() monkey-patch plumbing against a fake get_image_features.
"""

from gr00t.efficient.benchmark.methods import DummyPruningMethod, SpecPruneMethod, VLAPrunerMethod
from gr00t.efficient.hooks.visual_merger_hook import VisualMergerHook, attach_visual_merger_hook
import torch


def _two_camera_embeds(seed: int = 0) -> tuple[list, list]:
    torch.manual_seed(seed)
    cam1 = torch.randn(64, 32)
    cam2 = torch.randn(64, 32)
    return [cam1, cam2], [64, 64]


def test_disabled_hook_returns_unchanged_list():
    hook = VisualMergerHook(method=DummyPruningMethod(keep_ratio=0.5, mode="first"), enabled=False)
    embeds, sizes = _two_camera_embeds()

    result = hook.apply_image_features(embeds, sizes)

    assert result is embeds
    assert hook.last_metadata["pruned"] is False
    assert hook.last_metadata["hook_enabled"] is False


def test_mask_only_preserves_shape_and_zeros_low_score_rows():
    hook = VisualMergerHook(method=DummyPruningMethod(keep_ratio=0.5, mode="first"), enabled=True)
    embeds, sizes = _two_camera_embeds()
    original = [t.clone() for t in embeds]

    result = hook.apply_image_features(embeds, sizes)

    assert len(result) == 2
    assert result[0].shape == (64, 32)
    assert result[1].shape == (64, 32)  # shape unchanged -- the whole point of mask-only

    combined = torch.cat(result, dim=0)
    combined_original = torch.cat(original, dim=0)
    nonzero_rows = combined.abs().sum(dim=-1) > 0
    zero_rows = ~nonzero_rows
    # "first" mode with keep_ratio=0.5 over 128 tokens keeps the first 64.
    assert int(nonzero_rows.sum()) == 64
    assert torch.equal(combined[nonzero_rows], combined_original[nonzero_rows])
    assert torch.equal(combined[zero_rows], torch.zeros_like(combined[zero_rows]))


def test_specprune_masks_through_merger_hook():
    # SpecPruneVLA previously never set `last_selected_indices` (only
    # `cached_indices`, for its own reuse bookkeeping), so this hook's
    # `_selected_indices()` lookup always came up empty and silently no-opped
    # with hook_error="missing_selected_indices" -- confirm it actually masks now.
    hook = VisualMergerHook(
        method=SpecPruneMethod(keep_ratio=0.5, score_mode="norm", reuse_steps=2), enabled=True
    )
    embeds, sizes = _two_camera_embeds()

    result = hook.apply_image_features(embeds, sizes)

    assert result[0].shape == (64, 32)
    assert result[1].shape == (64, 32)
    nonzero_rows = torch.cat(result, dim=0).abs().sum(dim=-1) > 0
    assert int(nonzero_rows.sum()) == 64
    assert "hook_error" not in hook.last_metadata


def test_mask_only_metadata_matches_spec():
    hook = VisualMergerHook(
        method=VLAPrunerMethod(keep_ratio=0.75, score_mode="norm"), enabled=True
    )
    embeds, sizes = _two_camera_embeds()

    hook.apply_image_features(embeds, sizes)
    meta = hook.last_metadata

    assert meta["hook_name"] == "after_visual_merger"
    assert meta["mode"] == "mask_only"
    assert meta["visual_token_count_before"] == 128
    assert meta["visual_token_count_after"] == 128  # unchanged shape by design
    assert meta["nonzero_token_count_after"] == 96  # ceil(128*0.75)
    assert abs(meta["effective_keep_ratio"] - 0.75) < 1e-6
    assert abs(meta["token_reduction_ratio"] - 0.25) < 1e-6
    assert meta["pruning_method"] == "vlapruner"
    assert meta["per_camera_token_counts"] == [64, 64]
    assert meta["pruned"] is True


def test_keep_ratio_one_masks_nothing():
    hook = VisualMergerHook(method=DummyPruningMethod(keep_ratio=1.0, mode="first"), enabled=True)
    embeds, sizes = _two_camera_embeds()
    original = [t.clone() for t in embeds]

    result = hook.apply_image_features(embeds, sizes)

    assert torch.equal(torch.cat(result, dim=0), torch.cat(original, dim=0))
    assert hook.last_metadata["nonzero_token_count_after"] == 128
    assert hook.last_metadata["pruned"] is False


def test_gather_mode_request_falls_back_to_mask_only():
    # Gather is rejected at this hook point (get_placeholder_mask would raise
    # ValueError on a shape mismatch); the hook must not silently attempt it.
    hook = VisualMergerHook(method=DummyPruningMethod(keep_ratio=0.5), enabled=True, mode="gather")
    assert hook.mode == "mask_only"


class _FakeQwen3VLModel:
    """Stand-in for Qwen3VLModel exposing only get_image_features."""

    def __init__(self, embeds, sizes):
        self._embeds = embeds
        self._sizes = sizes
        self.calls = 0

    def get_image_features(self, pixel_values, image_grid_thw=None):
        self.calls += 1
        return tuple(self._embeds), ["deepstack_placeholder"]


class _FakeConditionalGenerationModel:
    """Stand-in for Qwen3VLForConditionalGeneration: exposes .model = Qwen3VLModel."""

    def __init__(self, qwen_model):
        self.model = qwen_model


class _FakeBackboneModel:
    """Stand-in for Qwen3Backbone: exposes .model = Qwen3VLForConditionalGeneration."""

    def __init__(self, qwen_model):
        self.model = _FakeConditionalGenerationModel(qwen_model)


class _FakeGr00tModel:
    """Stand-in for Gr00tN1d7: exposes .backbone = Qwen3Backbone."""

    def __init__(self, qwen_model):
        self.backbone = _FakeBackboneModel(qwen_model)


def test_attach_patches_get_image_features_and_preserves_deepstack():
    embeds, sizes = _two_camera_embeds()
    qwen_model = _FakeQwen3VLModel(embeds, sizes)
    model = _FakeGr00tModel(qwen_model)

    hook = VisualMergerHook(method=DummyPruningMethod(keep_ratio=0.5, mode="first"), enabled=True)
    attach_visual_merger_hook(model, hook)

    result_embeds, deepstack = qwen_model.get_image_features(pixel_values=None, image_grid_thw=None)

    assert len(result_embeds) == 2
    assert result_embeds[0].shape == (64, 32)
    assert deepstack == ["deepstack_placeholder"]  # untouched, as documented
    assert hook.last_metadata["hook_name"] == "after_visual_merger"
    assert qwen_model.calls == 1


def test_attach_is_idempotent():
    embeds, sizes = _two_camera_embeds()
    qwen_model = _FakeQwen3VLModel(embeds, sizes)
    model = _FakeGr00tModel(qwen_model)

    hook1 = VisualMergerHook(method=DummyPruningMethod(keep_ratio=0.5), enabled=True)
    attach_visual_merger_hook(model, hook1)
    patched_once = qwen_model.get_image_features

    hook2 = VisualMergerHook(method=DummyPruningMethod(keep_ratio=0.9), enabled=True)
    attach_visual_merger_hook(model, hook2)

    assert qwen_model.get_image_features == patched_once  # not re-wrapped
    assert qwen_model._efficient_merger_hook is hook2  # but the active hook updates

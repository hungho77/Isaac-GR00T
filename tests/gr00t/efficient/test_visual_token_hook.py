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
Test VisualTokenHook safety on backbone outputs: the hook must only ever
prune isolated visual tokens and must no-op (never touch text/state tokens)
when visual tokens cannot be isolated.
"""

from gr00t.efficient.benchmark.methods import (
    ADPMethod,
    DummyPruningMethod,
    SpecPruneMethod,
    VLAPrunerMethod,
)
from gr00t.efficient.hooks.visual_token_hook import VisualTokenHook
import torch


def _backbone_output(
    batch_size: int = 2,
    seq_len: int = 10,
    dim: int = 8,
    visual_positions: tuple[int, ...] = (2, 3, 4, 5, 6, 7),
) -> dict:
    torch.manual_seed(0)
    features = torch.randn(batch_size, seq_len, dim)
    image_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    image_mask[:, list(visual_positions)] = True
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    return {
        "backbone_features": features,
        "image_mask": image_mask,
        "backbone_attention_mask": attention_mask,
    }


def _backbone_output_with_visual_norms(
    norms: list[float],
    batch_size: int = 1,
    seq_len: int = 10,
    visual_positions: tuple[int, ...] = (2, 3, 4, 5, 6, 7),
) -> dict:
    """Backbone output whose visual rows are one-hot vectors scaled by ``norms``.

    Each original visual position gets its own orthogonal one-hot direction, so
    a pruned row's surviving position can be decoded later via argmax -- this
    is what lets the reuse test below tell "same indices carried over" apart
    from "coincidentally scored the same" across calls with different inputs.
    """
    dim = len(visual_positions)
    features = torch.zeros(batch_size, seq_len, dim)
    for row, (pos, norm) in enumerate(zip(visual_positions, norms)):
        features[:, pos, row] = norm
    image_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    image_mask[:, list(visual_positions)] = True
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    return {
        "backbone_features": features,
        "image_mask": image_mask,
        "backbone_attention_mask": attention_mask,
    }


def _kept_original_visual_indices(result: dict) -> list[int]:
    """Decode which original one-hot visual position(s) survived pruning."""
    features = result["backbone_features"]
    mask = result["image_mask"]
    visual_rows = features[0][mask[0]]
    return sorted(int(idx) for idx in visual_rows.abs().argmax(dim=-1).tolist())


def test_missing_image_mask_is_noop():
    hook = VisualTokenHook(method=DummyPruningMethod(keep_ratio=0.5, mode="first"), enabled=True)
    output = {"backbone_features": torch.randn(2, 10, 8)}
    original = output["backbone_features"].clone()

    result = hook.apply_backbone_output(output)

    assert torch.equal(result["backbone_features"], original)
    assert hook.last_metadata["hook_error"] == "visual_tokens_not_isolated"
    assert hook.last_metadata["pruned"] is False


def test_unequal_visual_counts_is_noop():
    hook = VisualTokenHook(method=DummyPruningMethod(keep_ratio=0.5, mode="first"), enabled=True)
    output = _backbone_output()
    output["image_mask"][1, 2] = False
    original = output["backbone_features"].clone()

    result = hook.apply_backbone_output(output)

    assert torch.equal(result["backbone_features"], original)
    assert result["image_mask"].shape == (2, 10)
    assert hook.last_metadata["hook_error"] == "visual_tokens_not_isolated"
    assert hook.last_metadata["pruned"] is False


def test_visual_only_pruning_preserves_non_visual_tokens():
    hook = VisualTokenHook(method=DummyPruningMethod(keep_ratio=0.5, mode="first"), enabled=True)
    output = _backbone_output()
    features = output["backbone_features"].clone()

    result = hook.apply_backbone_output(output)

    # keep = ceil(6 * 0.5) = 3 visual tokens; "first" keeps visual positions 2, 3, 4.
    # Non-visual positions 0, 1, 8, 9 must all survive.
    expected_positions = [0, 1, 2, 3, 4, 8, 9]
    pruned = result["backbone_features"]
    assert pruned.shape == (2, 7, 8)
    assert torch.equal(pruned, features[:, expected_positions, :])
    assert result["image_mask"].shape == (2, 7)
    assert int(result["image_mask"].sum()) == 2 * 3
    assert result["backbone_attention_mask"].shape == (2, 7)
    assert bool(result["backbone_attention_mask"].all())
    assert hook.last_metadata["hook_scope"] == "visual_tokens_only"
    assert hook.last_metadata["full_token_count_before"] == 10
    assert hook.last_metadata["full_token_count_after"] == 7


def test_vlapruner_hook_prunes_and_keeps_masks_consistent():
    hook = VisualTokenHook(method=VLAPrunerMethod(keep_ratio=0.5, score_mode="norm"), enabled=True)
    output = _backbone_output()

    result = hook.apply_backbone_output(output)

    pruned = result["backbone_features"]
    assert pruned.shape == (2, 7, 8)
    assert result["image_mask"].shape[1] == pruned.shape[1]
    assert result["backbone_attention_mask"].shape[1] == pruned.shape[1]
    assert int(result["image_mask"].sum(dim=1)[0]) == 3


def test_score_mode_effective_reports_silent_fallback():
    hook = VisualTokenHook(
        method=VLAPrunerMethod(keep_ratio=0.5, score_mode="attention"), enabled=True
    )
    output = _backbone_output()

    hook.apply_backbone_output(output)

    # No attention tensor is available in the hook path, so VLAPruner silently
    # falls back to norm scoring; the metadata must make that visible.
    assert hook.last_metadata["score_mode"] == "attention"
    assert hook.last_metadata["score_mode_effective"] == "norm"
    assert hook.last_metadata["used_attention"] is False


def test_specprune_hook_reuses_cached_indices_across_consecutive_calls():
    # keep_ratio=0.5 over 6 visual positions -> keep 3. temporal_momentum=0.0
    # isolates index reuse from SpecPrune's separate EMA score-smoothing (which
    # persists prev_score across calls, including reuse steps).
    hook = VisualTokenHook(
        method=SpecPruneMethod(
            keep_ratio=0.5, score_mode="norm", reuse_steps=2, temporal_momentum=0.0
        ),
        enabled=True,
    )

    # timestep=0 (fresh compute): increasing norms -> top-3 are relative indices [3, 4, 5].
    result0 = hook.apply_backbone_output(_backbone_output_with_visual_norms([1, 2, 3, 4, 5, 6]))
    assert hook.last_metadata["reused_indices"] is False
    kept0 = _kept_original_visual_indices(result0)
    assert kept0 == [3, 4, 5]

    # timestep=1 (should reuse): norms reversed, so a fresh score would instead
    # pick [0, 1, 2]. If the hook still keeps [3, 4, 5], the cached indices from
    # the previous call were genuinely reused, not just coincidentally repeated.
    result1 = hook.apply_backbone_output(_backbone_output_with_visual_norms([6, 5, 4, 3, 2, 1]))
    assert hook.last_metadata["reused_indices"] is True
    assert _kept_original_visual_indices(result1) == kept0

    # timestep=2 (reuse_steps boundary): must recompute, and with the norms
    # still reversed the fresh selection must actually differ from the stale one.
    result2 = hook.apply_backbone_output(_backbone_output_with_visual_norms([6, 5, 4, 3, 2, 1]))
    assert hook.last_metadata["reused_indices"] is False
    kept2 = _kept_original_visual_indices(result2)
    assert kept2 == [0, 1, 2]
    assert kept2 != kept0


def test_adp_hook_prunes_instead_of_silently_noop():
    # ADPMethod used to create a fresh, transient DummyVisualTokenPruner every
    # call and never expose its selection, so this hook's _selected_indices()
    # lookup always came up empty -- a real n=100 checkpoint run showed
    # ADP's token_reduction_ratio stuck at 0.000 the whole time. Confirm the
    # hook now actually shortens the sequence.
    hook = VisualTokenHook(method=ADPMethod(default_keep_ratio=0.5, mode="first"), enabled=True)
    output = _backbone_output()
    original_seq_len = output["backbone_features"].shape[1]

    result = hook.apply_backbone_output(output)

    assert result["backbone_features"].shape[1] < original_seq_len
    assert hook.last_metadata.get("hook_error") is None
    assert hook.last_metadata["hook_scope"] == "visual_tokens_only"


def test_disabled_hook_is_noop():
    hook = VisualTokenHook(method=DummyPruningMethod(keep_ratio=0.5, mode="first"), enabled=False)
    output = _backbone_output()
    original = output["backbone_features"].clone()

    result = hook.apply_backbone_output(output)

    assert torch.equal(result["backbone_features"], original)

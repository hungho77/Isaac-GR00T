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
Test BackboneVisualTokenHook against a tiny real Qwen3VLTextModel: the patched
forward must be numerically identical when the hook is disabled or keep_ratio
is 1.0, and must shorten the sequence (visual tokens only) when pruning.
"""

import pytest
import torch


transformers = pytest.importorskip("transformers")

from gr00t.efficient.benchmark.methods import DummyPruningMethod, SpecPruneMethod  # noqa: E402
from gr00t.efficient.hooks.backbone_token_hook import (  # noqa: E402
    BackboneVisualTokenHook,
    patch_text_model,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig  # noqa: E402
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel  # noqa: E402


SEQ_LEN = 12
HIDDEN = 32
VISUAL_POSITIONS = (2, 3, 4, 5, 6, 7)


def _tiny_text_model() -> Qwen3VLTextModel:
    torch.manual_seed(0)
    config = Qwen3VLTextConfig(
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=64,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
    )
    config._attn_implementation = "sdpa"
    return Qwen3VLTextModel(config).eval()


def _inputs() -> dict:
    torch.manual_seed(1)
    inputs_embeds = torch.randn(1, SEQ_LEN, HIDDEN)
    visual_pos_masks = torch.zeros(1, SEQ_LEN, dtype=torch.bool)
    visual_pos_masks[:, list(VISUAL_POSITIONS)] = True
    return {"inputs_embeds": inputs_embeds, "visual_pos_masks": visual_pos_masks}


def test_disabled_hook_matches_original_forward():
    model = _tiny_text_model()
    inputs = _inputs()
    with torch.no_grad():
        expected = model(**inputs).last_hidden_state

    hook = BackboneVisualTokenHook(
        method=DummyPruningMethod(keep_ratio=0.5, mode="first"), enabled=False, prune_layer=1
    )
    patch_text_model(model, hook)
    with torch.no_grad():
        actual = model(**inputs).last_hidden_state

    torch.testing.assert_close(actual, expected)


def test_keep_ratio_one_is_noop_and_matches_original():
    model = _tiny_text_model()
    inputs = _inputs()
    with torch.no_grad():
        expected = model(**inputs).last_hidden_state

    hook = BackboneVisualTokenHook(
        method=DummyPruningMethod(keep_ratio=1.0, mode="first"), enabled=True, prune_layer=1
    )
    patch_text_model(model, hook)
    with torch.no_grad():
        actual = model(**inputs).last_hidden_state

    torch.testing.assert_close(actual, expected)
    assert hook.last_full_indices is None


def test_pruning_shortens_sequence_and_keeps_non_visual_tokens():
    model = _tiny_text_model()
    inputs = _inputs()
    hook = BackboneVisualTokenHook(
        method=DummyPruningMethod(keep_ratio=0.5, mode="first"), enabled=True, prune_layer=1
    )
    patch_text_model(model, hook)
    with torch.no_grad():
        output = model(**inputs).last_hidden_state

    # 6 visual tokens -> keep ceil(6 * 0.5) = 3; 6 non-visual tokens always survive.
    assert output.shape == (1, 9, HIDDEN)
    keep = hook.last_full_indices.tolist()
    non_visual = [i for i in range(SEQ_LEN) if i not in VISUAL_POSITIONS]
    assert set(non_visual).issubset(set(keep))
    assert hook.last_metadata["hook_scope"] == "backbone_layer"
    assert hook.last_metadata["full_token_count_before"] == SEQ_LEN
    assert hook.last_metadata["full_token_count_after"] == 9


def test_specprune_prunes_through_backbone_hook():
    # SpecPruneVLA previously never set `last_selected_indices` (it only tracked
    # `cached_indices` for its own reuse bookkeeping), so this hook's
    # `_selected_indices()` lookup always came up empty and silently no-opped --
    # this must actually shorten the sequence like DummyPruningMethod does above.
    model = _tiny_text_model()
    inputs = _inputs()
    hook = BackboneVisualTokenHook(
        method=SpecPruneMethod(keep_ratio=0.5, score_mode="norm", reuse_steps=2),
        enabled=True,
        prune_layer=1,
    )
    patch_text_model(model, hook)
    with torch.no_grad():
        output = model(**inputs).last_hidden_state

    assert output.shape == (1, 9, HIDDEN)
    assert hook.last_metadata["hook_scope"] == "backbone_layer"
    assert "hook_error" not in hook.last_metadata


def test_batch_size_two_is_safe_noop():
    model = _tiny_text_model()
    inputs = _inputs()
    inputs = {
        "inputs_embeds": inputs["inputs_embeds"].expand(2, -1, -1).contiguous(),
        "visual_pos_masks": inputs["visual_pos_masks"].expand(2, -1).contiguous(),
    }
    hook = BackboneVisualTokenHook(
        method=DummyPruningMethod(keep_ratio=0.5, mode="first"), enabled=True, prune_layer=1
    )
    patch_text_model(model, hook)
    with torch.no_grad():
        output = model(**inputs).last_hidden_state

    assert output.shape == (2, SEQ_LEN, HIDDEN)
    assert hook.last_metadata["hook_error"] == "backbone_prune_requires_batch_size_1"


def test_prune_layer_clamped_after_deepstack():
    hook = BackboneVisualTokenHook(
        method=DummyPruningMethod(keep_ratio=0.5), enabled=True, prune_layer=0
    )
    assert hook.resolve_prune_layer(num_deepstack_layers=3, num_layers=16) == 2
    assert hook.resolve_prune_layer(num_deepstack_layers=0, num_layers=16) == 0
    assert hook.resolve_prune_layer(num_deepstack_layers=0, num_layers=0) == -1

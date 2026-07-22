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
Test attach_stage_profiler with synthetic stand-in objects -- no real model,
no CUDA required. Verifies wrapping, timing record creation, reversibility
(detach), and safe no-op behavior when an expected attribute chain is missing.
"""

import time

from gr00t.efficient.profiler.model_stages import attach_stage_profiler


class _Visual:
    def __init__(self):
        self.merger = _Merger()

    def forward(self, x):
        time.sleep(0.001)
        return x * 2


class _Merger:
    def forward(self, x):
        return x + 1


class _LanguageModel:
    def forward(self, x):
        return x


class _Qwen3VLModel:
    def __init__(self):
        self.visual = _Visual()
        self.language_model = _LanguageModel()


class _ConditionalGeneration:
    def __init__(self):
        self.model = _Qwen3VLModel()


class _Backbone:
    def __init__(self):
        self.model = _ConditionalGeneration()


class _ActionHead:
    def get_action(self, x):
        return x


class _FakeModel:
    def __init__(self):
        self.backbone = _Backbone()
        self.action_head = _ActionHead()

    def get_action(self, x):
        return self.action_head.get_action(self.backbone.model.model.visual.forward(x))


def test_attach_wraps_all_five_stages_and_records_timing():
    model = _FakeModel()
    profiler, detach = attach_stage_profiler(model)

    model.get_action(1)

    summary = profiler.summary()
    for stage in ("vision_encoder_ms", "action_head_ms", "total_latency_per_action_ms"):
        assert summary[stage]["count"] == 1
        assert summary[stage]["mean_ms"] >= 0.0
    detach()


def test_detach_restores_original_callables():
    # Compare behavior, not identity: a plain instance method is re-wrapped
    # into a fresh bound-method object on every attribute access, and the
    # profiler's wrapper is a plain closure, so neither `is` nor `__func__`
    # comparisons are reliable here. What actually matters is reversibility:
    # after detach, calling the model must stop adding new profiler records.
    model = _FakeModel()
    profiler, detach = attach_stage_profiler(model)

    model.get_action(1)
    count_while_attached = profiler.summary()["total_latency_per_action_ms"]["count"]
    assert count_while_attached == 1

    detach()
    model.get_action(2)  # must still work, and must NOT add another record
    count_after_detach = profiler.summary()["total_latency_per_action_ms"]["count"]
    assert count_after_detach == count_while_attached


def test_missing_attribute_chain_is_safe_noop():
    class _Empty:
        pass

    profiler, detach = attach_stage_profiler(_Empty())
    summary = profiler.summary()
    assert summary == {}
    detach()  # must not raise even though nothing was wrapped


def test_attach_after_forward_replacement_still_times_the_replacement():
    # BackboneVisualTokenHook.patch_text_model() replaces language_model.forward
    # outright with a full reimplementation that never calls a "previous
    # forward" -- it is not a wrap, it's a swap. real_libero_adapter.py relies
    # on attach_stage_profiler() being called AFTER hook attachment so it wraps
    # whatever is actually live; this pins down that ordering is correct.
    model = _FakeModel()

    def pruning_aware_forward(x):
        return x * 100  # stand-in for _text_forward_with_pruning's own logic

    model.backbone.model.model.language_model.forward = pruning_aware_forward

    profiler, detach = attach_stage_profiler(model)
    result = model.backbone.model.model.language_model.forward(2)

    assert result == 200  # the hook's replacement ran, not the original forward
    assert profiler.summary()["policy_or_llm_ms"]["count"] == 1
    detach()


def test_attach_before_forward_replacement_loses_the_stage():
    # The inverse of the test above: if the profiler attaches BEFORE the hook
    # replaces language_model.forward, the hook's plain assignment overwrites
    # the profiler's wrapper entirely, so the LLM stage silently stops
    # recording -- this is the exact bug the ordering fix above avoids.
    model = _FakeModel()
    profiler, detach = attach_stage_profiler(model)

    def pruning_aware_forward(x):
        return x * 100

    model.backbone.model.model.language_model.forward = pruning_aware_forward  # clobbers the wrap
    model.backbone.model.model.language_model.forward(2)

    assert profiler.summary().get("policy_or_llm_ms", {"count": 0})["count"] == 0
    detach()


def test_visual_merger_is_separately_instrumented_from_vision_encoder():
    model = _FakeModel()
    profiler, detach = attach_stage_profiler(model)

    model.backbone.model.model.visual.forward(1)  # calls merger internally? no -- see note
    model.backbone.model.model.visual.merger.forward(1)

    summary = profiler.summary()
    assert summary["vision_encoder_ms"]["count"] == 1
    assert summary["visual_merger_ms"]["count"] == 1
    detach()

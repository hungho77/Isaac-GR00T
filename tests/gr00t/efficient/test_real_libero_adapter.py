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
Test RealLiberoAdapter metric helpers that must not require a model,
simulator, or GPU: the action-L2 trace guard, visual metadata aggregation,
and per-episode temporal-state isolation.
"""

from argparse import Namespace
from unittest.mock import patch

from gr00t.efficient.benchmark.real_libero_adapter import RealLiberoAdapter
from gr00t.efficient.benchmark.real_metrics import save_action_trace
import pytest


def _adapter(tmp_path, method_name: str = "vlapruner") -> RealLiberoAdapter:
    args = Namespace(method=method_name, real_output_dir=str(tmp_path))
    return RealLiberoAdapter(args=args)


def test_action_l2_returns_zero_when_current_trace_missing(tmp_path):
    adapter = _adapter(tmp_path)
    task = "libero_sim/debug_task"
    baseline_trace_path = adapter._trace_path("baseline", task, 0)
    save_action_trace(baseline_trace_path, [[0.0, 0.0], [1.0, 1.0]])
    trace_path = adapter._trace_path("vlapruner", task, 0)

    assert not trace_path.exists()
    assert adapter._compute_action_l2(trace_path, baseline_trace_path) == 0.0


def test_action_l2_returns_zero_for_baseline_method(tmp_path):
    adapter = _adapter(tmp_path, method_name="baseline")
    task = "libero_sim/debug_task"
    trace_path = adapter._trace_path("baseline", task, 0)
    save_action_trace(trace_path, [[0.0, 0.0]])

    assert adapter._compute_action_l2(trace_path, trace_path) == 0.0


def test_action_l2_computed_when_both_traces_exist(tmp_path):
    adapter = _adapter(tmp_path)
    task = "libero_sim/debug_task"
    baseline_trace_path = adapter._trace_path("baseline", task, 0)
    trace_path = adapter._trace_path("vlapruner", task, 0)
    save_action_trace(baseline_trace_path, [[0.0, 0.0]])
    save_action_trace(trace_path, [[3.0, 4.0]])

    assert adapter._compute_action_l2(trace_path, baseline_trace_path) == 5.0


def test_aggregate_visual_metadata_averages_per_step_counts():
    class _Hook:
        last_metadata = {"method": "vlapruner", "original_tokens": 240, "kept_tokens": 180}

    metadata = RealLiberoAdapter._aggregate_visual_metadata(
        _Hook(), [(240, 180), (260, 200), (None, None)]
    )

    assert metadata["original_tokens"] == 250
    assert metadata["kept_tokens"] == 190
    assert metadata["token_count_samples"] == 3


def test_aggregate_visual_metadata_without_hook():
    assert RealLiberoAdapter._aggregate_visual_metadata(None, []) is None


class _FakeMethod:
    """Stand-in for a stateful pruning method (e.g. VLAPruner) that tracks reset() calls."""

    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


def test_run_episodes_one_at_a_time_resets_method_between_every_episode():
    # Regression test: run_rollout_gymnasium_policy calls policy.reset() exactly
    # once, before its OWN internal multi-episode loop begins (rollout_policy.py
    # never resets between episodes -- see its comment "we don't properly handle
    # policy reset... policy are stateless"). VLA-Pruner's temporal-momentum
    # score is NOT stateless, so batching N episodes through one call let
    # episode k's last score bleed into episode k+1's first steps. This test
    # locks in the fix: one rollout call per episode, with an explicit
    # method.reset() between them.
    pytest.importorskip("transformers")
    method = _FakeMethod()
    adapter = RealLiberoAdapter(args=Namespace(method="vlapruner"), method=method)

    seen_seeds = []

    def fake_rollout(env_name, policy, wrapper_configs, n_episodes, n_envs, seed):
        assert n_episodes == 1
        assert n_envs == 1
        seen_seeds.append(seed)
        return env_name, [True], {"episode_lengths": [10]}

    with patch("gr00t.eval.rollout_policy.run_rollout_gymnasium_policy", side_effect=fake_rollout):
        successes, episode_infos, per_episode = adapter._run_episodes_one_at_a_time(
            task="libero_sim/debug_task",
            base_policy=object(),
            visual_hook=None,
            wrapper_configs=object(),
            num_episodes=3,
            seed=42,
        )

    assert method.reset_calls == 3
    assert successes == [True, True, True]
    assert episode_infos["episode_lengths"] == [10, 10, 10]
    assert len(per_episode) == 3
    # Each episode gets a distinct, incrementing seed -- not the same initial
    # state replayed three times.
    assert seen_seeds == [42, 43, 44]


def test_run_episodes_one_at_a_time_handles_none_seed():
    pytest.importorskip("transformers")
    method = _FakeMethod()
    adapter = RealLiberoAdapter(args=Namespace(method="vlapruner"), method=method)
    seen_seeds = []

    def fake_rollout(env_name, policy, wrapper_configs, n_episodes, n_envs, seed):
        seen_seeds.append(seed)
        return env_name, [False], {}

    with patch("gr00t.eval.rollout_policy.run_rollout_gymnasium_policy", side_effect=fake_rollout):
        adapter._run_episodes_one_at_a_time(
            task="libero_sim/debug_task",
            base_policy=object(),
            visual_hook=None,
            wrapper_configs=object(),
            num_episodes=2,
            seed=None,
        )

    assert seen_seeds == [None, None]
    assert method.reset_calls == 2

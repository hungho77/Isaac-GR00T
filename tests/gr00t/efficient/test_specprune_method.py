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
SpecPruneVLA's cached-index-reuse mechanism only activates when a `timestep`
is supplied to `process_visual_tokens`, but none of the three real hooks
(after_visual_merger / backbone / action_head) ever pass one -- so without an
internal counter, `SpecPruneMethod` always recomputes and the reuse this
method exists for never fires. These tests pin down the auto-timestep
counter added to `SpecPruneMethod` to close that gap.
"""

from gr00t.efficient.benchmark.methods import SpecPruneMethod
import torch


def _tokens_with_norms(norms: list[float]) -> torch.Tensor:
    """[1, N, N] tokens where row i is a one-hot vector scaled to norms[i]."""
    n = len(norms)
    return (torch.eye(n) * torch.tensor(norms).unsqueeze(1)).unsqueeze(0)


def test_auto_timestep_increments_every_call_without_explicit_timestep():
    method = SpecPruneMethod(keep_ratio=0.5, score_mode="norm", reuse_steps=2)
    tokens = _tokens_with_norms([1, 2, 3, 4])

    timesteps = []
    for _ in range(4):
        _, metadata = method.process_visual_tokens(tokens)
        timesteps.append(metadata["timestep"])

    assert timesteps == [0, 1, 2, 3]


def test_reuse_activates_on_off_boundary_steps_with_auto_timestep():
    method = SpecPruneMethod(keep_ratio=0.5, score_mode="norm", reuse_steps=2)

    # timestep=0: 0 % reuse_steps == 0 -> fresh compute (never reuse on the first call).
    _, metadata0 = method.process_visual_tokens(_tokens_with_norms([1, 2, 3, 4]))
    assert metadata0["reused_indices"] is False

    # timestep=1: reuse the indices cached at timestep 0.
    _, metadata1 = method.process_visual_tokens(_tokens_with_norms([1, 2, 3, 4]))
    assert metadata1["reused_indices"] is True

    # timestep=2: 2 % reuse_steps == 0 -> recompute again.
    _, metadata2 = method.process_visual_tokens(_tokens_with_norms([1, 2, 3, 4]))
    assert metadata2["reused_indices"] is False

    # timestep=3: reuse again.
    _, metadata3 = method.process_visual_tokens(_tokens_with_norms([1, 2, 3, 4]))
    assert metadata3["reused_indices"] is True


def test_reused_step_keeps_cached_indices_even_when_input_would_score_differently():
    # temporal_momentum=0.0: isolate index reuse from SpecPrune's separate EMA
    # score-smoothing (which persists prev_score across calls, including reuse
    # steps, and would otherwise blend stale scores into the "fresh" recompute).
    method = SpecPruneMethod(
        keep_ratio=0.5, score_mode="norm", reuse_steps=2, temporal_momentum=0.0
    )

    # timestep=0 (fresh): top-2 by norm are indices [2, 3].
    method.process_visual_tokens(_tokens_with_norms([1, 2, 3, 4]))
    selected_call0 = list(method.pruner.cached_indices[0].tolist())
    assert selected_call0 == [2, 3]

    # timestep=1 (reuse): input norms are reversed, so a fresh computation would
    # pick indices [0, 1] instead -- but the cached [2, 3] must be reused verbatim.
    method.process_visual_tokens(_tokens_with_norms([4, 3, 2, 1]))
    selected_call1 = list(method.pruner.cached_indices[0].tolist())
    assert selected_call1 == [2, 3]

    # timestep=2 (recompute boundary): with the norms still reversed, the fresh
    # selection must actually change to prove this call really rescored.
    method.process_visual_tokens(_tokens_with_norms([4, 3, 2, 1]))
    selected_call2 = list(method.pruner.cached_indices[0].tolist())
    assert selected_call2 == [0, 1]


def test_reset_restarts_the_timestep_counter():
    method = SpecPruneMethod(keep_ratio=0.5, score_mode="norm", reuse_steps=2)
    tokens = _tokens_with_norms([1, 2, 3, 4])

    method.process_visual_tokens(tokens)  # timestep=0
    method.process_visual_tokens(tokens)  # timestep=1

    method.reset()

    _, metadata = method.process_visual_tokens(tokens)
    assert metadata["timestep"] == 0
    # cached_indices was also cleared by reset(), so this must be a fresh compute.
    assert metadata["reused_indices"] is False


def test_explicit_timestep_kwarg_overrides_internal_counter():
    method = SpecPruneMethod(keep_ratio=0.5, score_mode="norm", reuse_steps=2)
    tokens = _tokens_with_norms([1, 2, 3, 4])

    _, metadata = method.process_visual_tokens(tokens, timestep=99)
    assert metadata["timestep"] == 99

    # The internal counter still advances by exactly one per call regardless of
    # whether that call's timestep came from a kwarg or the auto counter.
    _, metadata_next = method.process_visual_tokens(tokens)
    assert metadata_next["timestep"] == 1

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
ADPMethod.process_visual_tokens() used to create a fresh DummyVisualTokenPruner
every call and discard it without ever exposing its selection -- every hook's
_selected_indices() lookup (method.pruner / method.vlapruner / a direct
method.last_selected_indices fallback) came up empty, so ADP silently no-opped
(hook_error="missing_selected_indices") in every real run to date, regardless
of dynamic_keep_ratio. This was found via a real n=100 checkpoint run where
ADP's mean_token_reduction_ratio came back 0.000 -- these tests pin down the fix.
"""

from gr00t.efficient.benchmark.methods import ADPMethod
import torch


def _tokens(batch=1, n=8, dim=4):
    torch.manual_seed(0)
    return torch.randn(batch, n, dim)


def test_process_visual_tokens_exposes_last_selected_indices():
    method = ADPMethod(default_keep_ratio=0.5, mode="first")

    _, metadata = method.process_visual_tokens(_tokens())

    assert method.last_selected_indices is not None
    assert metadata["kept_tokens"] == 4  # ceil(8 * 0.5)
    assert list(method.last_selected_indices) == [0, 1, 2, 3]  # mode="first"


def test_reset_clears_last_selected_indices():
    method = ADPMethod(default_keep_ratio=0.5, mode="first")
    method.process_visual_tokens(_tokens())
    assert method.last_selected_indices is not None

    method.reset()

    assert method.last_selected_indices is None


def test_last_selected_indices_tracks_the_dynamic_keep_ratio():
    # With no action_state, the scheduler falls back to default_keep_ratio --
    # this is itself a separate, real gap (no hook threads action_state today,
    # so ADP's contact/move/idle switching never activates in practice) but is
    # not what this test is pinning down; it just confirms the selection
    # length matches whatever keep ratio the scheduler actually returned.
    method = ADPMethod(default_keep_ratio=0.25, mode="first")

    _, metadata = method.process_visual_tokens(_tokens(n=8))

    assert metadata["dynamic_keep_ratio"] == 0.25
    assert len(method.last_selected_indices) == 2  # ceil(8 * 0.25)

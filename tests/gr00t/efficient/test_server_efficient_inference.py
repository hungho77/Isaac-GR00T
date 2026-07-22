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
run_gr00t_server.py's ServerConfig originally only forwarded keep_ratio and
score_mode to build_method(), so selecting 'adp'/'adp_vlapruner' via
--efficient-method silently used their hardcoded default scheduler
hyperparameters (default_keep_ratio, contact/move/idle_keep_ratio, alpha,
beta) regardless of any --efficient-* flags passed. These tests pin down
that every ServerConfig field actually reaches build_method().
"""

from unittest.mock import MagicMock, patch

import pytest


pytest.importorskip("transformers")

from gr00t.eval.run_gr00t_server import ServerConfig, _attach_efficient_inference  # noqa: E402


class _FakeActionHead:
    def process_backbone_output(self, x):
        return x


class _FakeModel:
    def __init__(self):
        self.action_head = _FakeActionHead()


def _config(**overrides) -> ServerConfig:
    config = ServerConfig(model_path="unused", efficient_method="adp_vlapruner")
    config.efficient_prune_stage = "action_head"  # avoid needing the backbone attribute chain
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_all_efficient_config_fields_reach_build_method():
    config = _config(
        efficient_keep_ratio=0.6,
        efficient_score_mode="mean_abs",
        efficient_reuse_steps=4,
        efficient_temporal_momentum=0.9,
        efficient_alpha=0.3,
        efficient_beta=0.7,
        efficient_default_keep_ratio=0.4,
        efficient_contact_keep_ratio=0.8,
        efficient_move_keep_ratio=0.5,
        efficient_idle_keep_ratio=0.3,
        efficient_action_delta_threshold=0.1,
    )
    fake_method = MagicMock()

    with patch(
        "gr00t.efficient.benchmark.registry.build_method", return_value=fake_method
    ) as build_method:
        _attach_efficient_inference(_FakeModel(), config)

    build_method.assert_called_once_with(
        "adp_vlapruner",
        keep_ratio=0.6,
        score_mode="mean_abs",
        reuse_steps=4,
        temporal_momentum=0.9,
        alpha=0.3,
        beta=0.7,
        default_keep_ratio=0.4,
        contact_keep_ratio=0.8,
        move_keep_ratio=0.5,
        idle_keep_ratio=0.3,
        action_delta_threshold=0.1,
    )


def test_attach_returns_the_built_method_for_reset_wiring():
    config = _config()
    fake_method = MagicMock()

    with patch("gr00t.efficient.benchmark.registry.build_method", return_value=fake_method):
        returned = _attach_efficient_inference(_FakeModel(), config)

    assert returned is fake_method

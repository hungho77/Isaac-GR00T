# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only regression tests for the LIBERO Gymnasium observation contract."""

from __future__ import annotations

import importlib
import math
import sys
import types
import warnings

import numpy as np


def _install_libero_import_stubs(monkeypatch):
    class _OffScreenRenderEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    benchmark = types.SimpleNamespace(get_benchmark_dict=lambda: {})
    libero = types.ModuleType("libero")
    libero.__path__ = []
    libero_libero = types.ModuleType("libero.libero")
    libero_libero.__path__ = []
    libero_libero.benchmark = benchmark
    libero_envs = types.ModuleType("libero.libero.envs")
    libero_envs.OffScreenRenderEnv = _OffScreenRenderEnv
    libero_utils = types.ModuleType("libero.libero.utils")
    libero_utils.get_libero_path = lambda name: f"/{name}"

    libero.libero = libero_libero
    for name, module in {
        "libero": libero,
        "libero.libero": libero_libero,
        "libero.libero.envs": libero_envs,
        "libero.libero.utils": libero_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def _import_libero_env(monkeypatch):
    _install_libero_import_stubs(monkeypatch)
    module_name = "gr00t.eval.sim.LIBERO.libero_env"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    return importlib.import_module(module_name)


def test_processed_observation_matches_declared_space(monkeypatch):
    module = _import_libero_env(monkeypatch)
    env = module.LiberoEnv("task.bddl", "put the object in the basket")
    # A negative quaternion scalar produces a valid non-canonical angle above
    # pi in the branch's existing conversion and must remain inside the space.
    angle = 4.0
    quat = np.array([math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)], dtype=np.float64)
    raw = {
        "robot0_eef_pos": np.array([1.25, -1.5, 0.75], dtype=np.float64),
        "robot0_eef_quat": quat.copy(),
        "robot0_gripper_qpos": np.array([-1.2, 1.2], dtype=np.float64),
        "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        observation = env._process_observation(raw)
        assert env.observation_space.contains(observation)

    assert caught == []
    assert np.array_equal(raw["robot0_eef_quat"], quat), "processing must not mutate raw obs"
    for key, value in observation.items():
        if key.startswith("state."):
            assert isinstance(value, np.ndarray)
            assert value.dtype == np.float32
    assert observation["video.image"].flags.c_contiguous
    assert observation["video.wrist_image"].flags.c_contiguous


def test_action_space_explicitly_uses_float32(monkeypatch):
    module = _import_libero_env(monkeypatch)
    env = module.LiberoEnv("task.bddl", "task")

    assert all(space.dtype == np.float32 for space in env.action_space.spaces.values())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = REPO_ROOT / "examples/LIBERO/quantization/modal_phase1_runner.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("modal_phase1_runner_for_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_all_suite_configs_use_twenty_evaluation_trials() -> None:
    runner = _load_runner()

    assert runner.FULL_ROLLOUT_EPISODES_PER_TASK == 20
    for suite in runner.SUITE_TASKS:
        config_path = REPO_ROOT / f"examples/LIBERO/quantization/configs/{suite}.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["evaluation_trials_per_task"] == runner.FULL_ROLLOUT_EPISODES_PER_TASK
        assert config["calibration_trajectories"] == 10


def test_full_rollout_dispatches_twenty_paired_episodes(monkeypatch, tmp_path) -> None:
    runner = _load_runner()
    pack_path = tmp_path / "pack.pt"
    pack_path.touch()
    for mode in runner.ROLLOUT_MODES:
        smoke_path = tmp_path / "smoke" / mode / "task_00.json"
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        smoke_path.write_text("{}\n", encoding="utf-8")
    cfg = SimpleNamespace(
        suite="object",
        suite_root=tmp_path,
        pack_path=pack_path,
        smoke_task_index=0,
        evaluation_seed_base=10_000,
        n_envs=1,
    )
    calls = []

    @contextmanager
    def policy_server(_cfg, *, mode):
        yield

    def run_shard(_cfg, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(runner, "_require_manifest", lambda _cfg: {})
    monkeypatch.setattr(runner, "_validate_l4", lambda _cfg: {})
    monkeypatch.setattr(runner, "_policy_server", policy_server)
    monkeypatch.setattr(runner, "_run_shard", run_shard)
    monkeypatch.setattr(runner, "_write_metrics", lambda _cfg: {})
    monkeypatch.setattr(runner, "_phase_marker", lambda *_args, **_kwargs: None)

    runner._run_rollout_phase(cfg, smoke=False)

    assert len(calls) == len(runner.ROLLOUT_MODES) * runner.EXPECTED_TASKS
    assert {call["mode"] for call in calls} == set(runner.ROLLOUT_MODES)
    assert {call["n_episodes"] for call in calls} == {runner.FULL_ROLLOUT_EPISODES_PER_TASK}
    for task_index in range(runner.EXPECTED_TASKS):
        paired = [call for call in calls if call["task_index"] == task_index]
        assert len(paired) == len(runner.ROLLOUT_MODES)
        assert len({call["seed"] for call in paired}) == 1

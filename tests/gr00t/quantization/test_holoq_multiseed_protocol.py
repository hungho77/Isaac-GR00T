# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "LIBERO"
    / "quantization"
    / "modal_phase1_runner.py"
)
SPEC = importlib.util.spec_from_file_location("holoq_modal_phase1_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def _config(tmp_path: Path, **overrides):
    values = {
        "suite": "long",
        "work_phase": "status",
        "repo_path": tmp_path / "repo",
        "artifact_root": tmp_path / "artifacts",
        "checkpoint_repo": "nvidia/GR00T-N1.7-LIBERO",
        "checkpoint_ref": "revision",
        "server_port": 5555,
        "calibration_seed": 0,
        "evaluation_seeds": (10000, 20000, 30000),
        "n_envs": 1,
        "n_action_steps": 8,
        "max_episode_steps": 720,
        "calibration_topk": 512,
        "smoke_task_index": 0,
        "record_videos": False,
        "package_include_videos": False,
        "require_l4": True,
        "holoq_backend": "native",
        "holoq_scopes": "llm,dit,vit",
        "holoq_include_vit_mergers": True,
        "holoq_include_vit_patch_embed": True,
        "benchmark_warmup": 5,
        "benchmark_iterations": 20,
    }
    values.update(overrides)
    return RUNNER.Config(**values)


def test_native_long_protocol_requires_three_distinct_seeds(tmp_path: Path) -> None:
    RUNNER._validate_native_multiseed_protocol(_config(tmp_path))
    with pytest.raises(RuntimeError, match="three distinct"):
        RUNNER._validate_native_multiseed_protocol(
            _config(tmp_path, evaluation_seeds=(10000, 10000, 30000))
        )


def test_native_long_protocol_requires_complete_transformer_scope(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="exact scopes"):
        RUNNER._validate_native_multiseed_protocol(_config(tmp_path, holoq_scopes="llm,dit"))
    with pytest.raises(RuntimeError, match="patch embedding"):
        RUNNER._validate_native_multiseed_protocol(
            _config(tmp_path, holoq_include_vit_patch_embed=False)
        )


def test_three_seed_confidence_summary_is_explicit() -> None:
    summary = RUNNER._mean_ci95([0.8, 0.9, 1.0])
    assert summary["mean"] == pytest.approx(0.9)
    assert summary["std"] == pytest.approx(0.1)
    assert summary["ci95_low"] < summary["mean"] < summary["ci95_high"]


def test_rpc_latency_discards_only_cold_samples() -> None:
    summary = RUNNER._latency_seconds_summary([0.9, 0.8, 0.7, 0.1, 0.2], discard_first=3)
    assert summary["sample_count"] == 2
    assert summary["mean"] == pytest.approx(150.0)


def test_w4a4_server_requires_compact_native_checkpoint(tmp_path: Path, monkeypatch) -> None:
    cfg = _config(tmp_path)
    cfg.repo_path.mkdir()
    cfg.pack_path.parent.mkdir(parents=True)
    cfg.pack_path.write_bytes(b"pack")
    monkeypatch.setattr(RUNNER, "_port_is_open", lambda *_args: False)
    monkeypatch.setattr(RUNNER, "_require_manifest", lambda _cfg: {})

    with pytest.raises(RuntimeError, match="never falls back"):
        with RUNNER._policy_server(cfg, mode="w4a4"):
            pass

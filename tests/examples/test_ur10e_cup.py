# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "UR10eCup"


def _load_prepare_module():
    path = EXAMPLE_ROOT / "prepare_dataset.py"
    spec = importlib.util.spec_from_file_location("ur10e_prepare_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_info(module) -> dict:
    return {
        "codebase_version": "v3.0",
        "robot_type": "ur10e",
        "total_episodes": 81,
        "total_frames": 49_779,
        "total_tasks": 1,
        "fps": 20,
        "features": {
            "observation.images.side": {
                "dtype": "video",
                "shape": [480, 640, 3],
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": [480, 640, 3],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [7],
                "names": {"motors": module.EXPECTED_MOTOR_NAMES},
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": {"motors": module.EXPECTED_MOTOR_NAMES},
            },
        },
    }


def _write_synthetic_dataset(root: Path, module, *, break_action_shift: bool = False) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True)
    info = _source_info(module)
    info.update(
        {
            "codebase_version": "v2.1",
            "total_episodes": 2,
            "total_frames": 8,
        }
    )
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": module.EXPECTED_TASK}) + "\n",
        encoding="utf-8",
    )
    (meta / "modality.json").write_text(
        (EXAMPLE_ROOT / "modality.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    for episode_index in range(2):
        states = np.zeros((4, 7), dtype=np.float32)
        states[:, 0] = np.arange(4, dtype=np.float32) * 0.01 + episode_index
        states[:, 4] = 1.5708
        states[:, 6] = [1, 1, 0, 0]
        actions = np.concatenate([states[1:], states[-1:]], axis=0)
        if break_action_shift and episode_index == 1:
            actions[0, 0] += 0.1
        table = pa.table(
            {
                "observation.state": states.tolist(),
                "action": actions.tolist(),
                "timestamp": np.arange(4, dtype=np.float32) / 20,
                "frame_index": np.arange(4, dtype=np.int64),
                "episode_index": np.full(4, episode_index, dtype=np.int64),
            }
        )
        parquet_path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet_path)


def test_source_schema_contract_accepts_pinned_dataset() -> None:
    module = _load_prepare_module()
    module.validate_source_info(_source_info(module))


def test_source_schema_contract_rejects_motor_reordering() -> None:
    module = _load_prepare_module()
    info = _source_info(module)
    info["features"]["action"]["names"]["motors"] = list(reversed(module.EXPECTED_MOTOR_NAMES))
    with pytest.raises(ValueError, match="motor order"):
        module.validate_source_info(info)


def test_converted_dataset_validates_absolute_action_shift(tmp_path: Path) -> None:
    module = _load_prepare_module()
    _write_synthetic_dataset(tmp_path, module)
    report = module.validate_converted_dataset(
        tmp_path,
        expected_episodes=2,
        expected_frames=8,
        require_videos=False,
    )
    assert report["status"] == "ok"
    assert report["max_action_shift_error"] == 0
    assert report["wrist_2_span_rad"] == 0


def test_converted_dataset_rejects_misaligned_actions(tmp_path: Path) -> None:
    module = _load_prepare_module()
    _write_synthetic_dataset(tmp_path, module, break_action_shift=True)
    with pytest.raises(ValueError, match="Action contract mismatch"):
        module.validate_converted_dataset(
            tmp_path,
            expected_episodes=2,
            expected_frames=8,
            require_videos=False,
        )

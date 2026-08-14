# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare and validate khanhnd61/ur10e-cup for GR00T N1.7.

This wrapper pins the public LeRobot v3 snapshot, checks the robot/action
contract, invokes the repository's v3-to-v2 converter, installs modality.json,
and validates every converted episode.  It intentionally does not delete the
converter's ``*_v30`` source backup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from huggingface_hub import snapshot_download
import numpy as np
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPO_ID = "khanhnd61/ur10e-cup"
SOURCE_REVISION = "12bbeaad3995e84cf43b99adecb0a8bc216cedbf"
EXPECTED_EPISODES = 81
EXPECTED_FRAMES = 49_779
EXPECTED_FPS = 20
EXPECTED_TASK = "pick up the cup"
EXPECTED_ROBOT_TYPE = "ur10e"
EXPECTED_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
    "gripper",
]
EXPECTED_VIDEO_KEYS = ["observation.images.side", "observation.images.wrist"]
ACTION_SHIFT_TOLERANCE = 5e-4
GRIPPER_BINARY_TOLERANCE = 1e-3


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_info(info: dict[str, Any]) -> None:
    """Reject a source snapshot whose hardware or tensor contract changed."""

    expected_scalars = {
        "codebase_version": "v3.0",
        "robot_type": EXPECTED_ROBOT_TYPE,
        "total_episodes": EXPECTED_EPISODES,
        "total_frames": EXPECTED_FRAMES,
        "total_tasks": 1,
        "fps": EXPECTED_FPS,
    }
    for key, expected in expected_scalars.items():
        actual = info.get(key)
        if actual != expected:
            raise ValueError(f"Source metadata changed: {key}={actual!r}, expected {expected!r}")

    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("Source metadata has no features object")

    for key in EXPECTED_VIDEO_KEYS:
        feature = features.get(key)
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            raise ValueError(f"Missing video feature {key!r}")
        if feature.get("shape") != [480, 640, 3]:
            raise ValueError(f"Unexpected shape for {key}: {feature.get('shape')!r}")

    for key in ["observation.state", "action"]:
        feature = features.get(key)
        if not isinstance(feature, dict):
            raise ValueError(f"Missing numeric feature {key!r}")
        if feature.get("dtype") != "float32" or feature.get("shape") != [7]:
            raise ValueError(f"Unexpected {key} contract: {feature!r}")
        names = feature.get("names")
        if not isinstance(names, dict) or names.get("motors") != EXPECTED_MOTOR_NAMES:
            raise ValueError(f"Unexpected motor order for {key}: {names!r}")


def _load_task(dataset_path: Path) -> str:
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    lines = [line for line in tasks_path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 1:
        raise ValueError(f"Expected exactly one task in {tasks_path}, found {len(lines)}")
    row = json.loads(lines[0])
    if row.get("task_index") != 0 or row.get("task") != EXPECTED_TASK:
        raise ValueError(f"Unexpected task row: {row!r}")
    return str(row["task"])


def validate_converted_dataset(
    dataset_path: str | Path,
    *,
    expected_episodes: int = EXPECTED_EPISODES,
    expected_frames: int = EXPECTED_FRAMES,
    require_videos: bool = True,
) -> dict[str, Any]:
    """Validate the converted GR00T LeRobot v2 dataset and action alignment."""

    dataset_path = Path(dataset_path)
    info_path = dataset_path / "meta" / "info.json"
    modality_path = dataset_path / "meta" / "modality.json"
    expected_modality_path = Path(__file__).with_name("modality.json")
    info = _read_json(info_path)

    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"Expected converted v2.1 dataset, got {info.get('codebase_version')!r}")
    if info.get("robot_type") != EXPECTED_ROBOT_TYPE:
        raise ValueError(f"Unexpected robot_type: {info.get('robot_type')!r}")
    if info.get("fps") != EXPECTED_FPS:
        raise ValueError(f"Unexpected fps: {info.get('fps')!r}")
    if info.get("total_episodes") != expected_episodes:
        raise ValueError(f"Unexpected episode count: {info.get('total_episodes')!r}")
    if info.get("total_frames") != expected_frames:
        raise ValueError(f"Unexpected frame count: {info.get('total_frames')!r}")
    if _read_json(modality_path) != _read_json(expected_modality_path):
        raise ValueError(f"Dataset modality mapping differs from {expected_modality_path}")

    task = _load_task(dataset_path)
    parquet_paths = sorted((dataset_path / "data").glob("chunk-*/episode_*.parquet"))
    if len(parquet_paths) != expected_episodes:
        raise ValueError(
            f"Expected {expected_episodes} episode parquet files, found {len(parquet_paths)}"
        )

    total_frames = 0
    max_action_shift_error = 0.0
    wrist_2_min = np.inf
    wrist_2_max = -np.inf
    for episode_index, parquet_path in enumerate(parquet_paths):
        table = pq.read_table(
            parquet_path,
            columns=[
                "observation.state",
                "action",
                "timestamp",
                "frame_index",
                "episode_index",
            ],
        )
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64).reshape(-1)
        frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64).reshape(-1)
        episode_indices = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64).reshape(-1)

        if states.ndim != 2 or states.shape[1] != 7 or actions.shape != states.shape:
            raise ValueError(
                f"Invalid state/action shapes in {parquet_path}: {states.shape}, {actions.shape}"
            )
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"Non-finite state/action values in {parquet_path}")
        if not np.array_equal(frame_indices, np.arange(len(states))):
            raise ValueError(f"Non-contiguous frame_index in {parquet_path}")
        if not np.all(episode_indices == episode_index):
            raise ValueError(f"episode_index column does not match file order in {parquet_path}")
        if len(timestamps) > 1:
            timestamp_steps = np.diff(timestamps)
            if np.any(timestamp_steps <= 0):
                raise ValueError(f"Non-monotonic timestamps in {parquet_path}")
            if not np.isclose(np.median(timestamp_steps), 1 / EXPECTED_FPS, atol=2e-3):
                raise ValueError(f"Timestamp cadence is not {EXPECTED_FPS} FPS in {parquet_path}")
            shift_error = float(np.max(np.abs(actions[:-1] - states[1:])))
            max_action_shift_error = max(max_action_shift_error, shift_error)

        gripper_values = np.concatenate([states[:, 6], actions[:, 6]])
        distance_to_binary = np.minimum(np.abs(gripper_values), np.abs(gripper_values - 1.0))
        if float(distance_to_binary.max(initial=0.0)) > GRIPPER_BINARY_TOLERANCE:
            raise ValueError(f"Non-binary gripper values in {parquet_path}")

        wrist_2_min = min(wrist_2_min, float(states[:, 4].min()))
        wrist_2_max = max(wrist_2_max, float(states[:, 4].max()))
        total_frames += len(states)

    if total_frames != expected_frames:
        raise ValueError(f"Parquet rows total {total_frames}, expected {expected_frames}")
    if max_action_shift_error > ACTION_SHIFT_TOLERANCE:
        raise ValueError(
            "Action contract mismatch: expected absolute action[t] ~= state[t+1], "
            f"max error={max_action_shift_error:.6g}"
        )

    video_counts: dict[str, int] = {}
    for key in EXPECTED_VIDEO_KEYS:
        video_paths = sorted((dataset_path / "videos").glob(f"chunk-*/{key}/episode_*.mp4"))
        video_counts[key] = len(video_paths)
        if require_videos and len(video_paths) != expected_episodes:
            raise ValueError(
                f"Expected {expected_episodes} videos for {key}, found {len(video_paths)}"
            )
        empty_videos = [str(path) for path in video_paths if path.stat().st_size == 0]
        if empty_videos:
            raise ValueError(f"Empty video files detected: {empty_videos[:3]}")

    return {
        "status": "ok",
        "dataset_path": str(dataset_path.resolve()),
        "robot_type": EXPECTED_ROBOT_TYPE,
        "task": task,
        "episodes": expected_episodes,
        "frames": total_frames,
        "fps": EXPECTED_FPS,
        "video_counts": video_counts,
        "action_contract": "absolute joint target + gripper; action[t] ~= state[t+1]",
        "max_action_shift_error": max_action_shift_error,
        "wrist_2_range_rad": [wrist_2_min, wrist_2_max],
        "wrist_2_span_rad": wrist_2_max - wrist_2_min,
        "modality_sha256": _sha256(modality_path),
        "recommended_episode_split": {
            "train": "0:65",
            "validation": "65:73",
            "test": "73:81",
            "applied_by_converter": False,
        },
        "safety_note": (
            "wrist_2 is effectively constant in this dataset; clamp that target to the observed "
            "robot state during deployment and enforce an independent hardware safety controller"
        ),
    }


def prepare_dataset(repo_id: str, root: Path, revision: str) -> tuple[Path, dict[str, Any]]:
    if repo_id != SOURCE_REPO_ID:
        raise ValueError(f"This recipe is pinned to {SOURCE_REPO_ID!r}, got {repo_id!r}")

    dataset_path = root / repo_id
    if not dataset_path.exists():
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=dataset_path,
        )

    info = _read_json(dataset_path / "meta" / "info.json")
    version = info.get("codebase_version")
    if version == "v3.0":
        validate_source_info(info)
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from scripts.lerobot_conversion.convert_v3_to_v2 import convert_dataset

        convert_dataset(repo_id=repo_id, root=root)
    elif version != "v2.1":
        raise ValueError(f"Unsupported local dataset version: {version!r}")

    modality_source = Path(__file__).with_name("modality.json")
    modality_target = dataset_path / "meta" / "modality.json"
    shutil.copy2(modality_source, modality_target)

    report = validate_converted_dataset(dataset_path)
    report.update(
        {
            "source_repo_id": repo_id,
            "source_revision": revision,
            "source_v3_backup": str(dataset_path.parent / f"{dataset_path.name}_v30"),
        }
    )
    report_path = dataset_path / "meta" / "gr00t_conversion.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return dataset_path, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=SOURCE_REPO_ID)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "examples" / "UR10eCup" / "ur10e_cup_lerobot",
        help="Parent directory; the repository namespace is appended below it.",
    )
    parser.add_argument("--revision", default=SOURCE_REVISION)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing converted dataset without downloading or converting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.root / args.repo_id
    if args.validate_only:
        report = validate_converted_dataset(dataset_path)
    else:
        dataset_path, report = prepare_dataset(args.repo_id, args.root, args.revision)
    print(json.dumps(report, indent=2))
    print(f"GR00T dataset ready: {dataset_path}")


if __name__ == "__main__":
    main()

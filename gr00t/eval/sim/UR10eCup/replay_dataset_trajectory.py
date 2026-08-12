# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay a recorded UR10e trajectory as a simulator calibration gate.

This check validates the simulator before policy success rate is measured.  A
failure means the proxy scene/controller is not aligned well enough with the
dataset; it is not evidence about model or pruning quality.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .ur10e_cup_env import UR10eCupEnv


@dataclass
class ReplayResult:
    """Serializable ground-truth replay result."""

    status: str
    episode_index: int
    episode_file: str
    steps_available: int
    steps_executed: int
    approached_cup: bool
    left_contact_seen: bool
    right_contact_seen: bool
    both_contacts_seen: bool
    lifted_seen: bool
    stable_success: bool
    hardware_safety_violation: bool
    minimum_tool_cup_distance_m: float
    maximum_cup_height_m: float
    final_info: dict[str, Any]


def _episode_file(dataset_path: Path, episode_index: int) -> Path:
    matches = sorted(dataset_path.glob(f"data/chunk-*/episode_{episode_index:06d}.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one parquet file for episode {episode_index}, found {len(matches)} "
            f"under {dataset_path}"
        )
    return matches[0]


def load_episode(dataset_path: Path, episode_index: int) -> tuple[Path, np.ndarray, np.ndarray]:
    """Load state/action arrays and enforce the seven-dimensional absolute contract."""

    episode_file = _episode_file(dataset_path, episode_index)
    table = pq.read_table(episode_file, columns=["observation.state", "action"])
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    shape_is_valid = (
        states.ndim == 2
        and actions.ndim == 2
        and states.shape[1:] == (7,)
        and actions.shape[1:] == (7,)
    )
    if not shape_is_valid:
        raise ValueError(
            "Episode must contain (T, 7) state/action arrays; "
            f"got {states.shape} and {actions.shape}"
        )
    if states.shape[0] != actions.shape[0] or states.shape[0] < 2:
        raise ValueError(
            f"State/action row counts are invalid: {states.shape[0]} and {actions.shape[0]}"
        )
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Episode contains non-finite state/action values")
    return episode_file, states, actions


def _open_video_writer(path: Path | None, frame: np.ndarray, fps: int):
    if path is None:
        return None
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Video output requires opencv-python-headless") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(frame.shape[1]), int(frame.shape[0])),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create replay video: {path}")
    return writer


def _video_frame(observation: dict[str, Any]) -> np.ndarray:
    return np.concatenate((observation["video.side"], observation["video.wrist"]), axis=1)


def replay_episode(
    dataset_path: Path,
    episode_index: int,
    *,
    config_path: str | None = None,
    max_steps: int | None = None,
    video_path: Path | None = None,
) -> ReplayResult:
    """Replay one demonstration and return the simulator acceptance contract."""

    episode_file, states, actions = load_episode(dataset_path, episode_index)
    limit = len(actions) if max_steps is None else min(len(actions), max_steps)
    if limit < 1:
        raise ValueError("max_steps must allow at least one action")

    env = UR10eCupEnv(config_path=config_path)
    writer = None
    try:
        observation, info = env.reset(
            seed=episode_index,
            options={
                "arm_qpos": states[0, :6],
                "gripper_open_fraction": states[0, 6],
            },
        )
        frame = _video_frame(observation)
        writer = _open_video_writer(video_path, frame, env.metadata["render_fps"])
        if writer is not None:
            import cv2

            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        approached = bool(info["approached_cup"])
        left_contact = bool(info["left_finger_contact"])
        right_contact = bool(info["right_finger_contact"])
        both_contacts = left_contact and right_contact
        lifted = bool(info["lifted"])
        success = bool(info["success"])
        hardware_safety = bool(info["hardware_safety_violation"])
        minimum_distance = float(info["tool_cup_distance_m"])
        maximum_height = float(info["max_cup_height_m"])
        steps_executed = 0

        for action_row in actions[:limit]:
            action = {
                "action.arm_joints": action_row[:6].astype(np.float32),
                "action.gripper": np.array([action_row[6]], dtype=np.float32),
            }
            observation, _, terminated, _, info = env.step(action)
            steps_executed += 1
            approached |= bool(info["approached_cup"])
            left_contact |= bool(info["left_finger_contact"])
            right_contact |= bool(info["right_finger_contact"])
            both_contacts |= bool(info["left_finger_contact"] and info["right_finger_contact"])
            lifted |= bool(info["lifted"])
            success |= bool(info["success"])
            hardware_safety |= bool(info["hardware_safety_violation"])
            minimum_distance = min(minimum_distance, float(info["tool_cup_distance_m"]))
            maximum_height = max(maximum_height, float(info["max_cup_height_m"]))
            if writer is not None:
                import cv2

                frame = _video_frame(observation)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if terminated:
                break

        passed = approached and both_contacts and lifted and success and not hardware_safety
        return ReplayResult(
            status="passed" if passed else "failed",
            episode_index=episode_index,
            episode_file=str(episode_file),
            steps_available=len(actions),
            steps_executed=steps_executed,
            approached_cup=approached,
            left_contact_seen=left_contact,
            right_contact_seen=right_contact,
            both_contacts_seen=both_contacts,
            lifted_seen=lifted,
            stable_success=success,
            hardware_safety_violation=hardware_safety,
            minimum_tool_cup_distance_m=minimum_distance,
            maximum_cup_height_m=maximum_height,
            final_info=info,
        )
    finally:
        if writer is not None:
            writer.release()
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=73)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-path")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--allow-failure",
        action="store_true",
        help="Write diagnostics but return exit code zero when the calibration gate fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / "ground_truth_replay.mp4" if args.save_video else None
    result = replay_episode(
        args.dataset_path,
        args.episode_index,
        config_path=args.config_path,
        max_steps=args.max_steps,
        video_path=video_path,
    )
    report_path = args.output_dir / "ground_truth_replay_report.json"
    report_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    print(json.dumps(asdict(result), indent=2))
    print(f"Report: {report_path}")
    if result.status != "passed" and not args.allow_failure:
        raise SystemExit(
            "Ground-truth replay calibration gate failed. Do not report policy success rate until "
            "the simulator passes approach, two-finger contact, lift, and stable-success checks."
        )


if __name__ == "__main__":
    main()

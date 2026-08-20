# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one resumable LIBERO rollout shard and atomically write structured JSON."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

from gr00t.eval.rollout_policy import run_gr00t_sim_policy
import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json_dump(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    temporary_path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--suite", required=True, choices=("object", "spatial", "goal", "long"))
    parser.add_argument("--mode", required=True, choices=("calibration", "bf16", "w4a4"))
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--n-episodes", required=True, type=int)
    parser.add_argument("--n-envs", default=1, type=int)
    parser.add_argument("--n-action-steps", default=8, type=int)
    parser.add_argument("--max-episode-steps", default=720, type=int)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", default=5555, type=int)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    args = parser.parse_args()

    if args.n_episodes <= 0:
        raise ValueError("--n-episodes must be positive")
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive")

    args.video_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    start_time = time.monotonic()
    env_name, successes, episode_infos = run_gr00t_sim_policy(
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        model_path="",
        policy_client_host=args.server_host,
        policy_client_port=args.server_port,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
        video_dir=str(args.video_dir),
        seed=args.seed,
    )
    elapsed_seconds = time.monotonic() - start_time

    successes = [bool(value) for value in successes]
    if len(successes) != args.n_episodes:
        raise RuntimeError(f"Expected {args.n_episodes} episode results, received {len(successes)}")

    json_infos = _jsonable(episode_infos)
    episodes: list[dict[str, Any]] = []
    for episode_index, success in enumerate(successes):
        record: dict[str, Any] = {
            "episode_index": episode_index,
            "success": success,
        }
        for key, values in json_infos.items():
            if isinstance(values, list) and len(values) == len(successes):
                record[key] = values[episode_index]
        episodes.append(record)

    payload = {
        "schema_version": 1,
        "suite": args.suite,
        "mode": args.mode,
        "task_index": args.task_index,
        "env_name": env_name,
        "seed": args.seed,
        "n_episodes": args.n_episodes,
        "n_envs": args.n_envs,
        "n_action_steps": args.n_action_steps,
        "max_episode_steps": args.max_episode_steps,
        "source_revision": args.source_revision,
        "checkpoint_revision": args.checkpoint_revision,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "success_count": sum(successes),
        "success_rate": float(np.mean(successes)),
        "episodes": episodes,
        "episode_infos": json_infos,
        "video_dir": str(args.video_dir),
    }
    _atomic_json_dump(payload, args.output_path)
    print(
        json.dumps({"output_path": str(args.output_path), "success_rate": payload["success_rate"]})
    )


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-suite HoloQ-style W4A4 pipeline on a plain GPU host (no Modal, no manifest).

Reproduces the report's protocol with the fake-quant backend and the static per-step
per-channel DiT activation tables: (1) calibration server + one seeded episode per task,
(2) ``tools/build_holoq_n1d7_pack.py``, (3) W4A4 server + ``--n-episodes`` per task.
``--rotation-mode svd`` swaps the HoloQ-style SVD-Hadamard block rotation for the
DuQuant-style SVD-only rotation; everything else is identical, so the two arms differ in
exactly that one choice.

Example::

    .venv/bin/python examples/LIBERO/quantization/run_local_w4a4.py \\
        --suite long --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \\
        --rotation-mode svd --out-root results/local_w4a4/duquant_long
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "examples" / "LIBERO" / "quantization"))
from modal_phase1_runner import SUITE_TASKS  # noqa: E402


def _git_rev(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _start_server(args, extra: list[str], log_path: Path) -> subprocess.Popen:
    command = [
        str(REPO / ".venv" / "bin" / "python"),
        str(REPO / "gr00t" / "eval" / "run_gr00t_server.py"),
        "--model-path", str(args.model_path),
        "--embodiment-tag", args.embodiment_tag,
        "--use-sim-policy-wrapper",
        "--port", str(args.port),
        *extra,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "a")
    log.write(" ".join(command) + "\n")
    process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + args.server_timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"policy server exited early; see {log_path}")
        if _port_open(args.port):
            time.sleep(5)
            return process
        time.sleep(2)
    process.kill()
    raise RuntimeError(f"policy server did not open port {args.port}; see {log_path}")


def _stop_server(process: subprocess.Popen) -> None:
    # The server writes the calibration artifact in the ``finally`` that follows its
    # ``except KeyboardInterrupt`` — i.e. it flushes on SIGINT.  SIGTERM (``terminate``)
    # kills the interpreter before that block runs and no artifact is written.
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=300)
        return
    except subprocess.TimeoutExpired:
        pass
    process.terminate()
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def _run_shard(args, *, mode: str, task_index: int, n_episodes: int, seed: int, out_dir: Path, revisions: dict) -> Path:
    out = out_dir / f"task_{task_index:02d}.json"
    if out.is_file():
        return out
    command = [
        str(args.client_python),
        str(REPO / "examples" / "LIBERO" / "quantization" / "run_libero_rollout_shard.py"),
        "--env-name", SUITE_TASKS[args.suite][task_index],
        "--output-path", str(out),
        "--suite", args.suite,
        "--mode", mode,
        "--task-index", str(task_index),
        "--seed", str(seed),
        "--n-episodes", str(n_episodes),
        "--n-envs", "1",
        "--n-action-steps", str(args.n_action_steps),
        "--max-episode-steps", str(args.max_episode_steps),
        "--server-port", str(args.port),
        "--video-dir", str(out_dir / "videos" / f"task_{task_index:02d}"),
        "--source-revision", revisions["source"],
        "--checkpoint-revision", revisions["checkpoint"],
    ]
    subprocess.run(command, cwd=REPO, check=True, env={**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, choices=tuple(SUITE_TASKS))
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--rotation-mode", default="svd_hadamard", choices=("svd_hadamard", "svd"))
    parser.add_argument("--checkpoint-revision", default="2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21")
    parser.add_argument("--embodiment-tag", default="LIBERO_PANDA")
    parser.add_argument("--client-python", type=Path, default=REPO / "gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python")
    parser.add_argument("--calibration-seed", default=0, type=int)
    parser.add_argument("--eval-seed", default=1, type=int)
    parser.add_argument("--n-episodes", default=20, type=int)
    parser.add_argument("--n-action-steps", default=8, type=int)
    parser.add_argument("--max-episode-steps", default=720, type=int)
    parser.add_argument("--calibration-topk", default=512, type=int)
    parser.add_argument("--port", default=5555, type=int)
    parser.add_argument("--server-timeout", default=900, type=int)
    parser.add_argument("--skip-rollout", action="store_true")
    args = parser.parse_args()

    root = args.out_root
    root.mkdir(parents=True, exist_ok=True)
    revisions = {"source": _git_rev(REPO), "checkpoint": args.checkpoint_revision}
    calibration = root / "calibration" / "holoq_calibration.pt"
    pack = root / "pack" / "quantized_w4a4.pt"

    # 1. calibration: one seeded episode per task through the bf16 policy with collectors attached
    if not calibration.is_file():
        partial = calibration.with_name(calibration.name + ".partial")
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.unlink(missing_ok=True)
        server = _start_server(
            args,
            [
                "--holoq-calibration-output", str(partial),
                "--holoq-suite", args.suite,
                "--holoq-calibration-run-id", f"{args.suite}-one-per-task-seed-{args.calibration_seed}",
                "--holoq-calibration-topk", str(args.calibration_topk),
                "--holoq-scopes", "llm,dit",
                "--holoq-dit-activation-granularity", "static-per-step-per-channel",
                "--holoq-rotation-mode", args.rotation_mode,
            ],
            root / "logs" / "server_calibration.log",
        )
        try:
            for task_index in range(len(SUITE_TASKS[args.suite])):
                _run_shard(args, mode="calibration", task_index=task_index, n_episodes=1,
                           seed=args.calibration_seed, out_dir=root / "calibration" / "rollouts", revisions=revisions)
        finally:
            _stop_server(server)
        if not partial.is_file():
            raise RuntimeError("calibration server wrote no artifact; see logs/server_calibration.log")
        os.replace(partial, calibration)

    # 2. pack
    if not pack.is_file():
        pack.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                str(REPO / ".venv" / "bin" / "python"), str(REPO / "tools" / "build_holoq_n1d7_pack.py"),
                "--model-path", str(args.model_path), "--calibration-path", str(calibration),
                "--output-path", str(pack), "--suite", args.suite,
                "--checkpoint-revision", revisions["checkpoint"], "--source-revision", revisions["source"],
                "--dit-activation-granularity", "static-per-step-per-channel",
            ],
            cwd=REPO, check=True,
        )
    if args.skip_rollout:
        return

    # 3. W4A4 rollout, fake-quant backend, n episodes per task
    roll_dir = root / "rollouts" / f"w4a4_seed{args.eval_seed}"
    server = _start_server(
        args,
        ["--holoq-pack-path", str(pack), "--holoq-suite", args.suite, "--holoq-backend", "fake"],
        root / "logs" / "server_w4a4.log",
    )
    outputs = []
    try:
        for task_index in range(len(SUITE_TASKS[args.suite])):
            outputs.append(_run_shard(args, mode="w4a4", task_index=task_index, n_episodes=args.n_episodes,
                                      seed=args.eval_seed, out_dir=roll_dir, revisions=revisions))
    finally:
        _stop_server(server)
    shards = [json.loads(p.read_text()) for p in outputs]
    summary = {
        "suite": args.suite, "rotation_mode": args.rotation_mode, "backend": "fake",
        "dit_activation_granularity": "static-per-step-per-channel",
        "n_episodes_per_task": args.n_episodes, "eval_seed": args.eval_seed, "calibration_seed": args.calibration_seed,
        "n_action_steps": args.n_action_steps, "max_episode_steps": args.max_episode_steps,
        "checkpoint_revision": revisions["checkpoint"], "source_revision": revisions["source"],
        "per_task": {s["env_name"]: s["success_count"] for s in shards},
        "successes": sum(s["success_count"] for s in shards),
        "episodes": sum(s["n_episodes"] for s in shards),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("suite", "rotation_mode", "successes", "episodes")}))


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase runner used by the hosted Modal notebook for GR00T-N1.7 LIBERO W4A4."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from typing import Any, Iterator
import zipfile


SUITE_CHECKPOINT_SUBDIR = {
    "object": "libero_object",
    "spatial": "libero_spatial",
    "goal": "libero_goal",
    "long": "libero_10",
}

SUITE_TASKS = {
    "long": (
        "libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "libero_sim/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "libero_sim/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "libero_sim/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "libero_sim/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "libero_sim/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "libero_sim/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "libero_sim/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    ),
    "goal": (
        "libero_sim/open_the_middle_drawer_of_the_cabinet",
        "libero_sim/put_the_bowl_on_the_stove",
        "libero_sim/put_the_wine_bottle_on_top_of_the_cabinet",
        "libero_sim/open_the_top_drawer_and_put_the_bowl_inside",
        "libero_sim/put_the_bowl_on_top_of_the_cabinet",
        "libero_sim/push_the_plate_to_the_front_of_the_stove",
        "libero_sim/put_the_cream_cheese_in_the_bowl",
        "libero_sim/turn_on_the_stove",
        "libero_sim/put_the_bowl_on_the_plate",
        "libero_sim/put_the_wine_bottle_on_the_rack",
    ),
    "object": (
        "libero_sim/pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "libero_sim/pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "libero_sim/pick_up_the_salad_dressing_and_place_it_in_the_basket",
        "libero_sim/pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        "libero_sim/pick_up_the_ketchup_and_place_it_in_the_basket",
        "libero_sim/pick_up_the_tomato_sauce_and_place_it_in_the_basket",
        "libero_sim/pick_up_the_butter_and_place_it_in_the_basket",
        "libero_sim/pick_up_the_milk_and_place_it_in_the_basket",
        "libero_sim/pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
        "libero_sim/pick_up_the_orange_juice_and_place_it_in_the_basket",
    ),
    "spatial": (
        "libero_sim/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "libero_sim/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "libero_sim/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "libero_sim/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "libero_sim/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "libero_sim/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "libero_sim/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "libero_sim/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "libero_sim/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ),
}

ROLLOUT_MODES = ("bf16", "w4a4")
SOURCE_BRANCH = "duc-quan"
EXPECTED_TASKS = 10
EXPECTED_LLM_LINEARS = 112
EXPECTED_DIT_LINEARS = 192
EXPECTED_TOTAL_LINEARS = 304


@dataclass(frozen=True)
class Config:
    suite: str
    work_phase: str
    repo_path: Path
    artifact_root: Path
    checkpoint_repo: str
    checkpoint_ref: str
    server_port: int
    calibration_seed: int
    evaluation_seed_base: int
    n_envs: int
    n_action_steps: int
    max_episode_steps: int
    calibration_topk: int
    smoke_task_index: int
    record_videos: bool
    package_include_videos: bool
    require_l4: bool

    @property
    def suite_root(self) -> Path:
        return self.artifact_root / "runs" / self.suite

    @property
    def checkpoint_root(self) -> Path:
        return self.artifact_root / "checkpoints" / "GR00T-N1.7-LIBERO"

    @property
    def model_path(self) -> Path:
        return self.checkpoint_root / SUITE_CHECKPOINT_SUBDIR[self.suite]

    @property
    def manifest_path(self) -> Path:
        return self.suite_root / "manifest.json"

    @property
    def calibration_path(self) -> Path:
        return self.suite_root / "calibration" / f"{self.suite}-calibration.pt"

    @property
    def pack_path(self) -> Path:
        return self.suite_root / "packs" / f"{self.suite}-w4a4.pt"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command))
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def _git(cfg: Config, *arguments: str) -> str:
    result = _run(
        ["git", "-C", str(cfg.repo_path), *arguments],
        capture_output=True,
    )
    return result.stdout.strip()


def _phase_marker(cfg: Config, phase: str, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "phase": phase,
        "suite": cfg.suite,
        "completed_at_utc": _utc_now(),
        "source_revision": _git(cfg, "rev-parse", "HEAD"),
    }
    if extra:
        payload.update(extra)
    _atomic_json(cfg.suite_root / "phase_state" / f"{phase}.json", payload)


def _restore_libero_config(cfg: Config) -> None:
    persisted = cfg.artifact_root / "libero_home_config"
    target = Path.home() / ".libero"
    if persisted.is_dir() and not target.exists():
        shutil.copytree(persisted, target)


def _validate_repo(cfg: Config) -> str:
    if not (cfg.repo_path / ".git").is_dir():
        raise RuntimeError(f"Repository is missing: {cfg.repo_path}. Run WORK_PHASE='setup'.")
    if _git(cfg, "status", "--porcelain"):
        raise RuntimeError(
            "Persistent repository has local changes; refusing a non-reproducible run"
        )
    source_revision = _git(cfg, "rev-parse", "HEAD")
    if not source_revision or len(source_revision) != 40:
        raise RuntimeError(f"Invalid source revision: {source_revision!r}")
    return source_revision


def _validate_l4(cfg: Config) -> dict[str, Any]:
    result = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
    )
    first_gpu = result.stdout.strip().splitlines()[0]
    gpu_name, memory_mib, driver_version = [part.strip() for part in first_gpu.split(",", 2)]
    if cfg.require_l4 and "L4" not in gpu_name.upper():
        raise RuntimeError(f"This protocol requires an NVIDIA L4; detected {gpu_name!r}")
    return {
        "gpu_name": gpu_name,
        "gpu_memory_mib": int(memory_mib),
        "driver_version": driver_version,
    }


def _require_manifest(cfg: Config) -> dict[str, Any]:
    if not cfg.manifest_path.is_file():
        raise RuntimeError("Missing suite manifest. Run WORK_PHASE='prepare'.")
    manifest = _load_json(cfg.manifest_path)
    source_revision = _validate_repo(cfg)
    if manifest.get("source_revision") != source_revision:
        raise RuntimeError("Manifest source revision does not match the persistent repository")
    if manifest.get("suite") != cfg.suite:
        raise RuntimeError("Manifest suite mismatch")
    current_protocol = {
        "calibration_seed": cfg.calibration_seed,
        "evaluation_seed_base": cfg.evaluation_seed_base,
        "n_envs": cfg.n_envs,
        "n_action_steps": cfg.n_action_steps,
        "max_episode_steps": cfg.max_episode_steps,
        "calibration_topk": cfg.calibration_topk,
    }
    mismatches = {
        key: {"manifest": manifest.get(key), "current": value}
        for key, value in current_protocol.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Notebook protocol changed after prepare: {mismatches}. Rerun WORK_PHASE='prepare'."
        )
    return manifest


def _resolve_checkpoint_revision(cfg: Config) -> str:
    python = cfg.repo_path / ".venv" / "bin" / "python"
    requested_ref = cfg.checkpoint_ref or "main"
    code = (
        "from huggingface_hub import HfApi; "
        f"print(HfApi().model_info({cfg.checkpoint_repo!r}, revision={requested_ref!r}).sha)"
    )
    result = _run([str(python), "-c", code], capture_output=True)
    revision = result.stdout.strip().splitlines()[-1]
    if len(revision) != 40:
        raise RuntimeError(f"Hugging Face returned an invalid revision: {revision!r}")
    return revision


def phase_prepare(cfg: Config) -> None:
    source_revision = _validate_repo(cfg)
    _restore_libero_config(cfg)
    gpu_info = _validate_l4(cfg)
    revision = _resolve_checkpoint_revision(cfg)
    checkpoint_subdir = SUITE_CHECKPOINT_SUBDIR[cfg.suite]
    hf_cli = cfg.repo_path / ".venv" / "bin" / "hf"
    cfg.checkpoint_root.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(hf_cli),
            "download",
            cfg.checkpoint_repo,
            "--revision",
            revision,
            "--include",
            f"{checkpoint_subdir}/*",
            "--local-dir",
            str(cfg.checkpoint_root),
        ],
        cwd=cfg.repo_path,
    )
    if not (cfg.model_path / "config.json").is_file():
        raise RuntimeError(f"Checkpoint download is incomplete: {cfg.model_path}")
    if not list(cfg.model_path.glob("*.safetensors")):
        raise RuntimeError(f"No safetensors found under {cfg.model_path}")

    cfg.suite_root.mkdir(parents=True, exist_ok=True)
    quant_config = (
        cfg.repo_path / "examples" / "LIBERO" / "quantization" / "configs" / f"{cfg.suite}.json"
    )
    config_copy = cfg.suite_root / "configs" / quant_config.name
    config_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(quant_config, config_copy)

    manifest = {
        "schema_version": 1,
        "suite": cfg.suite,
        "source_branch": SOURCE_BRANCH,
        "source_revision": source_revision,
        "checkpoint_repo": cfg.checkpoint_repo,
        "checkpoint_requested_ref": cfg.checkpoint_ref or "main",
        "checkpoint_revision": revision,
        "checkpoint_subdir": checkpoint_subdir,
        "model_path": str(cfg.model_path),
        "quant_config_sha256": _sha256(config_copy),
        "denoising_steps": 4,
        "expected_llm_linears": EXPECTED_LLM_LINEARS,
        "expected_dit_linears": EXPECTED_DIT_LINEARS,
        "expected_total_linears": EXPECTED_TOTAL_LINEARS,
        "calibration_seed": cfg.calibration_seed,
        "evaluation_seed_base": cfg.evaluation_seed_base,
        "n_envs": cfg.n_envs,
        "n_action_steps": cfg.n_action_steps,
        "max_episode_steps": cfg.max_episode_steps,
        "calibration_topk": cfg.calibration_topk,
        "tasks": list(SUITE_TASKS[cfg.suite]),
        "gpu": gpu_info,
        "prepared_at_utc": _utc_now(),
    }
    _atomic_json(cfg.manifest_path, manifest)
    _atomic_json(cfg.suite_root / "system" / "gpu.json", gpu_info)
    freeze_code = (
        "from importlib.metadata import distributions; "
        "rows=[d.metadata['Name']+'=='+d.version for d in distributions() "
        "if d.metadata.get('Name')]; print(chr(10).join(sorted(rows)))"
    )
    freeze = _run(
        [str(cfg.repo_path / ".venv" / "bin" / "python"), "-c", freeze_code],
        cwd=cfg.repo_path,
        capture_output=True,
    ).stdout
    (cfg.suite_root / "system" / "server-requirements.txt").write_text(freeze, encoding="utf-8")
    _phase_marker(cfg, "prepare", {"checkpoint_revision": revision})
    print(f"Prepared {cfg.suite} at checkpoint revision {revision}")


def _port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _wait_for_server(process: subprocess.Popen[str], port: int, timeout_seconds: int = 900) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Policy server exited early with code {process.returncode}")
        if _port_is_open("127.0.0.1", port):
            return
        time.sleep(2)
    raise TimeoutError(f"Policy server did not open port {port} within {timeout_seconds}s")


@contextmanager
def _policy_server(
    cfg: Config,
    *,
    mode: str,
    calibration_output: Path | None = None,
) -> Iterator[None]:
    if _port_is_open("127.0.0.1", cfg.server_port):
        raise RuntimeError(
            f"Port {cfg.server_port} is already in use. Stop the stale server or change SERVER_PORT."
        )
    _require_manifest(cfg)
    log_dir = cfg.suite_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"server-{mode}-{timestamp}.log"
    command = [
        str(cfg.repo_path / ".venv" / "bin" / "python"),
        "gr00t/eval/run_gr00t_server.py",
        "--model-path",
        str(cfg.model_path),
        "--embodiment-tag",
        "LIBERO_PANDA",
        "--use-sim-policy-wrapper",
        "--host",
        "127.0.0.1",
        "--port",
        str(cfg.server_port),
    ]
    if mode == "w4a4":
        if not cfg.pack_path.is_file():
            raise RuntimeError("Missing W4A4 pack. Run WORK_PHASE='build_pack'.")
        command.extend(["--holoq-pack-path", str(cfg.pack_path), "--holoq-suite", cfg.suite])
    elif mode == "calibration":
        if calibration_output is None:
            raise ValueError("calibration_output is required for calibration mode")
        command.extend(
            [
                "--holoq-calibration-output",
                str(calibration_output),
                "--holoq-suite",
                cfg.suite,
                "--holoq-calibration-run-id",
                f"{cfg.suite}-one-per-task-seed-{cfg.calibration_seed}",
                "--holoq-calibration-topk",
                str(cfg.calibration_topk),
            ]
        )
    elif mode != "bf16":
        raise ValueError(f"Unsupported server mode: {mode}")

    environment = os.environ.copy()
    environment.update(
        {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    print("+", " ".join(command))
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=cfg.repo_path,
            env=environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_for_server(process, cfg.server_port)
            print(f"Policy server ready: mode={mode}, log={log_path}")
            yield
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=30)
            if process.returncode not in (0, -signal.SIGINT):
                print(
                    f"WARNING: policy server exited with code {process.returncode}; inspect {log_path}"
                )


def _result_is_complete(
    path: Path,
    *,
    cfg: Config,
    manifest: dict[str, Any],
    mode: str,
    task_index: int,
    n_episodes: int,
    n_envs: int,
    seed: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        result = _load_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return all(
        (
            result.get("suite") == cfg.suite,
            result.get("mode") == mode,
            result.get("task_index") == task_index,
            result.get("n_episodes") == n_episodes,
            result.get("n_envs") == n_envs,
            result.get("n_action_steps") == cfg.n_action_steps,
            result.get("max_episode_steps") == cfg.max_episode_steps,
            result.get("seed") == seed,
            result.get("env_name") == SUITE_TASKS[cfg.suite][task_index],
            result.get("source_revision") == manifest["source_revision"],
            result.get("checkpoint_revision") == manifest["checkpoint_revision"],
            len(result.get("episodes", [])) == n_episodes,
        )
    )


def _run_shard(
    cfg: Config,
    *,
    mode: str,
    task_index: int,
    n_episodes: int,
    seed: int,
    output_path: Path,
    video_dir: Path,
    n_envs: int | None = None,
) -> None:
    manifest = _require_manifest(cfg)
    effective_n_envs = cfg.n_envs if n_envs is None else n_envs
    if _result_is_complete(
        output_path,
        cfg=cfg,
        manifest=manifest,
        mode=mode,
        task_index=task_index,
        n_episodes=n_episodes,
        n_envs=effective_n_envs,
        seed=seed,
    ):
        print(f"SKIP complete shard: {output_path}")
        return
    output_path.unlink(missing_ok=True)
    client_python = (
        cfg.repo_path
        / "gr00t"
        / "eval"
        / "sim"
        / "LIBERO"
        / "libero_uv"
        / ".venv"
        / "bin"
        / "python"
    )
    helper = cfg.repo_path / "examples" / "LIBERO" / "quantization" / "run_libero_rollout_shard.py"
    command = [
        str(client_python),
        str(helper),
        "--env-name",
        SUITE_TASKS[cfg.suite][task_index],
        "--output-path",
        str(output_path),
        "--suite",
        cfg.suite,
        "--mode",
        mode,
        "--task-index",
        str(task_index),
        "--seed",
        str(seed),
        "--n-episodes",
        str(n_episodes),
        "--n-envs",
        str(effective_n_envs),
        "--n-action-steps",
        str(cfg.n_action_steps),
        "--max-episode-steps",
        str(cfg.max_episode_steps),
        "--server-port",
        str(cfg.server_port),
        "--video-dir",
        str(video_dir),
        "--source-revision",
        manifest["source_revision"],
        "--checkpoint-revision",
        manifest["checkpoint_revision"],
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "PYTHONUNBUFFERED": "1",
        }
    )
    _run(command, cwd=cfg.repo_path, env=environment)
    if not _result_is_complete(
        output_path,
        cfg=cfg,
        manifest=manifest,
        mode=mode,
        task_index=task_index,
        n_episodes=n_episodes,
        n_envs=effective_n_envs,
        seed=seed,
    ):
        raise RuntimeError(f"Rollout helper produced an invalid result: {output_path}")
    if not cfg.record_videos and video_dir.is_dir():
        shutil.rmtree(video_dir)


def phase_calibrate(cfg: Config) -> None:
    manifest = _require_manifest(cfg)
    _validate_l4(cfg)
    cfg.calibration_path.parent.mkdir(parents=True, exist_ok=True)
    partial_artifact = cfg.calibration_path.with_name(cfg.calibration_path.name + ".partial")
    partial_artifact.unlink(missing_ok=True)
    rollout_root = cfg.suite_root / "calibration" / "rollouts"
    with _policy_server(cfg, mode="calibration", calibration_output=partial_artifact):
        for task_index in range(EXPECTED_TASKS):
            output_path = rollout_root / f"task_{task_index:02d}.json"
            output_path.unlink(missing_ok=True)
            _run_shard(
                cfg,
                mode="calibration",
                task_index=task_index,
                n_episodes=1,
                seed=cfg.calibration_seed,
                output_path=output_path,
                video_dir=cfg.suite_root / "videos" / "calibration" / f"task_{task_index:02d}",
                n_envs=1,
            )
    if not partial_artifact.is_file():
        raise RuntimeError(
            "Calibration server did not write its artifact; inspect the latest server log"
        )
    os.replace(partial_artifact, cfg.calibration_path)
    index = {
        "suite": cfg.suite,
        "source_revision": manifest["source_revision"],
        "checkpoint_revision": manifest["checkpoint_revision"],
        "seed": cfg.calibration_seed,
        "tasks": list(SUITE_TASKS[cfg.suite]),
        "artifact": str(cfg.calibration_path),
        "artifact_sha256": _sha256(cfg.calibration_path),
        "created_at_utc": _utc_now(),
    }
    _atomic_json(cfg.suite_root / "calibration" / "index.json", index)
    _phase_marker(cfg, "calibrate", {"calibration_sha256": index["artifact_sha256"]})
    print(f"Calibration complete: {cfg.calibration_path}")


def phase_build_pack(cfg: Config) -> None:
    manifest = _require_manifest(cfg)
    _validate_l4(cfg)
    if not cfg.calibration_path.is_file():
        raise RuntimeError("Missing calibration artifact. Run WORK_PHASE='calibrate'.")
    cfg.pack_path.parent.mkdir(parents=True, exist_ok=True)
    partial_pack = cfg.pack_path.with_name(cfg.pack_path.name + ".partial")
    partial_sidecar = partial_pack.with_suffix(partial_pack.suffix + ".sha256")
    partial_pack.unlink(missing_ok=True)
    partial_sidecar.unlink(missing_ok=True)
    _run(
        [
            str(cfg.repo_path / ".venv" / "bin" / "python"),
            "tools/build_holoq_n1d7_pack.py",
            "--model-path",
            str(cfg.model_path),
            "--calibration-path",
            str(cfg.calibration_path),
            "--output-path",
            str(partial_pack),
            "--suite",
            cfg.suite,
            "--checkpoint-revision",
            manifest["checkpoint_revision"],
            "--source-revision",
            manifest["source_revision"],
        ],
        cwd=cfg.repo_path,
    )
    if not partial_pack.is_file() or not partial_sidecar.is_file():
        raise RuntimeError("Pack builder did not produce the pack and checksum sidecar")
    pack_sha = _sha256(partial_pack)
    if partial_sidecar.read_text(encoding="ascii").strip() != pack_sha:
        raise RuntimeError("Pack checksum sidecar does not match the generated pack")
    os.replace(partial_pack, cfg.pack_path)
    final_sidecar = cfg.pack_path.with_suffix(cfg.pack_path.suffix + ".sha256")
    os.replace(partial_sidecar, final_sidecar)

    validate_code = (
        "import json, torch; "
        f"p=torch.load({str(cfg.pack_path)!r}, map_location='cpu', weights_only=False); "
        "m=p['manifest']; "
        "print(json.dumps({'suite':m['suite'],'llm_linears':m['llm_linears'],"
        "'dit_linears':m['dit_linears'],'total_linears':m['total_linears'],"
        "'num_inference_timesteps':m['num_inference_timesteps'],"
        "'source_revision':m['source_revision'],'checkpoint_revision':m['checkpoint_revision']}))"
    )
    result = _run(
        [str(cfg.repo_path / ".venv" / "bin" / "python"), "-c", validate_code],
        cwd=cfg.repo_path,
        capture_output=True,
    )
    pack_manifest = json.loads(result.stdout.strip().splitlines()[-1])
    expected = {
        "suite": cfg.suite,
        "llm_linears": EXPECTED_LLM_LINEARS,
        "dit_linears": EXPECTED_DIT_LINEARS,
        "total_linears": EXPECTED_TOTAL_LINEARS,
        "num_inference_timesteps": 4,
        "source_revision": manifest["source_revision"],
        "checkpoint_revision": manifest["checkpoint_revision"],
    }
    if pack_manifest != expected:
        raise RuntimeError(f"Strict pack validation failed: {pack_manifest!r} != {expected!r}")
    _atomic_json(
        cfg.suite_root / "packs" / "index.json",
        {"pack": str(cfg.pack_path), "sha256": pack_sha, "manifest": pack_manifest},
    )
    _phase_marker(cfg, "build_pack", {"pack_sha256": pack_sha})
    print(f"Strict W4A4 pack complete: {cfg.pack_path}")


def _run_rollout_phase(cfg: Config, *, smoke: bool) -> None:
    _require_manifest(cfg)
    _validate_l4(cfg)
    if not cfg.pack_path.is_file():
        raise RuntimeError("Missing W4A4 pack. Run WORK_PHASE='build_pack'.")
    if smoke:
        task_indices = (cfg.smoke_task_index,)
        n_episodes = 1
        phase_name = "smoke_rollout"
        result_root = cfg.suite_root / "smoke"
        video_root = cfg.suite_root / "videos" / "smoke"
    else:
        for mode in ROLLOUT_MODES:
            smoke_path = cfg.suite_root / "smoke" / mode / f"task_{cfg.smoke_task_index:02d}.json"
            if not smoke_path.is_file():
                raise RuntimeError("Missing smoke result. Run WORK_PHASE='smoke_rollout'.")
        task_indices = tuple(range(EXPECTED_TASKS))
        n_episodes = 10
        phase_name = "full_rollout"
        result_root = cfg.suite_root / "results"
        video_root = cfg.suite_root / "videos" / "full"

    for mode in ROLLOUT_MODES:
        with _policy_server(cfg, mode=mode):
            for task_index in task_indices:
                seed = cfg.evaluation_seed_base + task_index * 100
                _run_shard(
                    cfg,
                    mode=mode,
                    task_index=task_index,
                    n_episodes=n_episodes,
                    seed=seed,
                    output_path=result_root / mode / f"task_{task_index:02d}.json",
                    video_dir=video_root / mode / f"task_{task_index:02d}",
                    n_envs=1 if smoke else cfg.n_envs,
                )

    if not smoke:
        _write_metrics(cfg)
    _phase_marker(cfg, phase_name, {"episodes_per_task": n_episodes})
    print(f"{phase_name} complete for suite={cfg.suite}")


def phase_smoke_rollout(cfg: Config) -> None:
    _run_rollout_phase(cfg, smoke=True)


def phase_full_rollout(cfg: Config) -> None:
    _run_rollout_phase(cfg, smoke=False)


def _collect_mode_results(cfg: Config, mode: str) -> list[dict[str, Any]]:
    manifest = _require_manifest(cfg)
    results = []
    for task_index in range(EXPECTED_TASKS):
        path = cfg.suite_root / "results" / mode / f"task_{task_index:02d}.json"
        seed = cfg.evaluation_seed_base + task_index * 100
        if not _result_is_complete(
            path,
            cfg=cfg,
            manifest=manifest,
            mode=mode,
            task_index=task_index,
            n_episodes=10,
            n_envs=cfg.n_envs,
            seed=seed,
        ):
            raise RuntimeError(f"Incomplete final rollout shard: {path}")
        results.append(_load_json(path))
    return results


def _write_metrics(cfg: Config) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "suite": cfg.suite,
        "generated_at_utc": _utc_now(),
        "modes": {},
    }
    for mode in ROLLOUT_MODES:
        results = _collect_mode_results(cfg, mode)
        task_metrics = []
        total_successes = 0
        total_episodes = 0
        for result in results:
            successes = sum(bool(episode["success"]) for episode in result["episodes"])
            episodes = len(result["episodes"])
            total_successes += successes
            total_episodes += episodes
            task_metrics.append(
                {
                    "task_index": result["task_index"],
                    "env_name": result["env_name"],
                    "successes": successes,
                    "episodes": episodes,
                    "success_rate": successes / episodes,
                    "elapsed_seconds": result["elapsed_seconds"],
                }
            )
        payload["modes"][mode] = {
            "successes": total_successes,
            "episodes": total_episodes,
            "success_rate": total_successes / total_episodes,
            "tasks": task_metrics,
        }
    payload["w4a4_minus_bf16"] = (
        payload["modes"]["w4a4"]["success_rate"] - payload["modes"]["bf16"]["success_rate"]
    )
    _atomic_json(cfg.suite_root / "metrics" / "summary.json", payload)
    return payload


def phase_package(cfg: Config) -> None:
    manifest = _require_manifest(cfg)
    metrics = _write_metrics(cfg)
    if not cfg.calibration_path.is_file() or not cfg.pack_path.is_file():
        raise RuntimeError("Calibration or W4A4 pack is missing")

    selected: list[tuple[Path, str]] = []
    for path in sorted(cfg.suite_root.rglob("*")):
        if not path.is_file() or path.name.endswith(".partial") or path.suffix == ".zip":
            continue
        relative = path.relative_to(cfg.suite_root)
        if relative.parts and relative.parts[0] == "exports":
            continue
        if not cfg.package_include_videos and relative.parts and relative.parts[0] == "videos":
            continue
        selected.append((path, f"{cfg.suite}/{relative.as_posix()}"))

    inventory = []
    for path, archive_name in selected:
        inventory.append(
            {
                "path": archive_name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    handoff = {
        "schema_version": 1,
        "suite": cfg.suite,
        "source_revision": manifest["source_revision"],
        "checkpoint_revision": manifest["checkpoint_revision"],
        "bf16_episodes": metrics["modes"]["bf16"]["episodes"],
        "w4a4_episodes": metrics["modes"]["w4a4"]["episodes"],
        "package_includes_videos": cfg.package_include_videos,
        "file_count": len(inventory),
        "total_size_bytes": sum(item["size_bytes"] for item in inventory),
        "created_at_utc": _utc_now(),
        "files": inventory,
    }
    checksums = "".join(f"{item['sha256']}  {item['path']}\n" for item in inventory)

    export_dir = cfg.artifact_root / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    final_zip = export_dir / f"gr00t-n17-holoq-{cfg.suite}-{timestamp}.zip"
    with tempfile.TemporaryDirectory(dir=export_dir) as temp_dir:
        temporary_zip = Path(temp_dir) / final_zip.name
        with zipfile.ZipFile(temporary_zip, "w", allowZip64=True) as archive:
            for path, archive_name in selected:
                stored = path.suffix.lower() in {".pt", ".safetensors", ".mp4", ".zip"}
                archive.write(
                    path,
                    archive_name,
                    compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED,
                    compresslevel=None if stored else 6,
                )
            archive.writestr(
                "handoff.json",
                json.dumps(handoff, indent=2, ensure_ascii=False) + "\n",
                compress_type=zipfile.ZIP_DEFLATED,
            )
            archive.writestr(
                "checksums.sha256",
                checksums,
                compress_type=zipfile.ZIP_DEFLATED,
            )
        os.replace(temporary_zip, final_zip)

    zip_sha = _sha256(final_zip)
    sidecar = final_zip.with_suffix(final_zip.suffix + ".sha256")
    sidecar.write_text(f"{zip_sha}  {final_zip.name}\n", encoding="ascii")
    last_export = {
        "suite": cfg.suite,
        "zip_path": str(final_zip),
        "sha256_path": str(sidecar),
        "zip_sha256": zip_sha,
        "zip_size_bytes": final_zip.stat().st_size,
        "created_at_utc": _utc_now(),
    }
    _atomic_json(cfg.artifact_root / "last_export.json", last_export)
    _phase_marker(cfg, "package", last_export)
    print(json.dumps(last_export, indent=2))


def phase_status(cfg: Config) -> None:
    source_revision = _validate_repo(cfg)
    payload: dict[str, Any] = {
        "suite": cfg.suite,
        "source_revision": source_revision,
        "model_path_exists": cfg.model_path.is_dir(),
        "calibration_exists": cfg.calibration_path.is_file(),
        "pack_exists": cfg.pack_path.is_file(),
        "phases": {},
        "result_shards": {},
    }
    for phase in ("prepare", "calibrate", "build_pack", "smoke_rollout", "full_rollout", "package"):
        marker = cfg.suite_root / "phase_state" / f"{phase}.json"
        payload["phases"][phase] = _load_json(marker) if marker.is_file() else None
    for mode in ROLLOUT_MODES:
        payload["result_shards"][mode] = len(
            list((cfg.suite_root / "results" / mode).glob("*.json"))
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


PHASES = {
    "prepare": phase_prepare,
    "calibrate": phase_calibrate,
    "build_pack": phase_build_pack,
    "smoke_rollout": phase_smoke_rollout,
    "full_rollout": phase_full_rollout,
    "package": phase_package,
    "status": phase_status,
}


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=tuple(SUITE_TASKS))
    parser.add_argument("--work-phase", required=True, choices=tuple(PHASES))
    parser.add_argument("--repo-path", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--checkpoint-repo", default="nvidia/GR00T-N1.7-LIBERO")
    parser.add_argument("--checkpoint-ref", default="")
    parser.add_argument("--server-port", default=5555, type=int)
    parser.add_argument("--calibration-seed", default=0, type=int)
    parser.add_argument("--evaluation-seed-base", default=10000, type=int)
    parser.add_argument("--n-envs", default=1, type=int)
    parser.add_argument("--n-action-steps", default=8, type=int)
    parser.add_argument("--max-episode-steps", default=720, type=int)
    parser.add_argument("--calibration-topk", default=512, type=int)
    parser.add_argument("--smoke-task-index", default=0, type=int)
    parser.add_argument("--record-videos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--package-include-videos", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--require-l4", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not 0 <= args.smoke_task_index < EXPECTED_TASKS:
        parser.error(f"--smoke-task-index must be in [0, {EXPECTED_TASKS - 1}]")
    if args.n_envs <= 0:
        parser.error("--n-envs must be positive")
    if 10 % args.n_envs != 0:
        parser.error("--n-envs must divide the ten final-evaluation episodes per task")
    return Config(**vars(args))


def main() -> None:
    cfg = _parse_args()
    cfg.artifact_root.mkdir(parents=True, exist_ok=True)
    _restore_libero_config(cfg)
    print(
        json.dumps(
            {
                **asdict(cfg),
                "repo_path": str(cfg.repo_path),
                "artifact_root": str(cfg.artifact_root),
            },
            default=str,
            indent=2,
        )
    )
    PHASES[cfg.work_phase](cfg)


if __name__ == "__main__":
    main()

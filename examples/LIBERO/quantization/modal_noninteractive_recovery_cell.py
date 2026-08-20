"""One-time Modal cell: recover a partial LIBERO setup without any stdin prompt.

Paste this whole file into a code cell immediately after the configuration cell,
set WORK_PHASE="setup" and FORCE_SETUP=False, then run all.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess


if WORK_PHASE == "setup":
    recovery_root = Path(VOLUME_ROOT).resolve()
    recovery_repo = recovery_root / "Isaac-GR00T-duc-quan"
    recovery_artifacts = recovery_root / "artifacts"
    recovery_marker = recovery_root / "setup.json"
    libero_root = recovery_repo / "external_dependencies" / "LIBERO" / "libero" / "libero"
    libero_python = (
        recovery_repo
        / "gr00t"
        / "eval"
        / "sim"
        / "LIBERO"
        / "libero_uv"
        / ".venv"
        / "bin"
        / "python"
    )

    if FORCE_SETUP:
        raise RuntimeError("Set FORCE_SETUP=False to reuse the existing LIBERO environment")
    if not (recovery_repo / ".git").is_dir():
        raise RuntimeError("Persistent repo is missing; the regular setup phase must clone it first")
    if not libero_python.is_file():
        raise RuntimeError("Partial LIBERO venv is missing; the regular setup phase must build it")
    if not libero_root.is_dir():
        raise RuntimeError("LIBERO submodule is missing; initialize it before using this recovery cell")

    # Ignore only setup-generated state. Real source changes remain visible to
    # the bootstrap guard; no reset, checkout, clean, or deletion is performed.
    subprocess.run(
        [
            "git",
            "-C",
            str(recovery_repo),
            "config",
            "submodule.external_dependencies/LIBERO.ignore",
            "dirty",
        ],
        check=True,
    )
    info_exclude = recovery_repo / ".git" / "info" / "exclude"
    generated_rule = "/gr00t/eval/sim/LIBERO/libero_uv/"
    existing_rules = info_exclude.read_text(encoding="utf-8") if info_exclude.is_file() else ""
    if generated_rule not in existing_rules.splitlines():
        info_exclude.parent.mkdir(parents=True, exist_ok=True)
        with info_exclude.open("a", encoding="utf-8") as stream:
            if existing_rules and not existing_rules.endswith("\n"):
                stream.write("\n")
            stream.write(generated_rule + "\n")

    # Create the canonical LIBERO configuration before importing LIBERO. This
    # completely bypasses its interactive initialization branch.
    config_dir = Path.home() / ".libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    config_values = {
        "benchmark_root": libero_root,
        "bddl_files": libero_root / "bddl_files",
        "init_states": libero_root / "init_files",
        "datasets": libero_root.parent / "datasets",
        "assets": libero_root / "assets",
    }
    config_path.write_text(
        "".join(f"{key}: {Path(value).resolve().as_posix()}\n" for key, value in config_values.items()),
        encoding="utf-8",
    )
    print(f"Wrote non-interactive LIBERO config: {config_path}")

    recovery_environment = os.environ.copy()
    recovery_environment.update(
        {
            "MPLBACKEND": "Agg",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "PYTHONUNBUFFERED": "1",
        }
    )
    smoke_code = """
from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
register_libero_envs()
import gymnasium as gym
env = gym.make("libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate")
env.reset()
env.close()
print("Recovered LIBERO env OK:", type(env))
"""
    subprocess.run(
        [str(libero_python), "-c", smoke_code],
        cwd=recovery_repo,
        env=recovery_environment,
        check=True,
        text=True,
    )

    persisted_config = recovery_artifacts / "libero_home_config"
    recovery_artifacts.mkdir(parents=True, exist_ok=True)
    shutil.copytree(config_dir, persisted_config, dirs_exist_ok=True)
    source_revision = subprocess.check_output(
        ["git", "-C", str(recovery_repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    recovery_marker.write_text(
        json.dumps(
            {
                "source_branch": SOURCE_BRANCH,
                "source_revision": source_revision,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "recovery": "noninteractive_partial_setup",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("NON-INTERACTIVE PARTIAL SETUP RECOVERED")

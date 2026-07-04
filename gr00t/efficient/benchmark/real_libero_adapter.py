# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real LIBERO adapter for efficient inference benchmark runs."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any

from gr00t.efficient.benchmark.real_metrics import (
    compute_action_l2_trace,
    finish_episode_record,
    load_action_trace,
    record_action_latency,
    save_action_trace,
    start_episode_record,
)
from gr00t.efficient.hooks.visual_token_hook import VisualTokenHook, attach_visual_token_hook


DEFAULT_LIBERO_TASK = "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
DEFAULT_REAL_OUTPUT_DIR = Path("results/efficient_benchmark/real_libero")
DEFAULT_LIBERO_PYTHON = Path("gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python")
DEFAULT_MODEL_PATH = Path("checkpoints/GR00T-N1.7-LIBERO/libero_10")


class RealLiberoAdapter:
    """Run real LIBERO evaluation and return benchmark records."""

    def __init__(self, args: Any, method: Any | None = None) -> None:
        self.args = args
        self.method = method
        self.method_name = getattr(method, "method_name", getattr(args, "method", "baseline"))
        self.real_output_dir = Path(
            getattr(args, "real_output_dir", None) or DEFAULT_REAL_OUTPUT_DIR
        )

    def run(self) -> list[dict[str, Any]]:
        """Run real LIBERO evaluation and return list[dict] records."""
        if getattr(self.args, "real_command", ""):
            return self.run_existing_libero_baseline_subprocess(self.args.real_command)

        model_path = self._resolve_model_path()
        if model_path is not None:
            return self.run_existing_libero_baseline_python(model_path)

        if self.method_name != "baseline":
            raise RuntimeError(
                "Real pruning methods require --model-path or --checkpoint-path so the efficient "
                "adapter can attach the visual token hook to the loaded GR00T model instance."
            )

        libero_python = Path(getattr(self.args, "libero_python", "") or DEFAULT_LIBERO_PYTHON)
        if not libero_python.exists():
            raise RuntimeError(
                "Real LIBERO baseline needs either --model-path/--checkpoint-path for in-process "
                f"evaluation or a LIBERO sim Python at {libero_python}. Run "
                "gr00t/eval/sim/LIBERO/setup_libero.sh, start run_gr00t_server.py, or pass "
                "--real-command with the existing working rollout command."
            )
        return self.run_existing_libero_baseline_subprocess(self._build_default_rollout_command())

    def run_existing_libero_baseline_subprocess(self, command: str | list[str]) -> list[dict[str, Any]]:
        """Run the existing rollout script in a subprocess and parse coarse metrics."""
        if self.method_name != "baseline":
            raise RuntimeError(
                "Subprocess mode can only verify baseline unless the target server was started "
                "with an efficient visual-token hook. Use --model-path for dummy/vlapruner runs."
            )

        cmd = shlex.split(command) if isinstance(command, str) else list(command)
        start_time = time.perf_counter()
        completed = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            env=os.environ.copy(),
            text=True,
            capture_output=True,
            check=False,
        )
        elapsed_s = time.perf_counter() - start_time
        output = (completed.stdout or "") + "\n" + (completed.stderr or "")
        if completed.returncode != 0:
            raise RuntimeError(
                "Existing LIBERO rollout command failed with exit code "
                f"{completed.returncode}.\nCommand: {' '.join(cmd)}\n"
                f"Output tail:\n{output[-4000:]}"
            )

        success_rate = _parse_success_rate(output)
        num_episodes = int(getattr(self.args, "num_episodes", 1))
        episode_time_s = elapsed_s / max(1, num_episodes)
        latency_ms = (episode_time_s / max(1, int(getattr(self.args, "max_steps", 1)))) * 1000.0
        task = self._resolve_task()

        records = []
        for episode_id in range(num_episodes):
            record = start_episode_record(
                benchmark="LIBERO",
                method=self.method_name,
                task=task,
                episode_id=episode_id,
                keep_ratio=1.0,
                notes="real subprocess baseline; latency approximated from rollout wall time",
            )
            record["latency_per_action_ms"] = latency_ms
            records.append(
                finish_episode_record(
                    record,
                    success=success_rate > 0.0 if num_episodes > 1 else success_rate >= 1.0,
                    success_rate=success_rate,
                    episode_time_s=episode_time_s,
                    visual_metadata=None,
                    action_l2_vs_baseline=0.0,
                    notes="real subprocess baseline; existing rollout output parsed",
                )
            )
        return records

    def run_existing_libero_baseline_python(self, model_path: Path) -> list[dict[str, Any]]:
        """Run real LIBERO in-process so latency, traces, and hooks can be captured."""
        from gr00t.eval._horizon_contract import PolicyHorizonSpec
        from gr00t.eval.rollout_policy import (
            MultiStepConfig,
            VideoConfig,
            WrapperConfigs,
            run_rollout_gymnasium_policy,
        )
        from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
        from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

        task = self._resolve_task()
        embodiment_tag = get_embodiment_tag_from_env_name(task)
        gr00t_policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=str(model_path),
            device=getattr(self.args, "device", "cuda"),
        )
        base_policy = Gr00tSimPolicyWrapper(gr00t_policy)

        visual_hook = None
        if self._should_enable_hook():
            model = getattr(gr00t_policy, "model", None)
            if model is None:
                raise RuntimeError("Could not locate Gr00tPolicy.model for visual token hook attachment.")
            visual_hook = VisualTokenHook(method=self.method, enabled=True)
            attach_visual_token_hook(model, visual_hook)

        metric_policy = _MetricPolicyWrapper(base_policy)
        contract = PolicyHorizonSpec.from_policy(
            metric_policy,
            n_action_steps=int(getattr(self.args, "n_action_steps", 8)),
        )
        wrapper_configs = WrapperConfigs(
            multistep=MultiStepConfig(
                contract=contract,
                max_episode_steps=int(getattr(self.args, "max_steps", 720)),
                terminate_on_success=True,
            ),
            video=VideoConfig(
                video_dir=getattr(self.args, "video_dir", None),
                max_episode_steps=int(getattr(self.args, "max_steps", 720)),
            ),
        )

        start_time = time.perf_counter()
        env_name, successes, episode_infos = run_rollout_gymnasium_policy(
            env_name=task,
            policy=metric_policy,
            wrapper_configs=wrapper_configs,
            n_episodes=int(getattr(self.args, "num_episodes", 1)),
            n_envs=int(getattr(self.args, "n_envs", 1)),
            seed=getattr(self.args, "seed", None),
        )
        elapsed_s = time.perf_counter() - start_time

        del env_name
        episode_count = len(successes)
        episode_time_s = elapsed_s / max(1, episode_count)
        latency_samples = metric_policy.action_latencies_ms
        visual_metadata = visual_hook.last_metadata if visual_hook is not None else None
        success_rate = sum(bool(success) for success in successes) / max(1, episode_count)

        records = []
        for episode_id, success in enumerate(successes):
            trace_path = self._trace_path(self.method_name, task, episode_id)
            baseline_trace_path = self._trace_path("baseline", task, episode_id)
            action_l2 = 0.0
            if self._should_save_actions():
                save_action_trace(
                    trace_path,
                    metric_policy.action_trace,
                    metadata={
                        "method": self.method_name,
                        "task": task,
                        "episode_id": episode_id,
                        "visual_metadata": visual_metadata,
                    },
                )
            if self.method_name != "baseline" and baseline_trace_path.exists():
                action_l2 = compute_action_l2_trace(
                    load_action_trace(trace_path),
                    load_action_trace(baseline_trace_path),
                )

            record = start_episode_record(
                benchmark="LIBERO",
                method=self.method_name,
                task=task,
                episode_id=episode_id,
                keep_ratio=float(getattr(self.method, "keep_ratio", getattr(self.args, "keep_ratio", 1.0))),
                notes="real in-process LIBERO run",
            )
            records.append(
                finish_episode_record(
                    record,
                    success=bool(success),
                    success_rate=success_rate,
                    episode_time_s=_episode_time_from_infos(episode_infos, episode_id, episode_time_s),
                    action_latencies_ms=latency_samples,
                    visual_metadata=visual_metadata,
                    action_l2_vs_baseline=action_l2,
                    failure_type="" if success else "episode_failed",
                    notes="real in-process LIBERO run",
                )
            )
        return records

    def _resolve_model_path(self) -> Path | None:
        raw_path = (
            getattr(self.args, "model_path", None)
            or getattr(self.args, "checkpoint_path", None)
            or ""
        )
        if raw_path:
            path = Path(raw_path)
            if not path.exists():
                raise FileNotFoundError(f"Model/checkpoint path does not exist: {path}")
            return path
        if DEFAULT_MODEL_PATH.exists():
            return DEFAULT_MODEL_PATH
        return None

    def _resolve_task(self) -> str:
        task = getattr(self.args, "task", "") or "debug"
        if task == "debug":
            return DEFAULT_LIBERO_TASK
        if task.startswith("libero_sim/"):
            return task
        return f"libero_sim/{task}"

    def _build_default_rollout_command(self) -> list[str]:
        libero_python = str(Path(getattr(self.args, "libero_python", "") or DEFAULT_LIBERO_PYTHON))
        command = [
            libero_python,
            "gr00t/eval/rollout_policy.py",
            "--n-episodes",
            str(getattr(self.args, "num_episodes", 1)),
            "--policy-client-host",
            str(getattr(self.args, "policy_client_host", "127.0.0.1")),
            "--policy-client-port",
            str(getattr(self.args, "policy_client_port", 5555)),
            "--max-episode-steps",
            str(getattr(self.args, "max_steps", 720)),
            "--env-name",
            self._resolve_task(),
            "--n-action-steps",
            str(getattr(self.args, "n_action_steps", 8)),
            "--n-envs",
            str(getattr(self.args, "n_envs", 1)),
        ]
        seed = getattr(self.args, "seed", None)
        if seed is not None:
            command.extend(["--seed", str(seed)])
        return command

    def _should_enable_hook(self) -> bool:
        if getattr(self.args, "disable_visual_token_hook", False):
            return False
        if self.method_name == "baseline":
            return bool(getattr(self.args, "hook_visual_tokens", False))
        return True

    def _should_save_actions(self) -> bool:
        return bool(getattr(self.args, "save_actions", False) or getattr(self.args, "save_traces", False))

    def _trace_path(self, method: str, task: str, episode_id: int) -> Path:
        safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task.replace("libero_sim/", ""))
        return self.real_output_dir / "traces" / f"{method}_task_{safe_task}_episode_{episode_id}.npz"


class _MetricPolicyWrapper:
    """Small policy wrapper that records action latencies and action traces."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.action_latencies_ms: list[float] = []
        self.action_trace: list[Any] = []

    def get_action(self, observation: dict[str, Any], options: dict[str, Any] | None = None) -> Any:
        start_time = time.perf_counter()
        action, info = self.policy.get_action(observation, options)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        record_action_latency(self.action_latencies_ms, elapsed_ms)
        self.action_trace.append(action)
        return action, info

    def reset(self) -> Any:
        return self.policy.reset()

    def get_modality_config(self) -> Any:
        return self.policy.get_modality_config()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.policy, name)


def _parse_success_rate(output: str) -> float:
    match = re.search(r"success rate:\s*([0-9.]+)", output)
    if not match:
        raise RuntimeError(
            "Could not parse 'success rate:' from existing LIBERO rollout output.\n"
            f"Output tail:\n{output[-4000:]}"
        )
    return float(match.group(1))


def _episode_time_from_infos(
    episode_infos: dict[str, Any],
    episode_id: int,
    fallback_episode_time_s: float,
) -> float:
    del episode_id
    lengths = episode_infos.get("episode_lengths")
    if lengths:
        return float(fallback_episode_time_s)
    return float(fallback_episode_time_s)

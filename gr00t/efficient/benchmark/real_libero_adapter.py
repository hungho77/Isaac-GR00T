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
from gr00t.efficient.hooks.backbone_token_hook import (
    BackboneVisualTokenHook,
    attach_backbone_visual_token_hook,
)
from gr00t.efficient.hooks.visual_merger_hook import VisualMergerHook, attach_visual_merger_hook
from gr00t.efficient.hooks.visual_token_hook import VisualTokenHook, attach_visual_token_hook
from gr00t.efficient.profiler.model_stages import (
    attach_stage_profiler,
    stage_summary_with_gpu_memory,
)


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
        # Populated by run_existing_libero_baseline_python() when --profile-stages
        # is set; run_libero.py reads this to add a `stage_profile` block to the
        # output JSON (vision encoder / LLM backbone / DiT action head latency).
        self.last_stage_profile: dict[str, Any] | None = None

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

    def run_existing_libero_baseline_subprocess(
        self, command: str | list[str]
    ) -> list[dict[str, Any]]:
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
        from gr00t.utils.determinism import seed_everything

        # Seed torch/numpy RNGs so flow-matching action noise is reproducible and
        # action-L2 vs baseline measures pruning drift, not sampling noise.
        # Method reset happens per-episode below (see _run_episodes_one_at_a_time),
        # not here, so temporal-momentum state doesn't leak across episodes.
        seed = seed_everything(getattr(self.args, "seed", None))

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
                raise RuntimeError(
                    "Could not locate Gr00tPolicy.model for visual token hook attachment."
                )
            prune_stage = getattr(self.args, "prune_stage", "action_head")
            if prune_stage == "backbone":
                visual_hook = BackboneVisualTokenHook(
                    method=self.method,
                    enabled=True,
                    prune_layer=int(getattr(self.args, "prune_layer", 3)),
                )
                attach_backbone_visual_token_hook(model, visual_hook)
            elif prune_stage == "after_visual_merger":
                visual_hook = VisualMergerHook(
                    method=self.method,
                    enabled=True,
                    mode=getattr(self.args, "prune_mode", "mask_only"),
                )
                attach_visual_merger_hook(model, visual_hook)
            else:
                visual_hook = VisualTokenHook(method=self.method, enabled=True)
                attach_visual_token_hook(model, visual_hook)

        self._maybe_compile_model(getattr(gr00t_policy, "model", None))

        stage_profiler = None
        stage_detach = None
        if getattr(self.args, "profile_stages", False):
            # Attach only after the pruning hook (and any torch.compile) so the
            # profiler wraps whatever is actually live: backbone-stage pruning
            # replaces Qwen3VLTextModel.forward outright (a full reimplementation
            # that never calls a previous wrapper), so attaching the profiler
            # first would make it wrap a forward that's no longer on the call
            # path -- language_model.forward's timing would silently go missing.
            stage_profiler, stage_detach = attach_stage_profiler(
                getattr(gr00t_policy, "model", None)
            )

        contract = PolicyHorizonSpec.from_policy(
            base_policy,
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

        num_episodes = int(getattr(self.args, "num_episodes", 1))
        n_envs = int(getattr(self.args, "n_envs", 1))

        if n_envs == 1:
            successes, episode_infos, per_episode = self._run_episodes_one_at_a_time(
                task=task,
                base_policy=base_policy,
                visual_hook=visual_hook,
                wrapper_configs=wrapper_configs,
                num_episodes=num_episodes,
                seed=seed,
            )
        else:
            # Parallel-env path: per-episode temporal-state reset doesn't apply
            # cleanly to a batch of concurrently-running episodes, and
            # backbone-stage pruning already requires batch_size==1 (no-ops
            # otherwise), so this path is unaffected by the state-leak fix below.
            if self.method is not None and hasattr(self.method, "reset"):
                self.method.reset()
            metric_policy = _MetricPolicyWrapper(base_policy, visual_hook=visual_hook)
            start_time = time.perf_counter()
            _, successes, episode_infos = run_rollout_gymnasium_policy(
                env_name=task,
                policy=metric_policy,
                wrapper_configs=wrapper_configs,
                n_episodes=num_episodes,
                n_envs=n_envs,
                seed=seed,
            )
            elapsed_s = time.perf_counter() - start_time
            episode_time_s = elapsed_s / max(1, len(successes))
            per_episode = [
                {
                    "latencies_ms": metric_policy.action_latencies_ms,
                    "action_trace": metric_policy.action_trace,
                    "token_counts": metric_policy.token_counts,
                    "episode_time_s": episode_time_s,
                }
                for _ in successes
            ]

        episode_count = len(successes)
        success_rate = sum(bool(success) for success in successes) / max(1, episode_count)

        records = []
        for episode_id, success in enumerate(successes):
            ep = per_episode[episode_id]
            visual_metadata = self._aggregate_visual_metadata(visual_hook, ep["token_counts"])
            trace_path = self._trace_path(self.method_name, task, episode_id)
            baseline_trace_path = self._trace_path("baseline", task, episode_id)
            if self._should_save_actions():
                save_action_trace(
                    trace_path,
                    ep["action_trace"],
                    metadata={
                        "method": self.method_name,
                        "task": task,
                        "episode_id": episode_id,
                        "visual_metadata": visual_metadata,
                    },
                )
            action_l2 = self._compute_action_l2(trace_path, baseline_trace_path)

            record = start_episode_record(
                benchmark="LIBERO",
                method=self.method_name,
                task=task,
                episode_id=episode_id,
                keep_ratio=float(
                    getattr(self.method, "keep_ratio", getattr(self.args, "keep_ratio", 1.0))
                ),
                notes="real in-process LIBERO run",
            )
            records.append(
                finish_episode_record(
                    record,
                    success=bool(success),
                    success_rate=success_rate,
                    episode_time_s=_episode_time_from_infos(
                        episode_infos, episode_id, ep["episode_time_s"]
                    ),
                    action_latencies_ms=ep["latencies_ms"],
                    visual_metadata=visual_metadata,
                    action_l2_vs_baseline=action_l2,
                    failure_type="" if success else "episode_failed",
                    notes="real in-process LIBERO run",
                )
            )

        if stage_profiler is not None:
            try:
                self.last_stage_profile = stage_summary_with_gpu_memory(stage_profiler)
            finally:
                stage_detach()

        return records

    def _run_episodes_one_at_a_time(
        self,
        *,
        task: str,
        base_policy: Any,
        visual_hook: Any | None,
        wrapper_configs: Any,
        num_episodes: int,
        seed: int | None,
    ) -> tuple[list[bool], dict[str, Any], list[dict[str, Any]]]:
        """Run episodes one call at a time, resetting pruner state between them.

        ``run_rollout_gymnasium_policy`` calls ``policy.reset()`` exactly once,
        before its internal multi-episode loop begins (see its own comment:
        "we don't properly handle policy reset... policy are stateless").
        VLA-Pruner's temporal-momentum score is NOT stateless, so batching many
        episodes through one call let episode N's last score bleed into episode
        N+1's first few steps. Running one episode per call and resetting the
        method in between isolates each episode's temporal state, and as a
        side effect gives each episode its own action trace and latency
        samples (a fresh ``_MetricPolicyWrapper`` per episode) instead of every
        episode's saved trace containing the whole run's actions.

        Each episode advances the seed by its index so episodes remain varied
        (not the identical initial state replayed ``num_episodes`` times).
        """
        from gr00t.eval.rollout_policy import run_rollout_gymnasium_policy

        all_successes: list[bool] = []
        all_episode_infos: dict[str, list[Any]] = {}
        per_episode: list[dict[str, Any]] = []
        for episode_id in range(num_episodes):
            if self.method is not None and hasattr(self.method, "reset"):
                self.method.reset()
            episode_seed = None if seed is None else int(seed) + episode_id
            metric_policy = _MetricPolicyWrapper(base_policy, visual_hook=visual_hook)
            start_time = time.perf_counter()
            _, ep_successes, ep_infos = run_rollout_gymnasium_policy(
                env_name=task,
                policy=metric_policy,
                wrapper_configs=wrapper_configs,
                n_episodes=1,
                n_envs=1,
                seed=episode_seed,
            )
            episode_time_s = time.perf_counter() - start_time
            all_successes.extend(ep_successes)
            for key, values in ep_infos.items():
                all_episode_infos.setdefault(key, []).extend(values)
            per_episode.append(
                {
                    "latencies_ms": metric_policy.action_latencies_ms,
                    "action_trace": metric_policy.action_trace,
                    "token_counts": metric_policy.token_counts,
                    "episode_time_s": episode_time_s,
                }
            )
        return all_successes, all_episode_infos, per_episode

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
        return bool(
            getattr(self.args, "save_actions", False) or getattr(self.args, "save_traces", False)
        )

    @staticmethod
    def _aggregate_visual_metadata(
        visual_hook: Any | None,
        token_counts: list[tuple[Any, Any]],
    ) -> dict[str, Any] | None:
        """Merge per-step token counts into run-level hook metadata."""
        if visual_hook is None or not visual_hook.last_metadata:
            return None
        metadata = dict(visual_hook.last_metadata)
        befores = [b for b, _ in token_counts if isinstance(b, (int, float))]
        afters = [a for _, a in token_counts if isinstance(a, (int, float))]
        if befores and afters:
            metadata["original_tokens"] = sum(befores) / len(befores)
            metadata["kept_tokens"] = sum(afters) / len(afters)
            metadata["token_count_samples"] = len(token_counts)
        return metadata

    def _maybe_compile_model(self, model: Any | None) -> None:
        """torch.compile the DiT action head (the launch-overhead-dominated module).

        The vision tower is left eager (its rot_pos_emb is not dynamo-traceable in
        transformers 4.57), as are the LLM decoder layers (they carry the transformers
        hidden-states recorder and the backbone pruning loop). Profiling shows the DiT
        is ~66% of per-action latency, so it is where compilation pays.
        """
        if model is None or not getattr(self.args, "torch_compile", False):
            return
        import torch

        mode = getattr(self.args, "compile_mode", "reduce-overhead")
        action_head = getattr(model, "action_head", None)
        if action_head is not None and hasattr(action_head, "model"):
            action_head.model = torch.compile(action_head.model, mode=mode, dynamic=False)

    def _compute_action_l2(self, trace_path: Path, baseline_trace_path: Path) -> float:
        """Compare against the baseline trace only when both traces were written."""
        if self.method_name == "baseline":
            return 0.0
        if not (trace_path.exists() and baseline_trace_path.exists()):
            return 0.0
        return compute_action_l2_trace(
            load_action_trace(trace_path),
            load_action_trace(baseline_trace_path),
        )

    def _trace_path(self, method: str, task: str, episode_id: int) -> Path:
        safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task.replace("libero_sim/", ""))
        tag = method
        if method != "baseline":
            # Distinguish sweep runs so e.g. keep 0.75/0.6/0.5 don't overwrite each other.
            keep = getattr(self.method, "keep_ratio", getattr(self.args, "keep_ratio", None))
            if keep is not None:
                tag = f"{method}_kr{str(keep).replace('.', 'p')}"
        return self.real_output_dir / "traces" / f"{tag}_task_{safe_task}_episode_{episode_id}.npz"


class _MetricPolicyWrapper:
    """Small policy wrapper that records action latencies and action traces."""

    def __init__(self, policy: Any, visual_hook: Any | None = None) -> None:
        self.policy = policy
        self.visual_hook = visual_hook
        self.action_latencies_ms: list[float] = []
        self.action_trace: list[Any] = []
        self.token_counts: list[tuple[Any, Any]] = []

    def get_action(self, observation: dict[str, Any], options: dict[str, Any] | None = None) -> Any:
        start_time = time.perf_counter()
        action, info = self.policy.get_action(observation, options)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        record_action_latency(self.action_latencies_ms, elapsed_ms)
        self.action_trace.append(action)
        if self.visual_hook is not None and self.visual_hook.last_metadata:
            metadata = self.visual_hook.last_metadata
            # Hook B/C use original_tokens/kept_tokens; Hook A (after_visual_merger)
            # uses visual_token_count_before/nonzero_token_count_after (its shape
            # never changes, so "kept" means nonzero, not a shorter sequence).
            before = metadata.get("original_tokens", metadata.get("visual_token_count_before"))
            after = metadata.get("kept_tokens", metadata.get("nonzero_token_count_after"))
            self.token_counts.append((before, after))
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

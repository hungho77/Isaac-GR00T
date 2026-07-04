# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark runner abstraction for efficient inference experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gr00t.efficient.benchmark.methods import EfficientInferenceMethod
from gr00t.efficient.benchmark.metrics import BenchmarkRecord, summarize_metrics
from gr00t.efficient.profiler.latency import LatencyProfiler
from gr00t.efficient.profiler.memory import reset_peak_memory_stats


@dataclass
class _MockVisualTokens:
    """Small tensor-like object for dependency-free hook tests."""

    shape: tuple[int, ...]

    def __getitem__(self, item: Any) -> "_MockVisualTokens":
        if not isinstance(item, tuple):
            item = (item,)

        new_shape = list(self.shape)
        for axis, selector in enumerate(item):
            if axis >= len(new_shape):
                break
            if isinstance(selector, slice):
                start, stop, step = selector.indices(new_shape[axis])
                if step > 0:
                    length = max(0, (stop - start + step - 1) // step)
                else:
                    length = max(0, (start - stop - step - 1) // abs(step))
                new_shape[axis] = length
            elif isinstance(selector, int):
                new_shape[axis] = 1
            elif hasattr(selector, "__len__"):
                new_shape[axis] = len(selector)
        return _MockVisualTokens(tuple(new_shape))


class BenchmarkRunner:
    """Run benchmark modes through a selected efficient inference method."""

    def __init__(
        self,
        benchmark_name: str,
        method: EfficientInferenceMethod,
        profiler: LatencyProfiler | None = None,
    ) -> None:
        self.benchmark_name = benchmark_name
        self.method = method
        self.profiler = profiler or LatencyProfiler()

    def run_mock(
        self,
        num_episodes: int,
        task: str,
        visual_token_count: int = 256,
    ) -> list[BenchmarkRecord]:
        """Return deterministic mock records without importing LIBERO."""
        if num_episodes < 1:
            raise ValueError("num_episodes must be >= 1.")
        if visual_token_count < 1:
            raise ValueError("visual_token_count must be >= 1.")

        reset_peak_memory_stats()
        self.method.reset()
        records: list[BenchmarkRecord] = []
        latency_offset_ms = 0.0 if self.method.method_name == "baseline" else 1.0

        for episode_id in range(num_episodes):
            self.method.before_episode(episode_id=episode_id, task=task)
            visual_tokens = self._make_mock_visual_tokens(visual_token_count, episode_id)
            action_state, prev_action_state = self._make_mock_action_states(episode_id)
            _, hook_metadata = self.method.process_visual_tokens(
                visual_tokens,
                timestep=episode_id,
                episode_id=episode_id,
                task=task,
                action_state=action_state,
                prev_action_state=prev_action_state,
            )
            self.method.after_episode(episode_id=episode_id, task=task)

            kept_tokens = hook_metadata.get("kept_tokens")
            visual_token_count_after = int(kept_tokens) if kept_tokens is not None else None
            record = BenchmarkRecord(
                benchmark=self.benchmark_name,
                method=self.method.method_name,
                task=task,
                episode_id=episode_id,
                success=True,
                success_rate=1.0,
                latency_per_action_ms=42.0 + latency_offset_ms + (episode_id * 1.5),
                episode_time_s=12.0 + (episode_id * 0.25),
                gpu_memory_mb=4096.0,
                peak_gpu_memory_mb=5120.0,
                visual_token_count=visual_token_count_after,
                keep_ratio=self.method.keep_ratio,
                action_l2_vs_baseline=0.0,
                failure_type="",
                notes=f"mock record; hook_metadata={hook_metadata}",
            )
            records.append(self.method.update_metrics(record))

        return records

    def summarize(self, records: list[BenchmarkRecord]) -> dict[str, Any]:
        """Summarize benchmark records."""
        return summarize_metrics(records)

    @staticmethod
    def _make_mock_visual_tokens(visual_token_count: int, episode_id: int) -> Any:
        try:
            import torch
        except Exception:
            return _MockVisualTokens((1, visual_token_count, 64))

        generator = torch.Generator()
        generator.manual_seed(10_000 + episode_id)
        return torch.randn((1, visual_token_count, 64), generator=generator)

    @staticmethod
    def _make_mock_action_states(episode_id: int) -> tuple[Any | None, Any | None]:
        """Cycle deterministic action states for ADP scheduler mock coverage."""
        phase = episode_id % 3
        if phase == 0:
            return None, None

        prev = [0.0, 0.0, 0.0, 0.0]
        if phase == 1:
            return [0.2, 0.0, 0.0, 0.0], prev
        return [0.0, 0.0, 0.0, 1.0], prev

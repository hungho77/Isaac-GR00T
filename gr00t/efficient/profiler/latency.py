# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Latency profiling helpers with optional CUDA synchronization."""

from __future__ import annotations

from contextlib import contextmanager
from numbers import Number
import time
from typing import Iterator


def _cuda_synchronize_if_available() -> None:
    try:
        import torch
    except Exception:
        return

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


class LatencyProfiler:
    """Named wall-clock latency profiler for benchmark scaffolding."""

    def __init__(self, synchronize_cuda: bool = True) -> None:
        self.synchronize_cuda = synchronize_cuda
        self._starts_s: dict[str, float] = {}
        self.records_ms: dict[str, list[float]] = {}

    def start(self, name: str = "default") -> None:
        """Start timing a named section."""
        if self.synchronize_cuda:
            _cuda_synchronize_if_available()
        self._starts_s[name] = time.perf_counter()

    def stop(self, name: str = "default") -> float:
        """Stop timing a named section and record the elapsed milliseconds."""
        if name not in self._starts_s:
            raise RuntimeError(f"LatencyProfiler.stop({name!r}) called before start().")
        if self.synchronize_cuda:
            _cuda_synchronize_if_available()
        elapsed_ms = (time.perf_counter() - self._starts_s.pop(name)) * 1000.0
        self.record(name, elapsed_ms)
        return elapsed_ms

    def record(self, name: str, elapsed_ms: float) -> None:
        """Record an externally measured latency value."""
        if not isinstance(elapsed_ms, Number):
            raise TypeError("elapsed_ms must be numeric.")
        self.records_ms.setdefault(name, []).append(float(elapsed_ms))

    def summary(self) -> dict[str, dict[str, float | int | None]]:
        """Return count and aggregate latency stats for each named section."""
        result: dict[str, dict[str, float | int | None]] = {}
        for name, values in self.records_ms.items():
            if not values:
                result[name] = {
                    "count": 0,
                    "mean_ms": None,
                    "min_ms": None,
                    "max_ms": None,
                    "total_ms": 0.0,
                    "last_ms": None,
                }
                continue
            total_ms = sum(values)
            result[name] = {
                "count": len(values),
                "mean_ms": total_ms / len(values),
                "min_ms": min(values),
                "max_ms": max(values),
                "total_ms": total_ms,
                "last_ms": values[-1],
            }
        return result

    def reset(self) -> None:
        """Clear starts and recorded latency values."""
        self._starts_s.clear()
        self.records_ms.clear()

    @contextmanager
    def profile(self, name: str = "default") -> Iterator["LatencyProfiler"]:
        self.start(name)
        try:
            yield self
        finally:
            self.stop(name)

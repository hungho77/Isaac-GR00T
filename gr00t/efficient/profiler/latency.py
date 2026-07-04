# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Latency profiling helpers with optional CUDA synchronization."""

from __future__ import annotations

from contextlib import contextmanager
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
    """Minimal wall-clock latency profiler for benchmark scaffolding."""

    def __init__(self, synchronize_cuda: bool = True) -> None:
        self.synchronize_cuda = synchronize_cuda
        self._start_s: float | None = None
        self.last_elapsed_ms: float | None = None
        self.records_ms: list[float] = []

    def start(self) -> None:
        if self.synchronize_cuda:
            _cuda_synchronize_if_available()
        self._start_s = time.perf_counter()

    def stop(self) -> float:
        if self._start_s is None:
            raise RuntimeError("LatencyProfiler.stop() called before start().")
        if self.synchronize_cuda:
            _cuda_synchronize_if_available()
        elapsed_ms = (time.perf_counter() - self._start_s) * 1000.0
        self._start_s = None
        self.last_elapsed_ms = elapsed_ms
        self.records_ms.append(elapsed_ms)
        return elapsed_ms

    def reset(self) -> None:
        self._start_s = None
        self.last_elapsed_ms = None
        self.records_ms.clear()

    @contextmanager
    def profile(self) -> Iterator["LatencyProfiler"]:
        self.start()
        try:
            yield self
        finally:
            self.stop()


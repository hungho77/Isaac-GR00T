# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profiler helpers for efficient inference experiments."""

from gr00t.efficient.profiler.latency import LatencyProfiler
from gr00t.efficient.profiler.memory import get_cuda_memory_stats, reset_peak_memory_stats
from gr00t.efficient.profiler.model_stages import (
    STAGE_NAMES,
    attach_stage_profiler,
    stage_summary_with_gpu_memory,
)
from gr00t.efficient.profiler.token import get_token_shape, get_visual_token_count


__all__ = [
    "STAGE_NAMES",
    "LatencyProfiler",
    "attach_stage_profiler",
    "get_cuda_memory_stats",
    "get_token_shape",
    "get_visual_token_count",
    "reset_peak_memory_stats",
    "stage_summary_with_gpu_memory",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA memory helpers that tolerate missing torch or CUDA."""

from __future__ import annotations


def _empty_memory_stats() -> dict[str, float]:
    return {
        "allocated_mb": 0.0,
        "peak_allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "peak_reserved_mb": 0.0,
    }


def get_cuda_memory_stats() -> dict[str, float]:
    """Return CUDA memory stats in MB, or zeros when CUDA is unavailable."""
    try:
        import torch
    except Exception:
        return _empty_memory_stats()


def reset_peak_memory_stats() -> None:
    """Reset CUDA peak memory counters when torch and CUDA are available."""
    try:
        import torch
    except Exception:
        return

    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return

    try:
        if not torch.cuda.is_available():
            return _empty_memory_stats()

        bytes_per_mb = 1024.0 * 1024.0
        return {
            "allocated_mb": float(torch.cuda.memory_allocated()) / bytes_per_mb,
            "peak_allocated_mb": float(torch.cuda.max_memory_allocated()) / bytes_per_mb,
            "reserved_mb": float(torch.cuda.memory_reserved()) / bytes_per_mb,
            "peak_reserved_mb": float(torch.cuda.max_memory_reserved()) / bytes_per_mb,
        }
    except Exception:
        return _empty_memory_stats()

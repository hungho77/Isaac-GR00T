# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visual token counting helpers."""

from __future__ import annotations

from typing import Any


def get_token_shape(tensor: Any) -> tuple[int, ...] | None:
    """Return a tensor-like shape as a tuple of ints when available."""
    shape = getattr(tensor, "shape", None)
    if shape is not None:
        try:
            return tuple(int(dim) for dim in shape)
        except (TypeError, ValueError):
            return None

    try:
        return (len(tensor),)
    except TypeError:
        return None


def get_visual_token_count(tensor: Any) -> int | None:
    """Infer token count from common [B, N, D] or [N, D] tensor-like shapes."""
    shape = get_token_shape(tensor)
    if shape is None or len(shape) == 0:
        return None
    if len(shape) == 1:
        return shape[0]
    return shape[-2]

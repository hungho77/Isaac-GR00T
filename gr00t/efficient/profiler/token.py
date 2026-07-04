# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visual token counting helpers."""

from __future__ import annotations

from typing import Any


def get_visual_token_count(tensor: Any) -> int | None:
    """Infer token count from common [B, N, D] or [N, D] tensor-like shapes."""
    shape = getattr(tensor, "shape", None)
    if shape is not None:
        try:
            if len(shape) == 0:
                return None
            if len(shape) == 1:
                return int(shape[0])
            return int(shape[-2])
        except (TypeError, ValueError):
            return None

    try:
        return len(tensor)
    except TypeError:
        return None


# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Token selection utilities for visual-token pruning experiments."""

from __future__ import annotations

import math
import random
from typing import Any, Sequence


def compute_keep_count(num_tokens: int, keep_ratio: float) -> int:
    """Compute how many tokens to keep for a token count and keep ratio."""
    if num_tokens < 0:
        raise ValueError("num_tokens must be >= 0.")
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in the interval (0, 1].")
    if num_tokens == 0:
        return 0
    if keep_ratio >= 1.0:
        return num_tokens
    return max(1, min(num_tokens, int(math.ceil(num_tokens * keep_ratio))))


def first_k_indices(num_tokens: int, keep: int) -> list[int]:
    """Return the first K token indices."""
    _validate_index_args(num_tokens, keep)
    return list(range(keep))


def uniform_indices(num_tokens: int, keep: int) -> list[int]:
    """Return K approximately uniformly spaced token indices in original order."""
    _validate_index_args(num_tokens, keep)
    if keep == 0:
        return []
    if keep == num_tokens:
        return list(range(num_tokens))
    if keep == 1:
        return [0]

    indices = [round(i * (num_tokens - 1) / (keep - 1)) for i in range(keep)]
    unique_indices = []
    seen = set()
    for idx in indices:
        if idx not in seen:
            unique_indices.append(idx)
            seen.add(idx)

    if len(unique_indices) < keep:
        for idx in range(num_tokens):
            if idx not in seen:
                unique_indices.append(idx)
                seen.add(idx)
            if len(unique_indices) == keep:
                break

    return sorted(unique_indices[:keep])


def random_indices(num_tokens: int, keep: int, seed: int = 0) -> list[int]:
    """Return deterministic random token indices sorted in original order."""
    _validate_index_args(num_tokens, keep)
    if keep == 0:
        return []
    if keep == num_tokens:
        return list(range(num_tokens))
    rng = random.Random(seed)
    return sorted(rng.sample(range(num_tokens), keep))


def gather_tokens(visual_tokens: Any, indices: Any) -> Any:
    """Gather visual tokens along the token axis of a [B, N, D] tensor-like object."""
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(visual_tokens, torch.Tensor):
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=visual_tokens.device)
        if index_tensor.dim() == 1:
            _validate_indices(index_tensor.detach().cpu().tolist())
            return visual_tokens.index_select(1, index_tensor)
        if index_tensor.dim() == 2:
            if index_tensor.shape[0] != visual_tokens.shape[0]:
                raise ValueError(
                    "Per-batch token indices must have shape [B, K] matching visual_tokens."
                )
            if torch.any(index_tensor < 0):
                raise ValueError("Token indices must be >= 0.")
            gather_index = index_tensor.unsqueeze(-1).expand(-1, -1, visual_tokens.shape[-1])
            return visual_tokens.gather(dim=1, index=gather_index)
        raise ValueError("Token indices must be 1D [K] or 2D [B, K].")

    try:
        _validate_indices(indices)
        return visual_tokens[:, list(indices), :]
    except Exception as exc:
        raise TypeError(
            "Unsupported visual_tokens type for gather_tokens. Expected torch.Tensor or "
            "a tensor-like object supporting visual_tokens[:, indices, :]."
        ) from exc


def _validate_index_args(num_tokens: int, keep: int) -> None:
    if num_tokens < 0:
        raise ValueError("num_tokens must be >= 0.")
    if keep < 0:
        raise ValueError("keep must be >= 0.")
    if keep > num_tokens:
        raise ValueError("keep must be <= num_tokens.")


def _validate_indices(indices: Sequence[int]) -> None:
    for idx in indices:
        if not isinstance(idx, int):
            raise TypeError("Token indices must be integers.")
        if idx < 0:
            raise ValueError("Token indices must be >= 0.")

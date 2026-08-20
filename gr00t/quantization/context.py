# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Denoising-step context used by timestep-aware DiT quantization."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_STEP: ContextVar[int | None] = ContextVar("gr00t_holoq_dit_step", default=None)
_TOTAL_STEPS: ContextVar[int | None] = ContextVar("gr00t_holoq_dit_total_steps", default=None)


def get_dit_quant_step() -> int | None:
    """Return the active zero-based DiT denoising step, if any."""

    return _STEP.get()


def get_dit_quant_total_steps() -> int | None:
    """Return the number of steps in the active denoising loop, if any."""

    return _TOTAL_STEPS.get()


@contextmanager
def set_dit_quant_step(step: int, *, total_steps: int) -> Iterator[None]:
    """Set the active denoising step for quantized DiT linears.

    Context variables keep concurrent policy requests isolated and restore a
    surrounding context when the block exits.
    """

    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if not 0 <= step < total_steps:
        raise ValueError(f"step must be in [0, {total_steps}), got {step}")

    step_token = _STEP.set(step)
    total_token = _TOTAL_STEPS.set(total_steps)
    try:
        yield
    finally:
        _STEP.reset(step_token)
        _TOTAL_STEPS.reset(total_token)

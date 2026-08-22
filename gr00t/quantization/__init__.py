# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HoloQ-style uniform W4A4 support for GR00T-N1.7."""

from .context import get_dit_quant_step, get_dit_quant_total_steps, set_dit_quant_step
from .runtime import (
    HoloQLinear,
    HoloQPatchConv3d,
    NativeHoloQLinear,
    apply_holoq_pack,
    native_coverage,
)
from .scope import discover_n1d7_targets, normalize_scopes, validate_n1d7_scope


__all__ = [
    "HoloQLinear",
    "HoloQPatchConv3d",
    "NativeHoloQLinear",
    "apply_holoq_pack",
    "discover_n1d7_targets",
    "get_dit_quant_step",
    "get_dit_quant_total_steps",
    "normalize_scopes",
    "native_coverage",
    "set_dit_quant_step",
    "validate_n1d7_scope",
]

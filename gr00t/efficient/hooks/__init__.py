# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in hooks for efficient inference experiments."""

from gr00t.efficient.hooks.backbone_token_hook import (
    BackboneVisualTokenHook,
    attach_backbone_visual_token_hook,
)
from gr00t.efficient.hooks.visual_merger_hook import VisualMergerHook, attach_visual_merger_hook
from gr00t.efficient.hooks.visual_token_hook import VisualTokenHook, attach_visual_token_hook


__all__ = [
    "BackboneVisualTokenHook",
    "VisualMergerHook",
    "VisualTokenHook",
    "attach_backbone_visual_token_hook",
    "attach_visual_merger_hook",
    "attach_visual_token_hook",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in hooks for efficient inference experiments."""

from gr00t.efficient.hooks.visual_token_hook import VisualTokenHook, attach_visual_token_hook


__all__ = ["VisualTokenHook", "attach_visual_token_hook"]

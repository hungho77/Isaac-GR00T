# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UR10e + Robotiq cup-picking simulation support."""

from .ur10e_cup_env import ENV_ID, UR10eCupEnv, register_ur10e_cup_envs


__all__ = ["ENV_ID", "UR10eCupEnv", "register_ur10e_cup_envs"]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T modality configuration for khanhnd61/ur10e-cup.

The source dataset stores absolute 6-joint UR10e targets followed by the
Robotiq gripper command.  Keep that representation explicit: treating these
targets as deltas changes the hardware control contract.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


ur10e_cup_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["side", "wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["arm_joints", "gripper"],
        # Joint angles are periodic.  Sin/cos encoding also avoids amplifying
        # the nearly constant wrist_2 channel through min/max normalization.
        sin_cos_embedding_keys=["arm_joints"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["arm_joints", "gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# Custom hardware must use the checkpoint's reserved custom-embodiment slot.
register_modality_config(ur10e_cup_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)

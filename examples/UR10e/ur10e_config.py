# SPDX-License-Identifier: Apache-2.0
"""Modality config for the UR10e cup-manipulation dataset (khanhnd61/ur10e-cup).

Index ranges below mirror meta/modality.json and are derived from the dataset's
own `meta/info.json`, where both `observation.state` and `action` are float32[7]
named:

    [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, gripper]

so joints occupy 0:6 and the gripper is the trailing channel 6:7. Getting these
bounds wrong does not raise -- the model would simply learn the wrong channels --
so they must stay in sync with modality.json.
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


ur10e_config = {
    # Two cameras, matching the "video" entries in meta/modality.json.
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["side", "wrist"],  # third-person view + wrist egocentric
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "single_arm",  # 6 joint positions
            "gripper",
        ],
    ),
    "action": ModalityConfig(
        # 16-step prediction horizon. The dataset is 20 fps, so this is 0.8 s of
        # future motion per inference.
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "single_arm",
            "gripper",
        ],
        action_configs=[
            # Joints: RELATIVE (delta from current state) generalises better than
            # predicting absolute joint targets.
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,  # joint-space, not end-effector
                format=ActionFormat.DEFAULT,
            ),
            # Gripper: ABSOLUTE -- a near-binary open/close signal is better
            # predicted as a target position than as a delta.
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

register_modality_config(ur10e_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)

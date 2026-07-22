from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


aloha_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "wrist"]
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "single_arm",
            "gripper",
        ]
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=[
            "single_arm",
            "gripper",
        ],
        action_configs=[
            # single_arm
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT
            ),
            # gripper
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT
            ),
        ]
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"]
    )
}

register_modality_config(aloha_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)



# aloha_config = {
#     "video": ModalityConfig(
#         delta_indices=[0],
#         modality_keys=["front", "wrist"]
#     ),
#     "state": ModalityConfig(
#         delta_indices=[0],
#         modality_keys=[
#             "right_arm",
#         ]
#     ),
#     "action": ModalityConfig(
#         delta_indices=list(range(16)),
#         modality_keys=[
#             "right_arm",
#         ],
#         action_configs=[
#             # right_arm
#             ActionConfig(
#                 rep=ActionRepresentation.ABSOLUTE,
#                 type=ActionType.NON_EEF,
#                 format=ActionFormat.DEFAULT
#             ),
#         ]
#     ),
#     "language": ModalityConfig(
#         delta_indices=[0],
#         modality_keys=["annotation.human.task_description"]
#     )
# }

# register_modality_config(aloha_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
 


# aloha_config = {
#     "video": ModalityConfig(
#         delta_indices=[0],
#         modality_keys=["front", "wrist_right", "wrist_left"]
#     ),
#     "state": ModalityConfig(
#         delta_indices=[0],
#         modality_keys=[
#             "left_arm",
#             "right_arm",
#         ]
#     ),
#     "action": ModalityConfig(
#         delta_indices=list(range(0, 32, 2)),
#         modality_keys=[
#             "left_arm",
#             "right_arm",
#         ],
#         action_configs=[
#             # left_arm
#             ActionConfig(
#                 rep=ActionRepresentation.RELATIVE,
#                 type=ActionType.NON_EEF,
#                 format=ActionFormat.DEFAULT
#             ),
#             # right_arm
#             ActionConfig(
#                 rep=ActionRepresentation.RELATIVE,
#                 type=ActionType.NON_EEF,
#                 format=ActionFormat.DEFAULT
#             ),
#         ]
#     ),
#     "language": ModalityConfig(
#         delta_indices=[0],
#         modality_keys=["annotation.human.task_description"]
#     )
# }

# register_modality_config(aloha_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
 
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration and fidelity contract for the UR10e Cup simulator.

The public dataset identifies the gripper only as ``Robotiq 2F`` and does not
publish the camera or robot-to-table calibration.  Consequently, the bundled
scene is deliberately labelled a proxy.  A calibrated result must opt in with
an external MJCF and explicit verification flags; this prevents proxy success
rate from being reported as real-robot success rate by accident.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

import numpy as np


DATASET_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)
MUJOCO_JOINT_NAMES = tuple(f"{name}_joint" for name in DATASET_JOINT_NAMES)

# Observed limits from khanhnd61/ur10e-cup.  These are dataset-support limits,
# not the wider UR10e hardware limits.  Staying inside them is intentional for
# an evaluation proxy trained from this single-scene dataset.
DATASET_ARM_LOW = np.array([-1.6301, -1.9923, -1.6848, -1.6242, 1.5702, -3.2003], dtype=np.float64)
DATASET_ARM_HIGH = np.array([-1.0360, -1.5708, -1.2006, -1.1484, 1.5716, -2.6050], dtype=np.float64)
DATASET_ARM_RESET = (DATASET_ARM_LOW + DATASET_ARM_HIGH) / 2.0
WRIST_2_INDEX = DATASET_JOINT_NAMES.index("wrist_2")


@dataclass(frozen=True)
class UR10eCupSimConfig:
    """Runtime configuration for :class:`UR10eCupEnv`."""

    simulation_fidelity: str = "proxy"
    model_xml_path: str | None = None
    gripper_model: str = "unspecified_robotiq_2f_proxy"
    control_hz: int = 20
    physics_timestep: float = 0.002
    render_height: int = 480
    render_width: int = 640
    wrist_2_policy: str = "clamp_current"
    approach_distance_m: float = 0.12
    lift_height_m: float = 0.05
    stable_success_steps: int = 5
    reset_settle_control_steps: int = 20
    dataset_support_tolerance_rad: float = 0.02
    hardware_joint_limit_tolerance_rad: float = 0.01
    cup_reset_xy: tuple[float, float] = (-0.42, -0.30)
    cup_reset_xy_noise: float = 0.015
    cup_reset_z: float = 0.105
    task_description: str = "pick up the cup"
    tool_transform_verified: bool = False
    camera_calibration_verified: bool = False
    gripper_calibration_verified: bool = False
    scene_calibration_verified: bool = False

    @property
    def physics_substeps(self) -> int:
        value = 1.0 / (self.control_hz * self.physics_timestep)
        rounded = round(value)
        if not np.isclose(value, rounded, atol=1e-9):
            raise ValueError(
                "control_hz and physics_timestep must yield an integer number of substeps; "
                f"got {value}"
            )
        return int(rounded)

    @property
    def calibration_verified(self) -> bool:
        return all(
            (
                self.tool_transform_verified,
                self.camera_calibration_verified,
                self.gripper_calibration_verified,
                self.scene_calibration_verified,
            )
        )

    def validate(self) -> None:
        if self.simulation_fidelity not in {"proxy", "calibrated"}:
            raise ValueError("simulation_fidelity must be 'proxy' or 'calibrated'")
        if self.control_hz != 20:
            raise ValueError("UR10e Cup data was recorded at 20 Hz; control_hz must remain 20")
        if self.wrist_2_policy != "clamp_current":
            raise ValueError("wrist_2_policy must be 'clamp_current' for this dataset")
        if self.render_height <= 0 or self.render_width <= 0:
            raise ValueError("render dimensions must be positive")
        if self.stable_success_steps < 1:
            raise ValueError("stable_success_steps must be >= 1")
        if self.approach_distance_m <= 0:
            raise ValueError("approach_distance_m must be positive")
        if self.reset_settle_control_steps < 1:
            raise ValueError("reset_settle_control_steps must be >= 1")
        if self.dataset_support_tolerance_rad < 0:
            raise ValueError("dataset_support_tolerance_rad must be >= 0")
        if self.hardware_joint_limit_tolerance_rad < 0:
            raise ValueError("hardware_joint_limit_tolerance_rad must be >= 0")
        if self.physics_timestep <= 0:
            raise ValueError("physics_timestep must be positive")
        _ = self.physics_substeps
        if self.simulation_fidelity == "calibrated":
            if not self.model_xml_path:
                raise ValueError("calibrated fidelity requires model_xml_path")
            if not self.calibration_verified:
                raise ValueError(
                    "calibrated fidelity requires verified tool, camera, gripper, and scene calibration"
                )

    def to_dict(self) -> dict:
        result = asdict(self)
        result["physics_substeps"] = self.physics_substeps
        result["calibration_verified"] = self.calibration_verified
        return result


def load_sim_config(path: str | Path | None = None) -> UR10eCupSimConfig:
    """Load configuration from JSON or return safe proxy defaults."""

    path = path or os.environ.get("GR00T_UR10E_SIM_CONFIG")
    if path is None:
        config = UR10eCupSimConfig()
    else:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "cup_reset_xy" in payload:
            payload["cup_reset_xy"] = tuple(payload["cup_reset_xy"])
        config = UR10eCupSimConfig(**payload)
    config.validate()
    return config

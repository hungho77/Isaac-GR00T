# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gymnasium MuJoCo proxy for the UR10e + Robotiq Cup dataset.

The environment preserves the dataset I/O contract exactly (two 480x640 RGB
cameras, six absolute joint targets, and a binary gripper target).  The bundled
MJCF is a portable engineering proxy, not a calibrated digital twin.  Use a
calibration JSON and external MJCF before interpreting its success rate as a
sim-to-real claim.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register, registry
import numpy as np

from .sim_config import (
    DATASET_ARM_HIGH,
    DATASET_ARM_LOW,
    DATASET_ARM_RESET,
    DATASET_JOINT_NAMES,
    MUJOCO_JOINT_NAMES,
    WRIST_2_INDEX,
    UR10eCupSimConfig,
    load_sim_config,
)


ENV_ID = "ur10e_cup_sim/pick_up_the_cup"

# Self-contained proxy MJCF.  Link lengths follow the UR10e scale, while the
# visual/collision geometry, Robotiq fingers, camera extrinsics and table scene
# remain approximate because the public dataset does not publish calibration.
_PROXY_MJCF = """
<mujoco model="ur10e_cup_proxy">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="RK4" gravity="0 0 -9.81"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <default>
    <joint damping="4" armature="0.05"/>
    <geom condim="4" friction="1.0 0.01 0.001"/>
    <!-- Gravity compensation prevents static pose drift; these moderate
         gains remain stable at the 2-ms RK4 physics timestep. -->
    <position kp="180" kv="28"/>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.18 0.18 0.18"
             rgb2="0.24 0.24 0.24" width="256" height="256"/>
    <material name="table_mat" texture="grid" texrepeat="4 4"/>
  </asset>
  <worldbody>
    <light pos="0 -1.2 2.0" dir="0 0.5 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.15 0.15 0.15 1"/>
    <geom name="table" type="box" pos="-0.45 -0.30 0.025" size="0.55 0.55 0.025"
          material="table_mat"/>
    <camera name="side" pos="1.10 -1.25 1.05" xyaxes="0.76 0.65 0 -0.35 0.41 0.84"/>
    <body name="base" pos="0 0 0.08">
      <geom type="cylinder" size="0.10 0.08" contype="0" conaffinity="0"
            rgba="0.15 0.45 0.75 1"/>
      <body name="shoulder" pos="0 0 0.1273" gravcomp="1">
        <joint name="shoulder_pan_joint" axis="0 0 1" range="-6.283 6.283"/>
        <geom type="cylinder" size="0.09 0.10" contype="0" conaffinity="0"
              rgba="0.2 0.55 0.85 1"/>
        <body name="upper_arm" pos="0 0 0.176" gravcomp="1">
          <joint name="shoulder_lift_joint" axis="0 1 0" range="-6.283 6.283"/>
          <geom type="capsule" fromto="0 0 0 0.612 0 0" size="0.065"
                contype="0" conaffinity="0" rgba="0.55 0.65 0.72 1"/>
          <body name="forearm" pos="0.612 0 0" gravcomp="1">
            <joint name="elbow_joint" axis="0 1 0" range="-3.142 3.142"/>
            <geom type="capsule" fromto="0 0 0 0.5723 0 0" size="0.055"
                  contype="0" conaffinity="0" rgba="0.55 0.65 0.72 1"/>
            <body name="wrist_1_link" pos="0.5723 0 0" gravcomp="1">
              <joint name="wrist_1_joint" axis="0 1 0" range="-6.283 6.283"/>
              <geom type="cylinder" size="0.06 0.08" contype="0" conaffinity="0"
                    rgba="0.2 0.55 0.85 1"/>
              <body name="wrist_2_link" pos="0 0 0.1639" gravcomp="1">
                <joint name="wrist_2_joint" axis="0 0 1" range="-6.283 6.283"/>
                <geom type="cylinder" size="0.055 0.07" contype="0" conaffinity="0"
                      rgba="0.2 0.55 0.85 1"/>
                <body name="wrist_3_link" pos="0 0 0.1157" gravcomp="1">
                  <joint name="wrist_3_joint" axis="0 1 0" range="-6.283 6.283"/>
                  <geom type="cylinder" size="0.05 0.07" contype="0" conaffinity="0"
                        rgba="0.2 0.55 0.85 1"/>
                  <body name="tool" pos="0 0 0.11" gravcomp="1">
                    <!-- MuJoCo cameras look along local -Z.  Mounting the
                         wrist camera on the tool with +Y as image-up keeps
                         the fingers and workspace in view. -->
                    <camera name="wrist" pos="0.055 0 0.015" xyaxes="1 0 0 0 1 0"/>
                    <geom name="palm" type="box" size="0.035 0.055 0.025"
                          contype="0" conaffinity="0" rgba="0.12 0.12 0.12 1"/>
                    <body name="left_finger" pos="0 0.012 -0.065">
                      <joint name="left_finger_joint" type="slide" axis="0 1 0" range="0 0.035"/>
                      <geom name="left_finger_pad" type="box" size="0.012 0.008 0.055"
                            rgba="0.08 0.08 0.08 1"/>
                    </body>
                    <body name="right_finger" pos="0 -0.012 -0.065">
                      <joint name="right_finger_joint" type="slide" axis="0 -1 0" range="0 0.035"/>
                      <geom name="right_finger_pad" type="box" size="0.012 0.008 0.055"
                            rgba="0.08 0.08 0.08 1"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="cup" pos="-0.42 -0.30 0.105">
      <freejoint name="cup_freejoint"/>
      <geom name="cup_geom" type="cylinder" size="0.035 0.055" mass="0.12"
            rgba="0.85 0.15 0.12 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan" joint="shoulder_pan_joint" ctrlrange="-1.6301 -1.0360"/>
    <position name="shoulder_lift" joint="shoulder_lift_joint" ctrlrange="-1.9923 -1.5708"/>
    <position name="elbow" joint="elbow_joint" ctrlrange="-1.6848 -1.2006"/>
    <position name="wrist_1" joint="wrist_1_joint" ctrlrange="-1.6242 -1.1484"/>
    <position name="wrist_2" joint="wrist_2_joint" ctrlrange="1.5702 1.5716"/>
    <position name="wrist_3" joint="wrist_3_joint" ctrlrange="-3.2003 -2.6050"/>
    <position name="left_finger" joint="left_finger_joint" ctrlrange="0 0.035" kp="100" kv="10"/>
    <position name="right_finger" joint="right_finger_joint" ctrlrange="0 0.035" kp="100" kv="10"/>
  </actuator>
</mujoco>
"""


class UR10eCupEnv(gym.Env):
    """Dataset-contract-preserving MuJoCo environment for cup picking."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, config_path: str | None = None):
        os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "glfw")
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError(
                "UR10eCupEnv requires MuJoCo. Run "
                "bash gr00t/eval/sim/UR10eCup/setup_ur10e_cup_sim.sh"
            ) from exc

        self._mujoco = mujoco
        self.config: UR10eCupSimConfig = load_sim_config(config_path)
        if self.config.model_xml_path:
            model_path = Path(self.config.model_xml_path).expanduser().resolve()
            if not model_path.is_file():
                raise FileNotFoundError(f"UR10e simulator MJCF not found: {model_path}")
            self.model = mujoco.MjModel.from_xml_path(str(model_path))
        else:
            self.model = mujoco.MjModel.from_xml_string(_PROXY_MJCF)
        self.model.opt.timestep = self.config.physics_timestep
        self.data = mujoco.MjData(self.model)
        self._renderer = mujoco.Renderer(
            self.model,
            height=self.config.render_height,
            width=self.config.render_width,
        )
        self._joint_ids = np.array(
            [self._name_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in MUJOCO_JOINT_NAMES]
        )
        self._joint_qpos_addresses = self.model.jnt_qposadr[self._joint_ids].astype(int)
        self._joint_dof_addresses = self.model.jnt_dofadr[self._joint_ids].astype(int)
        joint_limited = self.model.jnt_limited[self._joint_ids].astype(bool)
        joint_ranges = self.model.jnt_range[self._joint_ids]
        self._hardware_arm_low = np.where(joint_limited, joint_ranges[:, 0], -np.inf)
        self._hardware_arm_high = np.where(joint_limited, joint_ranges[:, 1], np.inf)
        self._cup_joint_id = self._name_id(mujoco.mjtObj.mjOBJ_JOINT, "cup_freejoint")
        self._cup_qpos_address = int(self.model.jnt_qposadr[self._cup_joint_id])
        self._cup_body_id = self._name_id(mujoco.mjtObj.mjOBJ_BODY, "cup")
        self._tool_body_id = self._name_id(mujoco.mjtObj.mjOBJ_BODY, "tool")
        self._geom_ids = {
            name: self._name_id(mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("left_finger_pad", "right_finger_pad", "cup_geom", "table", "floor")
        }
        self._initial_cup_z = self.config.cup_reset_z
        self._success_streak = 0
        self._step_count = 0
        self._last_arm_target = DATASET_ARM_RESET.copy()
        self._last_gripper_target = 1.0
        self._episode_max_cup_height = self.config.cup_reset_z

        self.observation_space = spaces.Dict(
            {
                "video.side": spaces.Box(
                    0,
                    255,
                    shape=(self.config.render_height, self.config.render_width, 3),
                    dtype=np.uint8,
                ),
                "video.wrist": spaces.Box(
                    0,
                    255,
                    shape=(self.config.render_height, self.config.render_width, 3),
                    dtype=np.uint8,
                ),
                "state.arm_joints": spaces.Box(
                    low=np.full(6, -2 * np.pi, dtype=np.float32),
                    high=np.full(6, 2 * np.pi, dtype=np.float32),
                    dtype=np.float32,
                ),
                "state.gripper": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "annotation.human.task_description": spaces.Text(
                    max_length=128,
                    charset="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-",
                ),
            }
        )
        self.action_space = spaces.Dict(
            {
                "action.arm_joints": spaces.Box(
                    low=DATASET_ARM_LOW.astype(np.float32),
                    high=DATASET_ARM_HIGH.astype(np.float32),
                    dtype=np.float32,
                ),
                "action.gripper": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            }
        )

    def _name_id(self, object_type: Any, name: str) -> int:
        object_id = self._mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            object_name = object_type.name.removeprefix("mjOBJ_").lower()
            raise ValueError(
                f"MJCF is missing required {object_name} {name!r}"
            )
        return int(object_id)

    def _render_camera(self, camera: str) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=camera)
        return np.asarray(self._renderer.render(), dtype=np.uint8).copy()

    def _gripper_open_fraction(self) -> float:
        joint_id = self._name_id(self._mujoco.mjtObj.mjOBJ_JOINT, "left_finger_joint")
        qpos = float(self.data.qpos[self.model.jnt_qposadr[joint_id]])
        return float(np.clip(qpos / 0.035, 0.0, 1.0))

    def _observation(self) -> dict[str, Any]:
        return {
            "video.side": self._render_camera("side"),
            "video.wrist": self._render_camera("wrist"),
            "state.arm_joints": self.data.qpos[self._joint_qpos_addresses]
            .astype(np.float32)
            .copy(),
            "state.gripper": np.array([self._gripper_open_fraction()], dtype=np.float32),
            "annotation.human.task_description": self.config.task_description,
        }

    def _contact_pairs(self) -> set[frozenset[int]]:
        return {
            frozenset((int(contact.geom1), int(contact.geom2)))
            for contact in self.data.contact[: self.data.ncon]
        }

    def _task_metrics(self) -> dict[str, Any]:
        contacts = self._contact_pairs()
        cup = self._geom_ids["cup_geom"]
        left_contact = frozenset((cup, self._geom_ids["left_finger_pad"])) in contacts
        right_contact = frozenset((cup, self._geom_ids["right_finger_pad"])) in contacts
        table_support = any(
            frozenset((cup, self._geom_ids[name])) in contacts for name in ("table", "floor")
        )
        cup_height = float(self.data.xpos[self._cup_body_id, 2])
        self._episode_max_cup_height = max(self._episode_max_cup_height, cup_height)
        tool_position = self.data.xpos[self._tool_body_id].copy()
        cup_position = self.data.xpos[self._cup_body_id].copy()
        tool_cup_distance = float(np.linalg.norm(tool_position - cup_position))
        lifted = cup_height >= self._initial_cup_z + self.config.lift_height_m
        stable_now = left_contact and right_contact and lifted and not table_support
        self._success_streak = self._success_streak + 1 if stable_now else 0
        success = self._success_streak >= self.config.stable_success_steps
        arm = self.data.qpos[self._joint_qpos_addresses].copy()
        arm_velocity = self.data.qvel[self._joint_dof_addresses].copy()
        finite_mask = np.isfinite(arm) & np.isfinite(arm_velocity)
        dataset_tolerance = self.config.dataset_support_tolerance_rad
        dataset_violation_mask = finite_mask & (
            (arm < DATASET_ARM_LOW - dataset_tolerance)
            | (arm > DATASET_ARM_HIGH + dataset_tolerance)
        )
        hardware_tolerance = self.config.hardware_joint_limit_tolerance_rad
        hardware_violation_mask = (~finite_mask) | (
            arm < self._hardware_arm_low - hardware_tolerance
        ) | (arm > self._hardware_arm_high + hardware_tolerance)
        dataset_violation_joints = [
            name
            for name, violated in zip(DATASET_JOINT_NAMES, dataset_violation_mask, strict=True)
            if violated
        ]
        hardware_violation_joints = [
            name
            for name, violated in zip(DATASET_JOINT_NAMES, hardware_violation_mask, strict=True)
            if violated
        ]
        dataset_support_violation = bool(np.any(dataset_violation_mask))
        hardware_safety_violation = bool(
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
            or np.any(hardware_violation_mask)
        )
        return {
            "success": bool(success),
            "left_finger_contact": bool(left_contact),
            "right_finger_contact": bool(right_contact),
            "table_support": bool(table_support),
            "cup_height_m": cup_height,
            "max_cup_height_m": float(self._episode_max_cup_height),
            "tool_position_m": tool_position.tolist(),
            "cup_position_m": cup_position.tolist(),
            "tool_cup_distance_m": tool_cup_distance,
            "approached_cup": tool_cup_distance <= self.config.approach_distance_m,
            "lifted": bool(lifted),
            "stable_success_steps": int(self._success_streak),
            # Leaving the demonstration envelope is an out-of-distribution
            # warning, not a hardware fault.  Only non-finite state or a true
            # MJCF joint-limit breach may terminate an episode.
            "dataset_support_violation": dataset_support_violation,
            "dataset_support_violation_joints": dataset_violation_joints,
            "hardware_safety_violation": hardware_safety_violation,
            "hardware_safety_violation_joints": hardware_violation_joints,
            # Backwards-compatible alias; it now means hardware safety only.
            "safety_violation": hardware_safety_violation,
            "arm_qpos_rad": arm.tolist(),
            "arm_qvel_rad_s": arm_velocity.tolist(),
            "arm_target_rad": self._last_arm_target.tolist(),
            "arm_target_delta_rad": (self._last_arm_target - arm).tolist(),
            "gripper_open_fraction": self._gripper_open_fraction(),
            "gripper_target": float(self._last_gripper_target),
            "step_count": int(self._step_count),
            "simulation_fidelity": self.config.simulation_fidelity,
            "calibration_verified": self.config.calibration_verified,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        self._mujoco.mj_resetData(self.model, self.data)
        if "arm_qpos" in options:
            arm = np.asarray(options["arm_qpos"], dtype=np.float64).reshape(6)
            if not np.isfinite(arm).all():
                raise ValueError("reset arm_qpos contains non-finite values")
            arm = np.clip(arm, DATASET_ARM_LOW, DATASET_ARM_HIGH)
        else:
            arm_noise = self.np_random.uniform(-0.002, 0.002, size=6)
            arm_noise[WRIST_2_INDEX] = 0.0
            arm = np.clip(DATASET_ARM_RESET + arm_noise, DATASET_ARM_LOW, DATASET_ARM_HIGH)
        self.data.qpos[self._joint_qpos_addresses] = arm
        self.data.ctrl[:6] = arm
        self._last_arm_target = arm.copy()
        gripper_open_fraction = float(options.get("gripper_open_fraction", 1.0))
        if not np.isfinite(gripper_open_fraction):
            raise ValueError("reset gripper_open_fraction must be finite")
        gripper_open_fraction = float(np.clip(gripper_open_fraction, 0.0, 1.0))
        self._last_gripper_target = gripper_open_fraction
        self.data.ctrl[6:8] = 0.035 * gripper_open_fraction
        if "cup_xy" in options:
            cup_xy = np.asarray(options["cup_xy"], dtype=np.float64).reshape(2)
            if not np.isfinite(cup_xy).all():
                raise ValueError("reset cup_xy contains non-finite values")
        else:
            cup_xy = np.asarray(self.config.cup_reset_xy) + self.np_random.uniform(
                -self.config.cup_reset_xy_noise,
                self.config.cup_reset_xy_noise,
                size=2,
            )
        cup_pose = self.data.qpos[self._cup_qpos_address : self._cup_qpos_address + 7]
        cup_pose[:3] = (cup_xy[0], cup_xy[1], self.config.cup_reset_z)
        cup_pose[3:] = (1.0, 0.0, 0.0, 0.0)
        self._mujoco.mj_forward(self.model, self.data)
        # Let the cup establish table contact and let the gravity-compensated
        # arm servo converge before exposing the first observation.  Settling
        # is initialization and therefore does not consume episode steps.
        for _ in range(self.config.reset_settle_control_steps * self.config.physics_substeps):
            self._mujoco.mj_step(self.model, self.data)
        self._initial_cup_z = float(self.data.xpos[self._cup_body_id, 2])
        self._episode_max_cup_height = self._initial_cup_z
        self._success_streak = 0
        self._step_count = 0
        info = self._task_metrics()
        if info["hardware_safety_violation"]:
            raise RuntimeError(
                "UR10e proxy violated a hardware joint limit while settling: "
                f"{info['hardware_safety_violation_joints']}; "
                f"qpos={info['arm_qpos_rad']}; target={info['arm_target_rad']}"
            )
        return self._observation(), info

    def step(self, action: dict[str, np.ndarray]):
        arm_target = np.asarray(action["action.arm_joints"], dtype=np.float64).reshape(6)
        gripper_target = float(np.asarray(action["action.gripper"]).reshape(-1)[0])
        if not np.isfinite(arm_target).all() or not np.isfinite(gripper_target):
            raise ValueError("UR10e action contains non-finite values")
        arm_target = np.clip(arm_target, DATASET_ARM_LOW, DATASET_ARM_HIGH)
        # wrist_2 is nearly constant in all demonstrations (std ~= 8e-4 rad).
        # Preserve tensor shape but do not command extrapolated motion.
        arm_target[WRIST_2_INDEX] = self.data.qpos[self._joint_qpos_addresses[WRIST_2_INDEX]]
        self._last_arm_target = arm_target.copy()
        self._last_gripper_target = float(np.clip(gripper_target, 0.0, 1.0))
        self.data.ctrl[:6] = arm_target
        self.data.ctrl[6:8] = 0.035 * self._last_gripper_target
        for _ in range(self.config.physics_substeps):
            self._mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        info = self._task_metrics()
        terminated = bool(info["success"] or info["hardware_safety_violation"])
        reward = float(info["success"])
        return self._observation(), reward, terminated, False, info

    def render(self):
        return self._render_camera("side")

    def close(self):
        self._renderer.close()


def register_ur10e_cup_envs() -> None:
    """Register the UR10e Cup environment idempotently."""

    if ENV_ID not in registry:
        register(
            id=ENV_ID,
            entry_point="gr00t.eval.sim.UR10eCup.ur10e_cup_env:UR10eCupEnv",
        )

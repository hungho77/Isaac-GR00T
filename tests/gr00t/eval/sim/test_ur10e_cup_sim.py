# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe contract tests for the optional UR10e Cup simulator."""

import json

from gr00t.eval.sim.UR10eCup.sim_config import (
    DATASET_ARM_HIGH,
    DATASET_ARM_LOW,
    DATASET_ARM_RESET,
    DATASET_JOINT_NAMES,
    MUJOCO_JOINT_NAMES,
    WRIST_2_INDEX,
    UR10eCupSimConfig,
    load_sim_config,
)
import numpy as np
import pytest


def test_dataset_joint_mapping_is_name_based_and_ordered():
    assert MUJOCO_JOINT_NAMES == tuple(f"{name}_joint" for name in DATASET_JOINT_NAMES)
    assert DATASET_JOINT_NAMES[WRIST_2_INDEX] == "wrist_2"
    assert np.all(DATASET_ARM_LOW < DATASET_ARM_HIGH)


def test_proxy_contract_is_20_hz_and_explicitly_uncalibrated():
    config = UR10eCupSimConfig()
    config.validate()
    assert config.physics_substeps == 25
    assert config.approach_distance_m == pytest.approx(0.12)
    assert config.reset_settle_control_steps == 20
    assert config.dataset_support_tolerance_rad == pytest.approx(0.02)
    assert config.hardware_joint_limit_tolerance_rad == pytest.approx(0.01)
    assert config.simulation_fidelity == "proxy"
    assert config.calibration_verified is False


def test_calibrated_mode_rejects_missing_verification():
    with pytest.raises(ValueError, match="verified tool, camera, gripper, and scene"):
        UR10eCupSimConfig(
            simulation_fidelity="calibrated",
            model_xml_path="robot.xml",
        ).validate()


def test_load_config_round_trip(tmp_path):
    config_path = tmp_path / "sim.json"
    config_path.write_text(
        json.dumps({"cup_reset_xy": [-0.4, -0.2], "stable_success_steps": 7}),
        encoding="utf-8",
    )
    config = load_sim_config(config_path)
    assert config.cup_reset_xy == (-0.4, -0.2)
    assert config.stable_success_steps == 7


def test_optional_mujoco_safe_hold_and_safety_taxonomy():
    mujoco = pytest.importorskip("mujoco")
    del mujoco
    from gr00t.eval.sim.UR10eCup.ur10e_cup_env import UR10eCupEnv

    env = UR10eCupEnv()
    try:
        observation, info = env.reset(seed=7)
        assert observation["video.side"].shape == (480, 640, 3)
        assert observation["video.wrist"].shape == (480, 640, 3)
        assert float(observation["video.wrist"].mean()) > 1.0
        assert float(observation["video.wrist"].std()) > 1.0
        assert observation["state.arm_joints"].shape == (6,)
        assert observation["state.gripper"].shape == (1,)
        assert observation["annotation.human.task_description"] == "pick up the cup"
        assert info["simulation_fidelity"] == "proxy"
        assert info["calibration_verified"] is False
        assert info["table_support"] is True
        assert info["hardware_safety_violation"] is False
        assert info["safety_violation"] is False
        hold_arm = observation["state.arm_joints"].copy()
        action = {
            "action.arm_joints": hold_arm,
            "action.gripper": np.array([1.0], dtype=np.float32),
        }
        for _ in range(400):
            next_observation, reward, terminated, truncated, next_info = env.step(action)
            assert terminated is False, next_info
            assert truncated is False
            assert next_info["hardware_safety_violation"] is False, next_info
        assert env.observation_space.contains(next_observation)
        assert reward in {0.0, 1.0}
        assert isinstance(next_info["success"], bool)
        assert next_info["step_count"] == 400
        assert len(next_info["arm_qpos_rad"]) == 6
        assert len(next_info["arm_target_rad"]) == 6
        assert len(next_info["arm_target_delta_rad"]) == 6
        assert len(next_info["tool_position_m"]) == 3
        assert len(next_info["cup_position_m"]) == 3
        assert np.isfinite(next_info["tool_cup_distance_m"])
        assert next_info["max_cup_height_m"] >= next_info["cup_height_m"] - 1e-9
        assert next_info["gripper_target"] == pytest.approx(1.0)

        replay_arm = DATASET_ARM_RESET + np.array([0.001, 0, 0, 0, 0, 0])
        replay_observation, replay_info = env.reset(
            seed=8,
            options={"arm_qpos": replay_arm, "gripper_open_fraction": 0.0},
        )
        assert replay_observation["state.arm_joints"] == pytest.approx(replay_arm, abs=0.01)
        assert replay_info["gripper_target"] == pytest.approx(0.0)

        # The observed demonstration envelope is deliberately narrower than
        # the UR10e mechanical range.  Leaving it is diagnostic only.
        env.data.qpos[env._joint_qpos_addresses[0]] = DATASET_ARM_LOW[0] - 0.03
        env._mujoco.mj_forward(env.model, env.data)
        support_info = env._task_metrics()
        assert support_info["dataset_support_violation"] is True
        assert support_info["dataset_support_violation_joints"] == ["shoulder_pan"]
        assert support_info["hardware_safety_violation"] is False

        # A true MJCF mechanical-limit breach is terminating safety state.
        env.data.qpos[env._joint_qpos_addresses[0]] = env._hardware_arm_high[0] + 0.02
        env._mujoco.mj_forward(env.model, env.data)
        hardware_info = env._task_metrics()
        assert hardware_info["hardware_safety_violation"] is True
        assert hardware_info["hardware_safety_violation_joints"] == ["shoulder_pan"]
        assert hardware_info["safety_violation"] is True
    finally:
        env.close()

# Fine-tuning GR00T N1.7 on UR10e + Robotiq Cup Picking

This recipe prepares [`khanhnd61/ur10e-cup`](https://huggingface.co/datasets/khanhnd61/ur10e-cup), a real-robot LeRobot v3 dataset containing 81 VR-teleoperated demonstrations of **pick up the cup**. The robot is a 6-DoF UR10e with a Robotiq 2F gripper, recorded at 20 FPS with side and wrist cameras.

The source action contract is important: each 7-D action is an **absolute** target containing six joint angles followed by the gripper command, and `action[t]` is approximately `observation.state[t+1]`. The preparation script rejects a snapshot that does not satisfy this contract.

## 1. Convert and validate the dataset

GR00T currently consumes its LeRobot v2-compatible format. Run the pinned conversion in the isolated conversion environment:

```bash
uv run --project scripts/lerobot_conversion \
  python examples/UR10eCup/prepare_dataset.py \
  --root examples/UR10eCup/ur10e_cup_lerobot
```

The converted dataset is written to:

```text
examples/UR10eCup/ur10e_cup_lerobot/khanhnd61/ur10e-cup
```

The converter preserves the original v3 snapshot beside it as `ur10e-cup_v30`. Keep that backup until `meta/gr00t_conversion.json` reports `"status": "ok"`. The report checks all 81 episode parquet files, both sets of episode videos, timestamps, finite values, binary gripper values, and the absolute one-step action alignment.

To re-run only validation:

```bash
uv run --project scripts/lerobot_conversion \
  python examples/UR10eCup/prepare_dataset.py \
  --root examples/UR10eCup/ur10e_cup_lerobot \
  --validate-only
```

## 2. Generate GR00T statistics

```bash
uv run python gr00t/data/stats.py \
  --dataset-path examples/UR10eCup/ur10e_cup_lerobot/khanhnd61/ur10e-cup \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/UR10eCup/ur10e_cup_config.py
```

Do not reuse statistics from SO100, LIBERO, or another UR robot. State/action ordering and ranges are part of the model contract.

## 3. Fine-tune

The UR10e uses the checkpoint's reserved `NEW_EMBODIMENT` slot. `ur10e_cup_config.py` registers the two cameras, 6-D arm state, gripper state, and absolute 16-step action horizon.

```bash
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path examples/UR10eCup/ur10e_cup_lerobot/khanhnd61/ur10e-cup \
  --modality-config-path examples/UR10eCup/ur10e_cup_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /tmp/ur10e_cup_finetune
```

This source contains one task and one scene. A low training loss demonstrates fitting, not generalization. The preparation report records a deterministic recommended split of episodes 0-64 for training, 65-72 for validation, and 73-80 for testing, but the converter does **not** silently apply it. Build separate dataset subsets before claiming held-out accuracy.

## 4. Open-loop evaluation

```bash
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path examples/UR10eCup/ur10e_cup_lerobot/khanhnd61/ur10e-cup \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path /tmp/ur10e_cup_finetune/checkpoint-10000 \
  --traj-ids 0 1 2 \
  --execution-horizon 8 \
  --steps 800 \
  --modality-keys arm_joints gripper \
  --save-plot-path /tmp/ur10e_cup_open_loop
```

Report arm and gripper errors separately. The `wrist_2` channel is effectively constant in this dataset and should not dominate aggregate MSE/MAE.

## 5. MuJoCo proxy simulation

The repository includes an optional MuJoCo proxy for integration testing and
paired baseline/pruned rollouts. Install it with:

```bash
bash gr00t/eval/sim/UR10eCup/setup_ur10e_cup_sim.sh
```

Run a rollout using the same policy server/client path as other simulators:

```bash
MUJOCO_GL=egl uv run python gr00t/eval/rollout_policy.py \
  --env-name ur10e_cup_sim/pick_up_the_cup \
  --model-path /tmp/ur10e_cup_finetune/checkpoint-10000 \
  --n-envs 1 \
  --n-episodes 20 \
  --n-action-steps 8 \
  --max-episode-steps 400 \
  --video-dir /tmp/ur10e_cup_sim_videos
```

The environment preserves the dataset contract: 20 Hz control, two RGB
cameras named `video.side` and `video.wrist`, six absolute joint targets plus
`action.gripper`, and `1=open / 0=closed`. It clamps `wrist_2` to its current
value because that channel is effectively constant in the demonstrations.
Success requires both finger contacts, a lifted cup, no table support, and a
stable hold for several control steps.

The bundled scene is deliberately tagged `simulation_fidelity=proxy` and
`calibration_verified=false`. The public dataset does not identify the exact
Robotiq 2F variant or publish TCP, camera, table, cup, or controller
calibration. Therefore proxy success rate is useful for software regression
and matched model comparison, but **must not be reported as real-robot success
rate**.

Before any policy rollout, replay a held-out recorded demonstration through
the simulator. Episode 73 is the first episode in the recommended test split:

```bash
MUJOCO_GL=egl uv run python -m \
  gr00t.eval.sim.UR10eCup.replay_dataset_trajectory \
  --dataset-path examples/UR10eCup/ur10e_cup_lerobot/khanhnd61/ur10e-cup \
  --episode-index 73 \
  --output-dir /tmp/ur10e_cup_ground_truth_replay \
  --save-video
```

This is a mandatory simulator-calibration gate. It passes only when the
recorded action trajectory approaches the cup, establishes both finger
contacts, lifts the cup, and reaches stable success without a hardware safety
violation. A failed replay means that scene, camera, tool, gripper, or
controller calibration is still mismatched. In that state, changing pruning
rate or recovery steps cannot repair the simulator, and policy success rate
must not be reported. Use `--allow-failure` only while collecting diagnostics.

Every environment step reports arm qpos, commanded target, target error,
gripper target, finger contacts, tool/cup positions and distance, maximum cup
height, dataset-support warnings, and hardware-safety violations. These fields
are intended for diagnosing failed replays and closed-loop rollouts.

To use a calibrated MJCF, point `GR00T_UR10E_SIM_CONFIG` at a JSON file with
`simulation_fidelity="calibrated"`, `model_xml_path`, and all four verification
flags (`tool_transform_verified`, `camera_calibration_verified`,
`gripper_calibration_verified`, `scene_calibration_verified`) set to true. The
MJCF must expose the canonical joint, camera, body, and geom names documented
in `gr00t/eval/sim/UR10eCup/ur10e_cup_env.py`.

## 6. Real-robot deployment contract

The proxy does not replace hardware evaluation, so real success rate still
requires UR10e rollouts. Before sending predictions to the robot controller:

1. Interpret model output as six absolute joint targets plus one gripper target.
2. Clamp `wrist_2` to the observed/current joint value; the dataset contains almost no motion for that channel.
3. Enforce UR controller joint position, velocity, acceleration, workspace, timeout, and watchdog limits independently of the model.
4. Start with shadow-mode logging, then low-speed trials with a physical emergency stop.
5. Record the number of trials, successes, completion time, safety stops, and videos. Offline MSE/MAE is not a substitute for real-robot success rate.

This repository recipe validates data and the model I/O contract; it does not replace the certified UR/Robotiq safety controller or provide a hardware transport implementation.

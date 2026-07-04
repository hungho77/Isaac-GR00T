# Day 8 — Real LIBERO Entrypoints

## Existing Baseline Command
The documented GR00T N1.7 LIBERO baseline is a two-process server/client flow.

Server:
```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --embodiment-tag LIBERO_PANDA \
  --use-sim-policy-wrapper
```

Client:
```bash
gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
  --n-episodes 10 \
  --policy-client-host 127.0.0.1 \
  --policy-client-port 5555 \
  --max-episode-steps 720 \
  --env-name libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \
  --n-action-steps 8 \
  --n-envs 5
```

## Files And Functions
- `examples/LIBERO/README.md`: setup, checkpoint download, server command, rollout command, task lists.
- `gr00t/eval/run_gr00t_server.py`: loads `Gr00tPolicy`, optionally wraps it with `Gr00tSimPolicyWrapper`, and serves `PolicyServer`.
- `gr00t/eval/rollout_policy.py`: creates LIBERO envs, creates or connects to policy, runs rollouts, prints `results:` and `success rate:`.
- `gr00t/eval/sim/LIBERO/libero_env.py`: registers LIBERO Gymnasium tasks and loads BDDL/init states.
- `gr00t/policy/gr00t_policy.py`: `Gr00tPolicy._get_action()` processes observations and calls `self.model.get_action(...)`.
- `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`: `Gr00tN1d7.get_action()` runs backbone then action head.

## Checkpoint And Config Loading
- `run_gr00t_server.py` passes `--model-path` to `Gr00tPolicy`.
- `Gr00tPolicy` loads the model with Hugging Face `AutoModel` and the processor with `AutoProcessor`.
- The public LIBERO checkpoint path is `checkpoints/GR00T-N1.7-LIBERO/libero_10`.

## Environment Creation
- `rollout_policy.py:get_libero_env_fn()` imports `register_libero_envs()` from `gr00t/eval/sim/LIBERO/libero_env.py`.
- `gym.make(env_name)` creates tasks such as `libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`.
- `MultiStepWrapper` handles receding-horizon action execution and success termination.

## Action Prediction
- `run_rollout_gymnasium_policy()` calls `policy.get_action(observations)` once per macro step.
- In server mode this is `PolicyClient.get_action(...)`.
- In direct mode this is `Gr00tSimPolicyWrapper -> Gr00tPolicy.get_action() -> Gr00tN1d7.get_action()`.

## Success Computation
- LIBERO env info provides `success`.
- `rollout_policy.py` accumulates per-episode booleans in `episode_successes`.
- The final script prints `success rate: np.mean(results[1])`.

## Supported Episodes And Tasks
- `rollout_policy.py` supports arbitrary `--n-episodes`, `--n-envs`, `--max-episode-steps`, and any registered `libero_sim/...` task.
- `examples/LIBERO/README.md` lists LIBERO 10, Goal, Object, and Spatial tasks.

## Required Assets
- GR00T N1.7 LIBERO checkpoint, usually `checkpoints/GR00T-N1.7-LIBERO/libero_10`.
- LIBERO sim venv from `bash gr00t/eval/sim/LIBERO/setup_libero.sh`.
- `~/.libero` config and LIBERO assets/BDDL files.
- Optional running server at `127.0.0.1:5555` for subprocess client mode.

## Efficient Framework Command
In-process baseline, if the checkpoint and LIBERO are available in the current environment:
```bash
python -m gr00t.efficient.benchmark.run_libero \
  --method baseline \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --num-episodes 1 \
  --task debug \
  --save-actions \
  --output results/efficient_benchmark/real_libero/day8_baseline_real.json
```

Server/client baseline, if the original server is already running:
```bash
python -m gr00t.efficient.benchmark.run_libero \
  --method baseline \
  --real-command "gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py --n-episodes 1 --policy-client-host 127.0.0.1 --policy-client-port 5555 --max-episode-steps 720 --env-name libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it --n-action-steps 8 --n-envs 1" \
  --output results/efficient_benchmark/real_libero/day8_baseline_real.json
```

## Current Local Blockers
- The current project Python cannot import `libero`.
- `gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python` was not present during inspection.
- The default checkpoint path `checkpoints/GR00T-N1.7-LIBERO/libero_10` was not present during inspection.

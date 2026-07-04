# Day 2 Entry Point Notes

## Files Inspected
- `examples/LIBERO/README.md`
- `gr00t/eval/run_gr00t_server.py`
- `gr00t/eval/rollout_policy.py`
- `gr00t/eval/sim/LIBERO/libero_env.py`
- `scripts/eval/check_sim_eval_ready.py`
- `tests/examples/test_libero.py`

## Likely Baseline Entrypoint
The existing LIBERO baseline path is the server/client rollout workflow:

1. Start GR00T N1.7 policy serving with `gr00t/eval/run_gr00t_server.py`.
2. Run LIBERO rollouts with `gr00t/eval/rollout_policy.py`.
3. Read success from the rollout result tuple and `success rate` print.

`gr00t/eval/rollout_policy.py` already registers LIBERO environments through
`gr00t/eval/sim/LIBERO/libero_env.py` and supports `libero_sim/...` task names.

## LIBERO Availability
LIBERO is not available in the current project Python environment:

- `importlib.util.find_spec("libero")` returned `None`.
- `external_dependencies/LIBERO` exists, but `.git` is not initialized there.
- `gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python` is missing.
- `~/.libero` exists, but the dedicated simulator venv is required.

## Real Baseline Command Path
After LIBERO setup and checkpoint download, use the existing repo workflow:

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --embodiment-tag LIBERO_PANDA \
  --use-sim-policy-wrapper
```

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

## Blockers
- Run `bash gr00t/eval/sim/LIBERO/setup_libero.sh` to create the LIBERO venv.
- Download or provide a GR00T N1.7 LIBERO checkpoint.
- Add profiler hooks around the existing rollout/server path before treating
  real runs as Day 2 metric-complete results.
- No LIBERO-Plus-specific entrypoint was found during this inspection.


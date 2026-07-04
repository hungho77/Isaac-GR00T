# Day 10 — Real DummyPruner on LIBERO

## Goal
Run DummyPruner on real LIBERO to verify the visual-token pruning path before evaluating VLA-Pruner.

## Commands
```bash
python -m gr00t.efficient.benchmark.run_libero \
  --method dummy \
  --keep-ratio 0.75 \
  --dummy-mode uniform \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --num-episodes 1 \
  --task debug \
  --save-actions \
  --output results/efficient_benchmark/real_libero/day10_dummy_keep075_uniform_real.json
```

Repeat with `--keep-ratio 0.6` and `--keep-ratio 0.5`, and optionally `--dummy-mode first`.

## Checks
- `visual_token_count_before > visual_token_count_after`
- `token_reduction_ratio` matches the requested keep ratio.
- Action output remains valid.
- Episode records success/failure instead of crashing.
- `action_l2_vs_baseline` is computed when a matching baseline trace exists.

## Current Status
Real DummyPruner requires a local LIBERO setup and checkpoint. The hook code is implemented as an instance-level action-head patch through `VisualTokenHook`; it is enabled for non-baseline real runs in the efficient adapter.

Local validation was attempted with the requested command shape and stopped before rollout because real pruning needs `--model-path` or `--checkpoint-path` to attach the hook to the loaded GR00T model instance.

## Notes
If dummy pruning does not reduce latency, document it explicitly. The current hook is after Qwen3-VL backbone features are produced, so it may reduce action-head context size without reducing vision/backbone compute.

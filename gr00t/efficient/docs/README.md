# GR00T N1.7 Efficient Inference Benchmark

## Goal
Build a benchmark framework for measuring efficient inference methods on GR00T N1.7 without changing model behavior.

## Scope
This scaffold covers benchmark entry points, metric helpers, profiler utilities, pruner interfaces, scheduler placeholders, configs, docs, and result locations. It does not import LIBERO, implement pruning research methods, or modify existing GR00T training/inference logic.

## Supported Benchmarks
- LIBERO
- LIBERO-Plus

## Supported Method Families
- Visual Token Pruning
- Dynamic Scheduling
- Layer Pruning
- Quantization
- Runtime Optimization

## Current Day 1 Status
Day 1 is scaffold-only. Baseline and dummy configs exist, dry-run CLIs write JSON result files, and dummy token slicing is available only for pipeline testing.

## Dry-Run Commands
```bash
python -m gr00t.efficient.benchmark.run_libero --dry-run --method baseline --output results/efficient_benchmark/day1_libero_dryrun.json
python -m gr00t.efficient.benchmark.run_libero_plus --dry-run --method baseline --output results/efficient_benchmark/day1_libero_plus_dryrun.json
```

## Day 2 LIBERO Baseline Commands
Dry run:
```bash
python -m gr00t.efficient.benchmark.run_libero \
  --dry-run \
  --method baseline \
  --output results/efficient_benchmark/day2_libero_baseline_dryrun.json
```

Mock baseline:
```bash
python -m gr00t.efficient.benchmark.run_libero \
  --mock \
  --method baseline \
  --num-episodes 3 \
  --task debug \
  --output results/efficient_benchmark/day2_libero_baseline_mock.json
```

## Day 3 Method Registry Commands
List/dry-run baseline:
```bash
python -m gr00t.efficient.benchmark.run_libero \
  --dry-run \
  --method baseline \
  --output results/efficient_benchmark/day3_libero_baseline_dryrun.json
```

Mock baseline:
```bash
python -m gr00t.efficient.benchmark.run_libero \
  --mock \
  --method baseline \
  --num-episodes 3 \
  --task debug \
  --output results/efficient_benchmark/day3_libero_baseline_mock.json
```

Mock dummy:
```bash
python -m gr00t.efficient.benchmark.run_libero \
  --mock \
  --method dummy \
  --keep-ratio 0.75 \
  --num-episodes 3 \
  --task debug \
  --output results/efficient_benchmark/day3_libero_dummy_mock.json
```

## Day 4 Visual Token Pruner Commands
Validate pruners:
```bash
python -m gr00t.efficient.benchmark.validate_pruners \
  --output results/efficient_benchmark/day4_pruner_validation.json
```

Mock dummy first:
```bash
python -m gr00t.efficient.benchmark.run_libero \
  --mock \
  --method dummy \
  --keep-ratio 0.75 \
  --dummy-mode first \
  --visual-token-count 256 \
  --num-episodes 3 \
  --task debug \
  --output results/efficient_benchmark/day4_libero_dummy_first_mock.json
```

Mock dummy uniform:
```bash
python -m gr00t.efficient.benchmark.run_libero \
  --mock \
  --method dummy \
  --keep-ratio 0.5 \
  --dummy-mode uniform \
  --visual-token-count 256 \
  --num-episodes 3 \
  --task debug \
  --output results/efficient_benchmark/day4_libero_dummy_uniform_mock.json
```

## Day 5 VLA-Pruner MVP Commands
Validate VLA-Pruner:
```bash
python -m gr00t.efficient.benchmark.validate_vlapruner \
  --output results/efficient_benchmark/day5_vlapruner_validation.json
```

Dry-run:
```bash
python -m gr00t.efficient.benchmark.run_libero \
  --dry-run \
  --method vlapruner \
  --keep-ratio 0.75 \
  --score-mode norm \
  --output results/efficient_benchmark/day5_libero_vlapruner_dryrun.json
```

Mock VLA-Pruner norm:
```bash
python -m gr00t.efficient.benchmark.run_libero \
  --mock \
  --method vlapruner \
  --keep-ratio 0.75 \
  --score-mode norm \
  --visual-token-count 256 \
  --num-episodes 3 \
  --task debug \
  --output results/efficient_benchmark/day5_libero_vlapruner_norm_mock.json
```

Mock VLA-Pruner mean_abs:
```bash
python -m gr00t.efficient.benchmark.run_libero \
  --mock \
  --method vlapruner \
  --keep-ratio 0.5 \
  --score-mode mean_abs \
  --visual-token-count 256 \
  --num-episodes 3 \
  --task debug \
  --output results/efficient_benchmark/day5_libero_vlapruner_mean_abs_mock.json
```

## Day 6 SpecPrune + ADP Commands
Validate:
```bash
python -m gr00t.efficient.benchmark.validate_day6_methods \
  --output results/efficient_benchmark/day6_methods_validation.json
```

Mock SpecPrune:
```bash
python -m gr00t.efficient.benchmark.run_libero \
  --mock \
  --method specprune \
  --keep-ratio 0.75 \
  --reuse-steps 2 \
  --visual-token-count 256 \
  --num-episodes 3 \
  --task debug \
  --output results/efficient_benchmark/day6_libero_specprune_mock.json
```

Mock ADP:
```bash
python -m gr00t.efficient.benchmark.run_libero \
  --mock \
  --method adp \
  --visual-token-count 256 \
  --num-episodes 3 \
  --task debug \
  --output results/efficient_benchmark/day6_libero_adp_mock.json
```

Mock ADP + VLA-Pruner:
```bash
python -m gr00t.efficient.benchmark.run_libero \
  --mock \
  --method adp_vlapruner \
  --score-mode norm \
  --visual-token-count 256 \
  --num-episodes 3 \
  --task debug \
  --output results/efficient_benchmark/day6_libero_adp_vlapruner_mock.json
```

## Day 7 Comparison + Reporting Commands
Run full mock comparison:
```bash
python -m gr00t.efficient.benchmark.run_comparison \
  --mock \
  --benchmark LIBERO \
  --num-episodes 5 \
  --visual-token-count 256 \
  --task debug \
  --output-dir results/efficient_benchmark/day7_comparison
```

Collect results:
```bash
python -m gr00t.efficient.benchmark.collect_results \
  --input-dir results/efficient_benchmark/day7_comparison \
  --output-dir results/efficient_benchmark/day7_comparison
```

Generate report:
```bash
python -m gr00t.efficient.benchmark.generate_report \
  --summary-csv results/efficient_benchmark/day7_comparison/method_summary.csv \
  --output results/efficient_benchmark/day7_comparison/benchmark_report.md
```

## Real LIBERO Evaluation Commands
LIBERO-Plus is a future eval-only benchmark for robustness. Do not use LIBERO-Plus for training or fine-tuning in this framework.

The in-process real run loads the GR00T model and the LIBERO simulator in one process, so it
must use the LIBERO sim venv python (created by `bash gr00t/eval/sim/LIBERO/setup_libero.sh`)
and EGL rendering:

```bash
LIBERO_PY=gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python
```

Always pass `--seed <n>` for real runs: it seeds env init states and (via
`seed_everything`) the flow-matching action noise, so `action_l2_vs_baseline`
measures pruning drift instead of sampling noise. Baseline and pruned runs must
share the same seed.

Real LIBERO baseline:
```bash
MUJOCO_GL=egl $LIBERO_PY -m gr00t.efficient.benchmark.run_libero \
  --method baseline \
  --keep-ratio 1.0 \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --num-episodes 1 \
  --task debug \
  --save-actions \
  --output results/efficient_benchmark/real_libero/day8_baseline_real.json
```

Real DummyPruner:
```bash
MUJOCO_GL=egl $LIBERO_PY -m gr00t.efficient.benchmark.run_libero \
  --method dummy \
  --keep-ratio 0.75 \
  --dummy-mode uniform \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --num-episodes 1 \
  --task debug \
  --save-actions \
  --output results/efficient_benchmark/real_libero/day10_dummy_keep075_uniform_real.json
```

Real VLA-Pruner:
```bash
MUJOCO_GL=egl $LIBERO_PY -m gr00t.efficient.benchmark.run_libero \
  --method vlapruner \
  --keep-ratio 0.75 \
  --score-mode norm \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --num-episodes 1 \
  --task debug \
  --save-actions \
  --output results/efficient_benchmark/real_libero/day11_vlapruner_keep075_norm_real.json
```

## Pruning Stages
`--prune-stage` selects where visual tokens are dropped:

- `action_head` (default): after the backbone forward, before the DiT action head.
  Safest integration point, but the Qwen3-VL backbone still processes every token,
  so it cannot reduce latency (Day 11 finding).
- `backbone`: after an early Qwen3-VL decoder layer (`--prune-layer`, default 3,
  clamped after the DeepStack injection layers). The remaining decoder layers run
  on the shortened sequence, which is where the latency lives. Requires
  batch size 1 (`--n-envs 1`); kept tokens retain their original mrope position ids.

Real VLA-Pruner at the backbone stage:
```bash
MUJOCO_GL=egl $LIBERO_PY -m gr00t.efficient.benchmark.run_libero \
  --method vlapruner \
  --keep-ratio 0.5 \
  --score-mode norm \
  --prune-stage backbone \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --num-episodes 10 \
  --task debug \
  --seed 42 \
  --save-actions \
  --output results/efficient_benchmark/real_libero/vlapruner_backbone_kr0p5.json
```

If using the original two-terminal server/client baseline, pass `--real-command "<existing rollout command>"` to parse the rollout output into the efficient benchmark schema.

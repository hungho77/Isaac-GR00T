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

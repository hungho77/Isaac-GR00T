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


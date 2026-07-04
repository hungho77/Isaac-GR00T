# Efficient Benchmark Targets

## Milestones
- Day 1: Scaffold only.
- Day 2: Baseline LIBERO profiling.
- Day 3: Baseline LIBERO-Plus profiling.
- Day 4: `VisualTokenPruner` API plus dummy pruning.
- Day 5: VLA-Pruner MVP.
- Day 6: SpecPrune-VLA MVP plus ADP scheduler.
- Day 7: Full comparison report.

## Target Metrics
- `success_rate`
- `latency_per_action_ms`
- `episode_time_s`
- `gpu_memory_mb`
- `peak_gpu_memory_mb`
- `visual_token_count`
- `keep_ratio`
- `action_l2_vs_baseline`
- `failure_type`

## Method Targets
- Baseline: Measure unchanged GR00T N1.7 inference.
- Dummy visual token pruning: Validate pipeline plumbing with first-K token slicing only.
- VLA-Pruner MVP: Add learned or score-based visual token pruning after baseline profiling.
- SpecPrune-VLA MVP: Add task-conditioned or specification-guided pruning after dummy validation.
- ADP scheduler: Add adaptive keep-ratio scheduling after method metrics are stable.
- CLP_VLA layer pruning: Defer layer pruning until visual-token experiments are comparable.


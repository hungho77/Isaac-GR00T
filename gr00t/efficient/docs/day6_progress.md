# Day 6 — SpecPrune-VLA MVP + ADP Scheduler

## Goal
Implement SpecPrune-VLA MVP and ADP dynamic keep-ratio scheduler.

## Completed
- [x] Added SpecPruneVLA class
- [x] Added token index reuse
- [x] Added ADPScheduler
- [x] Added SpecPruneMethod wrapper
- [x] Added ADPMethod wrapper
- [x] Added ADP + VLA-Pruner hybrid wrapper
- [x] Registered specprune, adp, adp_vlapruner
- [x] Updated LIBERO mock benchmark
- [x] Added validation script
- [x] Added configs
- [x] Ran validation
- [x] Ran mock benchmarks

## Validation

| Case | Expected | Actual | Status |
|---|---|---|---|
| SpecPrune keep=0.75 | 256 -> 192 | 256 -> 192 | ok |
| SpecPrune timestep reuse | timestep 1 reused | timestep 1 reused | ok |
| ADP no action | default keep ratio | 0.5 | ok |
| ADP moving | move keep ratio | 0.6 | ok |
| ADP contact | contact keep ratio | 1.0 | ok |
| ADP + VLA-Pruner | dynamic token count | 128/154/256 | ok |

## Benchmark Metrics

| Method | Keep Ratio | Dynamic | Tokens Before | Tokens After | Reduction | Latency/action | Action L2 |
|---|---:|---|---:|---:|---:|---:|---:|
| baseline | 1.0 | no | 256 | 256 | 0.0 | 43.5 | 0 |
| vlapruner | 0.75 | no | 256 | 192 | 0.25 | 44.5 | 0 |
| specprune | 0.75 | partial | 256 | 192 | 0.25 | 44.5 | 0 |
| adp | dynamic | yes | 256 | 179.3 avg | 0.299 avg | 44.5 | 0 |
| adp_vlapruner | dynamic | yes | 256 | 179.3 avg | 0.299 avg | 44.5 | 0 |

## Blockers
- Real GR00T action_state/robot_state hook not connected yet.
- ADP phase detection currently uses mock action states.
- SpecPrune MVP uses simple mask reuse rather than full layer-wise speculative pruning.

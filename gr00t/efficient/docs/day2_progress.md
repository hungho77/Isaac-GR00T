# Day 2 — Baseline LIBERO Profiling

## Goal
Prepare and/or run baseline GR00T N1.7 profiling on LIBERO without pruning.

## Completed
- [x] Inspected repo evaluation entrypoints
- [x] Improved latency profiler
- [x] Improved memory profiler
- [x] Improved token profiler
- [x] Added baseline metrics schema
- [x] Extended LIBERO benchmark script
- [x] Added baseline config
- [x] Ran dry-run
- [x] Ran mock baseline

## Metrics
| Metric | Value |
|---|---:|
| success_rate | 1.0 |
| latency_per_action_ms | 43.5 |
| episode_time_s | 12.25 |
| gpu_memory_mb | 4096.0 |
| peak_gpu_memory_mb | 5120.0 |
| visual_token_count | 576 |
| keep_ratio | 1.0 |
| action_l2_vs_baseline | 0 |

## Entry Point Notes
See `gr00t/efficient/docs/day2_entrypoints.md`.

## Blockers
- LIBERO is not importable from the current project Python environment.
- `gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python` is missing.
- Real baseline profiling still needs hooks around the existing server/client rollout path.

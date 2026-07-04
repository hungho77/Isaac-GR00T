# Day 3 — Benchmark Framework + Method Registry

## Goal
Create benchmark method registry and method wrapper abstraction for baseline and future pruning methods.

## Completed
- [x] Added method registry
- [x] Added EfficientInferenceMethod abstraction
- [x] Added BaselineMethod
- [x] Added DummyPruningMethod wrapper
- [x] Updated run_libero.py to use registry
- [x] Added BenchmarkRunner abstraction
- [x] Added hook plan documentation
- [x] Added dummy config
- [x] Ran baseline dry-run/mock
- [x] Ran dummy dry-run/mock

## Metrics
| Method | Keep Ratio | Latency/action | GPU Mem | Visual Tokens | Action L2 |
|---|---:|---:|---:|---:|---:|
| baseline | 1.0 | 43.5 | 4096.0 | 576 | 0 |
| dummy | 0.75 | 44.5 | 4096.0 | 432 | 0 |

## Blockers
- Real LIBERO execution still depends on the simulator venv and checkpoint setup documented in `day2_entrypoints.md`.
- Visual-token pruning hooks are scaffolded but not integrated into GR00T model internals.

# Day 8 — Real LIBERO Baseline

## Goal
Connect real GR00T N1.7 LIBERO baseline execution to the efficient benchmark framework.

## Completed
- [x] Inspected real LIBERO entrypoints
- [x] Added real LIBERO adapter
- [x] Added real metric helpers
- [x] Wired real mode into `run_libero.py`
- [x] Added baseline action-trace support
- [ ] Ran real baseline locally to completion

## Command
```bash
python -m gr00t.efficient.benchmark.run_libero \
  --method baseline \
  --keep-ratio 1.0 \
  --num-episodes 1 \
  --task debug \
  --save-actions \
  --output results/efficient_benchmark/real_libero/day8_baseline_real.json
```

If no default checkpoint/server is available, pass `--model-path checkpoints/GR00T-N1.7-LIBERO/libero_10` or `--real-command "<existing rollout command>"`.

## Validation Attempt
The command above was attempted locally and stopped before rollout because neither a default model path nor the LIBERO sim Python was available.

Error:
```text
Real LIBERO baseline needs either --model-path/--checkpoint-path for in-process evaluation or a LIBERO sim Python at gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python.
```

## Metrics
| Metric | Value |
|---|---:|
| success_rate | blocked locally |
| latency_per_action_ms | blocked locally |
| gpu_memory_mb | blocked locally |
| visual_token_count_before | blocked locally |
| visual_token_count_after | blocked locally |

## Blockers
- Real local run requires LIBERO sim setup and GR00T N1.7 LIBERO checkpoint.
- Current inspection did not find `gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python`.
- Current inspection did not find `checkpoints/GR00T-N1.7-LIBERO/libero_10`.

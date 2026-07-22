# Day 11 — Real VLA-Pruner on LIBERO

## Goal
Evaluate VLA-Pruner MVP on real GR00T N1.7 LIBERO checkpoint without training or fine-tuning.

## Setup
- Checkpoint: `checkpoints/GR00T-N1.7-LIBERO/libero_10` (nvidia/GR00T-N1.7-LIBERO)
- Task: `libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` (`--task debug`)
- 5 episodes per run, `n_envs=1`, `n_action_steps=8`, `max_steps=720`, `--seed 42`
- Seeded via `seed_everything` (cuDNN deterministic mode on), so identical flow-matching
  action noise across runs; a keep-ratio 1.0 no-op control produced action L2 = 0.0 vs
  baseline, validating that the L2 column measures pruning drift only. Deterministic mode
  inflates absolute latency (~116 ms baseline vs ~88 ms unseeded); compare latencies only
  within this table. Unseeded first-pass results are archived under
  `results/efficient_benchmark/real_libero/unseeded_v1/`.
- Hardware: RTX 4070 Ti SUPER 16 GB; LIBERO sim venv (torch 2.5.1, transformers 4.57.3, mujoco 3.1.6)
- Backbone attention: SDPA fallback (flash-attn not installed in the sim venv); consistent across all runs
- Baseline emits 128 visual tokens per inference (2 cameras); peak GPU memory ~6.1 GB for every run

## Baseline
| Task | Episodes | SR | Latency/action | GPU Mem | Visual Tokens |
|---|---:|---:|---:|---:|---:|
| debug (KITCHEN_SCENE3 moka pot) | 5 | 1.00 | 116.5 ms | 6.1 GB | 128 |

## No-op Control (hook attached, keep_ratio 1.0)
| Method | SR | Latency/action | Tokens | Action L2 |
|---|---:|---:|---:|---:|
| vlapruner keep 1.0 | 1.00 | 123.4 ms | 128→128 | **0.000** |

The hook itself costs ~7 ms/action (gather/index bookkeeping) before any pruning.

## Dummy Pruning Sanity Check
| Method | Keep Ratio | SR | Latency/action | Speedup | Token Reduction | Action L2 |
|---|---:|---:|---:|---:|---:|---:|
| dummy uniform | 0.75 | 1.00 | 124.3 ms | 0.94x | 25% (128→96) | 2.17 |

Dummy 0.60/0.50 were skipped: the 0.75 run already confirmed real token pruning, mask consistency, and unchanged SR.

## VLA-Pruner Results
| Score Mode | Keep Ratio | SR | Latency/action | Speedup | GPU Mem | Tokens Before | Tokens After | Action L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| norm | 0.75 | 0.80 | 131.9 ms | 0.88x | 6.1 GB | 128 | 96 | 3.79 |
| norm | 0.60 | 1.00 | 131.8 ms | 0.88x | 6.1 GB | 128 | 77 | 3.74 |
| norm | 0.50 | 1.00 | 131.8 ms | 0.88x | 6.1 GB | 128 | 64 | 3.95 |
| mean_abs | 0.75 | 1.00 | 130.2 ms | 0.89x | 6.1 GB | 128 | 96 | 4.01 |

## Observations
- Did VLA-Pruner reduce actual visual token count? **Yes.** Every run reports exact keep-ratio reductions (128→96/77/64) with `hook_scope: visual_tokens_only` and masks pruned consistently.
- Did latency improve? **No.** Pruned runs are 6–13% slower than baseline. The hook sits after the Qwen3-VL backbone, so it only shrinks DiT cross-attention (a small fraction of total latency) while adding per-call gather/index and scoring overhead (~7 ms from the no-op control alone). Latency wins require pruning inside or before the backbone forward.
- Did success rate drop? **Mostly no.** 4 of 5 pruned runs scored 5/5, matching baseline; norm@0.75 dropped one episode (4/5). At 5 episodes per run none of these differences are statistically meaningful.
- Which keep ratio is safe? On this task, **0.50 was still safe** (SR 1.0 at half the tokens). A multi-task sweep is needed before generalizing.
- Does norm or mean_abs work better? Effectively tied: at 0.75 mean_abs went 5/5 vs norm's 4/5, with similar action drift (L2 4.01 vs 3.79). No basis yet to prefer one.
- Is action L2 correlated with failure? **No.** The single failed episode came from the run with the *lowest* VLA-Pruner L2 (3.79), and uniform dummy pruning drifts *less* (2.17) than importance-ranked selection (3.7–4.0). Notably, norm/mean_abs scoring produces more drift than spatially uniform dropping — MVP magnitude scores may discard spatially informative context that the alternate-VL DiT attends to.
- `score_mode_effective` confirms the score mode actually used (attention/action modes silently fall back to norm because the hook receives no attention or action-state signals).

## Blockers (resolved during this run)
- LIBERO setup: `egl-probe` build needs `CMAKE_POLICY_VERSION_MINIMUM=3.5` with cmake ≥ 4; robosuite 1.4.0 requires `mujoco==3.1.6` (3.10 changed the `mj_fullM` signature).
- In-process eval must run under `gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python` with `MUJOCO_GL=egl`, plus `diffusers`/`peft` installed into that venv.
- `nvidia/Cosmos-Reason2-2B` is a gated HF repo fetched at model load; requires an authenticated token with granted access.
- Unseeded runs make action-L2 meaningless (flow-matching noise dominates); the adapter now calls `seed_everything` so `--seed` covers torch RNGs, verified by the L2 = 0 no-op control.

## Next Steps
- ~~Latency: move the pruning hook before/inside the Qwen3-VL backbone forward~~ **Done (Day 12):**
  `--prune-stage backbone` prunes after an early decoder layer (post-DeepStack, FastV-style,
  original mrope positions kept), so the remaining ~13 of 16 LLM layers run on the shortened
  sequence. Smoke runs: SR 1.0 at keep 0.5, exact-match unit tests vs the unpatched forward.
  Single 2-episode runs on the RTX 4070 Ti show ±20 ms process-to-process latency noise —
  use ≥5 episodes and interleaved repeats to resolve the expected ~10-15% gain.
- Run SpecPrune/ADP real LIBERO.
- Scale to more episodes and the full libero_10 suite for statistically meaningful SR.
- Run best method on LIBERO-Plus eval only.
- Consider CLP_VLA layer pruning next.

# Day 11 — Real VLA-Pruner on LIBERO

## Goal
Evaluate VLA-Pruner MVP on real GR00T N1.7 LIBERO checkpoint without training or fine-tuning.

## Baseline
| Task | Episodes | SR | Latency/action | GPU Mem | Visual Tokens |
|---|---:|---:|---:|---:|---:|
| debug | TBD | TBD | TBD | TBD | TBD |

## Dummy Pruning Sanity Check
| Method | Keep Ratio | SR | Latency/action | Speedup | Token Reduction | Action L2 |
|---|---:|---:|---:|---:|---:|---:|
| dummy uniform | 0.75 | TBD | TBD | TBD | TBD | TBD |
| dummy uniform | 0.60 | TBD | TBD | TBD | TBD | TBD |
| dummy uniform | 0.50 | TBD | TBD | TBD | TBD | TBD |

## VLA-Pruner Results
| Score Mode | Keep Ratio | SR | Latency/action | Speedup | GPU Mem | Tokens Before | Tokens After | Action L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| norm | 0.75 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| norm | 0.60 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| mean_abs | 0.75 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Observations
- Did VLA-Pruner reduce actual visual token count? TBD.
- Did latency improve? TBD.
- Did success rate drop? TBD.
- Which keep ratio is safe? TBD.
- Does norm or mean_abs work better? TBD.
- Is action L2 correlated with failure? TBD.
- Local validation command was attempted and blocked before rollout because `--model-path` or `--checkpoint-path` is required for non-baseline hook attachment.

## Blockers
- Real local LIBERO run requires simulator setup and checkpoint availability.
- The first hook is after Qwen3-VL hidden states, so it may not reduce vision-encoder latency.

## Next Steps
- Run SpecPrune/ADP real LIBERO.
- Run best method on LIBERO-Plus eval only.
- Consider CLP_VLA layer pruning next.

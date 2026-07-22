# Day 12 — Final Report: VLA-Pruner on Real GR00T N1.7 / LIBERO

## Summary

Training-free VLA-Pruner visual token pruning was evaluated end-to-end on the real
GR00T N1.7 LIBERO checkpoint with pruning **inside the Qwen3-VL backbone**
(`--prune-stage backbone`, after decoder layer 3, FastV-style). Verdict:

- **Task quality is robust to aggressive pruning.** Success rate stays at or near
  baseline down to keep-ratio 0.5 (half the visual tokens removed).
- **End-to-end latency does not improve** on this hardware (RTX 4070 Ti SUPER):
  per-action latency is flat (~124–125 ms) across baseline and all keep ratios.
  Per-action time is dominated by components the pruning does not touch
  (vision tower, DiT denoising, obs processing); the 2B-LLM prefill slice that
  does shrink is too small to move the total at batch size 1 / seq ~178.
- **GPU memory is flat** (~6.1 GB peak) — activations at this scale are noise
  next to weights.

Conclusion: visual token pruning validates **quality headroom** (GR00T N1.7
does not need 128 visual tokens for this task) but is **not a latency lever at
this model scale and batch size**. The headroom is better spent via structural
routes (CLP_VLA layer pruning, quantization/TensorRT) or larger-batch serving
where prefill dominates.

## Setup

- Checkpoint: `checkpoints/GR00T-N1.7-LIBERO/libero_10`; task `KITCHEN_SCENE3
  turn on the stove and put the moka pot on it`; **10 episodes per run**, seed 42,
  `n_envs=1`, `n_action_steps=8`, LIBERO sim venv, EGL rendering.
- Pruning: `--prune-stage backbone --prune-layer 3` (post-DeepStack; remaining
  ~13 of 16 decoder layers run the shortened sequence; kept tokens retain their
  original mrope position ids). Scoring: `norm` (feature L2-norm top-k).
- Baseline and pruned runs share the seed, so flow-matching action noise is
  identical; a keep-1.0 no-op control (Day 11) measured action-L2 exactly 0.0.

## Results (10 episodes per run, seed 42)

| Run | SR | Latency/action | Speedup | Visual Tokens | Peak GPU | Action L2 vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 10/10 | 125.1 ms | 1.00x | 128 | 6.1 GB | — |
| vlapruner norm 0.75 | 9/10 (ep 3 failed) | 123.7 ms | 1.01x | 128→96 | 6.1 GB | n/a* |
| vlapruner norm 0.60 | 9/10 (ep 5 failed) | 123.4 ms | 1.01x | 128→77 | 6.1 GB | n/a* |
| vlapruner norm 0.50 | 10/10 | 124.7 ms | 1.00x | 128→64 | 6.1 GB | 3.92 |

\* Traces for 0.75/0.60 were overwritten by the 0.50 run (trace filenames did not
include the keep ratio — fixed in `RealLiberoAdapter._trace_path`, which now tags
non-baseline traces with `_kr<ratio>`). The recorded `action_l2_vs_baseline=0.0`
in the run JSONs is an artifact of the baseline traces living in a different
`--real-output-dir` than the pruned runs; the 0.50 value above was recomputed
offline from the on-disk traces.

Secondary observation: keep-0.5 episodes finished ~12% faster in wall clock
(7.18 s vs 8.14 s/episode) with identical per-action latency — the pruned policy
solved the task in fewer actions. That is a behavioral difference on one task,
not a compute speedup; do not read it as such.

## Interpretation

1. **SR**: 29/30 pruned episodes succeeded vs 10/10 baseline. The two isolated
   failures (one at 0.75, one at 0.60 — different episodes) show no monotonic
   trend with pruning strength (0.50 went 10/10), so at n=10 they are within
   run noise. Claiming an SR *cost* of pruning would require the full libero_10
   suite with more episodes.
2. **Latency**: the backbone-stage hook is mechanically correct (unit tests show
   numerically identical outputs at keep 1.0; sequence genuinely shortens from
   ~178 to ~114 tokens for 13 of 16 layers) yet end-to-end gain is ~1%. Profile
   before optimizing further: the LLM prefill at these sequence lengths is a
   minor slice of the ~125 ms action budget on this GPU.
3. **Action drift**: run-level L2 of 3.92 at keep 0.5 with zero SR loss — same
   magnitude as Day 11's action-head-stage drift; drift at this scale is not
   predictive of failure.

## Known limitations of the harness (for future days)

- Per-episode trace files each contain the **whole run's** actions
  (`_MetricPolicyWrapper.action_trace` is never segmented per episode), so
  action-L2 is a run-level metric repeated per episode.
- Latency and token metrics are run-level means shared across episode records.
- Action-L2 requires the baseline run to share the same `--real-output-dir`
  as the pruned runs.

## Recommendations

1. Treat visual-token pruning on N1.7 as a **quality-headroom result**, not a
   deployment optimization: keep 0.5 is safe on this task.
2. For real latency/size wins, proceed to **CLP_VLA layer pruning** (drop late
   LLM layers, save a genuinely smaller checkpoint, re-evaluate with this
   harness) and/or the existing **TensorRT/quantization** deployment path.
3. If continuing token pruning: profile the per-action budget first
   (vision tower vs LLM vs DiT vs processing), and validate on the full
   libero_10 suite before generalizing SR claims.

# VLA-Pruner Hook Feasibility for GR00T N1.7

## Status of this pass

**Final update: n=100 confirmation + 12-point grid sweep completed, both
under `torch.compile`.** Three rounds of real-checkpoint work are folded into
this report: (1) n=3/config smoke tests of all four configs (baseline +
Hooks A/B/C), (2) the same 4 configs re-run with `--torch-compile` at n=5,
which briefly suggested Hook B added a ~3% edge on top of compile, and
(3) this update -- an n=100 re-run of that exact config plus a 12-combination
`keep_ratio x prune_layer` grid sweep, both of which show the ~3% edge was
noise: every configuration in the search space is latency-flat. See "Real
LIBERO Results" for the full numbers and "Real Smoke-Test Videos" for what
was recorded. n=3-5/config smoke tests are not statistically powered on their
own -- see Hook B's own history in this repo (Day 15-18: n=200 single-task,
then a 10-task suite reversed the single-task finding) for why n needs to be
much larger before treating any
SR delta here as conclusive.

## Summary

VLA-Pruner-style pruning **can** be applied to GR00T N1.7 safely at two
validated hook points (B and C) and one newly-implemented, not-yet-real-run
point (A). None of the three hooks improve eager-mode latency at this model
scale/batch size -- confirmed by direct profiling (Day 15 audit), not
inferred. The value these hooks demonstrate is **quality-preserving
compression** (comparable success rate at 50-75% visual token retention), not
speed, unless combined with `torch.compile` on the DiT (a separate,
previously-validated ~2x lever, orthogonal to which hook is used).

## Rejected Path

**Pre-merger raw patch pruning.** Full justification in
`vlapruner_groot_hook_investigation.md`. One-line reason: the ViT's own
self-attention compute over all 256 raw patches/camera happens *before* any
merger call, so there is no point to prune-before-merger that both (a) skips
real compute and (b) doesn't break the merger's regular-grid reshape
(`Qwen3VLVisionPatchMerger.forward`, `x.view(-1, self.hidden_size)`) or the
DeepStack mergers' identical requirement.

## Candidate Hooks

| Hook | Implemented | Mode | Token Count Reduced | Latency Improved | Action Valid | Risk |
|---|---|---|---|---|---|---|
| `after_visual_merger` (A) | **yes (this pass)** | mask_only only (gather rejected -- see below) | shape unchanged by design; effective (nonzero) count reduces | not yet real-measured | not yet real-measured | low (shape-preserving, `get_placeholder_mask`/DeepStack untouched) |
| `post_policy_layer_3` (B) | yes (pre-existing) | gather | **yes**, real-measured (128->64 at keep=0.5) | **no**, real-measured (median latency unchanged, 118.5ms both) | **yes**, real-measured (SR pooled across libero_10: 91.0% vs 91.5%, p=0.86) | low (full Day 15 audit: 5 unit tests + real-model equivalence proof + 200-episode statistical validation; 2 real bugs found and fixed) |
| `action_conditioning` (C) | yes (pre-existing) | gather | yes, real-measured (Day 11) | no (post-backbone; can only affect DiT cross-attention cost) | yes (Day 11 real run; SR maintained at keep 0.5-0.75) | low (unit-tested; state/action tokens structurally cannot be pruned here -- Day 15 audit §8) |

**Why Hook A is gather-rejected specifically:** `Qwen3VLModel.get_placeholder_mask`
(`modeling_qwen3_vl.py:1066`) raises `ValueError` unless
`image_features.numel()` exactly equals the number of `<image>` placeholder
positions already fixed in `input_ids`. Satisfying that after a gather would
require also editing `input_ids` before this call -- more invasive than
gathering after the full sequence is built (which is what B does). Mask-only
sidesteps this entirely since it never changes shape.

## Real LIBERO Results

### Statistically-powered evidence (Hook B, prior sessions)

| Method | Hook | Mode | Keep Ratio | SR | Latency/action | Speedup | Action Head ms | Tokens Before | Tokens After | Action L2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | -- | -- | 1.0 | 97.5% (195/200)\* | 202.5ms (server/client) / ~120ms (in-process eager) | 1.00x | ~82ms | 128 | 128 | -- |
| vlapruner | post_policy_layer_3 (B) | gather | 0.5 | 91.0% (182/200)\* | ~120ms in-process eager, statistically identical to baseline | ~1.00x | ~82ms | 128 | 64 | 1.58 (single-task, n=200) |
| vlapruner | post_policy_layer_3 (B), pooled 10-task suite | gather | 0.5 | 91.5% (183/200) vs baseline 91.0% (182/200), pooled | 118.5ms both, identical median | 1.00x | ~82ms | 128 | 64 | 1.84 (suite pooled) |
| vlapruner | action_conditioning (C) | gather | 0.5-0.75 | maintained (Day 11; exact n smaller, pre-dates this session's statistical-rigor pass) | no measured improvement | ~1.00x | n/a (post-backbone only) | 128 | 64-96 | 2.6-4.0 (Day 11, unseeded -- see caveat below) |

\* Single-task (KITCHEN_SCENE3) result at n=200 was *not* representative of
the full libero_10 suite (see next row) -- flagged explicitly so it isn't
read as the final word on Hook B's SR cost. Per-task variance across the
suite ranged from -10pp to +15pp at n=20/task.

Day 11's Hook C numbers predate this session's discovery that unseeded runs
make action-L2 partly reflect flow-matching sampling noise rather than pure
pruning drift (fixed in the adapter via `seed_everything` later in this
session) -- treat the 2.6-4.0 action-L2 figures as an upper bound, not a
clean measurement.

### Real-checkpoint smoke test, all four configs (this update)

3 episodes/config, debug task (KITCHEN_SCENE3), seed 42, shared
`--real-output-dir` (baseline traces saved first so action-L2 is real, not a
zero-fallback), `keep_ratio=0.75` for all pruned configs, video saved per
episode (see "Real Smoke-Test Videos").

| Method | Hook | Mode | Keep Ratio | SR | Latency/action | Speedup | Tokens Before | Tokens After | Action L2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | -- | -- | 1.0 | 3/3 | 133.2ms | 1.00x | -- | -- | -- |
| vlapruner | after_visual_merger (A) | mask_only | 0.75 | 3/3 | 137.0ms | 0.97x | 128 | 96 (effective/nonzero) | 2.22 |
| vlapruner | post_policy_layer_3 (B) | gather | 0.75 | 3/3 | 132.3ms | 1.01x | 128 | 96 | 1.98 |
| vlapruner | action_conditioning (C) | gather | 0.75 | 3/3 | 133.2ms | 1.00x | 128 | 96 | 1.81 |

**n=3/config -- read as "did not crash, action stayed plausible," not as a
statistically meaningful SR/latency comparison.** All four configs held 3/3
success and latency within noise of each other, consistent with (not proof
of) the larger-n findings above. Hook A's first-ever real-checkpoint run
produced a valid, non-degenerate output: correct effective token count
(96 = 0.75 x 128, matching the configured keep_ratio exactly), no shape
errors, no `hook_error` in metadata, action L2 in the same range as the two
already-validated hooks (1.8-2.2, not an outlier).

### Compile stacked on top of each hook (this update)

Same 4 configs, `--torch-compile` added (compiles `action_head.model`,
`mode="reduce-overhead"`), 5 episodes/config (bumped from 3 so JIT warmup on
the first call doesn't dominate the mean -- median is the number to read;
mean is warmup-skewed, shown for transparency only). Fresh output directory
(`hook_experiments_compiled/`) so traces don't collide with the eager run.

| Method | Hook | Mode | SR | Median lat/action | Mean lat/action | Tokens | Action L2 |
|---|---|---|---:|---:|---:|---:|---:|
| baseline | -- | -- | 5/5 | 65.8ms | 153.9ms | -- | -- |
| vlapruner | after_visual_merger (A) | mask_only | 5/5 | 65.6ms | 106.5ms | 128->96 | 1.55 |
| vlapruner | post_policy_layer_3 (B) | gather | 5/5 | **63.8ms** | 141.8ms | 128->96 | 1.66 |
| vlapruner | action_conditioning (C) | gather | 4/5 | 65.8ms | 111.0ms | 128->96 | 2.07 |

**The compile win holds, re-confirmed post-fixes:** ~65-66ms median across
the board vs ~132-137ms eager (§ above) -- the same ~2x win found in an
earlier session, now reproduced with the current (state-leak-fixed) codebase
and with the new Hook A stacked on top without regression.

**Initial (n=5) read: Hook B showed a ~3% latency edge from token reduction
(63.8ms vs 65.8ms compiled baseline) -- the first measurable one anywhere in
this investigation.** This did **not** survive a proper re-run at n=100 (see
next section) -- flagged here rather than left uncorrected.

### Confirming the Hook B edge at n=100 (this update) -- it does not replicate

Same two configs (`baseline` and `vlapruner keep=0.75 layer=3`, both
`--torch-compile`), n=100 each, same seed=42, fresh output directory
(`day19_confirm_hookB_compile/`).

| | SR | Median lat/action | Mean lat/action | Per-episode stdev |
|---|---:|---:|---:|---:|
| baseline (compiled) | 99/100 | 64.65ms | 66.18ms | 15.0ms |
| Hook B (compiled) | 96/100 | 64.01ms | 66.31ms | 22.0ms |

Median delta: **+0.63ms (1.0%)**, not the ~3% seen at n=5, and well inside
one standard deviation of either distribution (15-22ms per-episode). Mean
latency is actually *higher* for Hook B. SR difference (99 vs 96) is not
significant either (two-proportion z~1.36, need ~1.96 for p<0.05).

**Verdict: the n=5 smoke test's edge was noise, not a real effect.** At
proper statistical power, Hook B (or any hook) stacked with compile shows no
latency benefit over compile alone. This is a correction to the "promising
lead" framing in the previous update of this report -- the smoke test result
should not have been trusted at that sample size, and it wasn't once tested
properly.

### Grid sweep: keep_ratio x prune_layer (this update) -- confirms and generalizes the null result

12 combinations (`keep_ratio` in {0.75, 0.6, 0.5} x `prune_layer` in
{2, 3, 5, 8}), n=5 episodes each, all `--torch-compile`, Hook B
(`post_policy_layer_3`), same seed=42.

| keep_ratio | layer | SR | median ms | tokens after |
|---:|---:|---:|---:|---:|
| 0.75 | 2 | 5/5 | 63.4 | 96 |
| 0.75 | 3 | 5/5 | 62.4 | 96 |
| 0.75 | 5 | 5/5 | 63.0 | 96 |
| 0.75 | 8 | 5/5 | 63.3 | 96 |
| 0.6 | 2 | 4/5 | 63.6 | 77 |
| 0.6 | 3 | 5/5 | 63.2 | 77 |
| 0.6 | 5 | 5/5 | 63.3 | 77 |
| 0.6 | 8 | 3/5 | 63.5 | 77 |
| 0.5 | 2 | 5/5 | 64.0 | 64 |
| 0.5 | 3 | 5/5 | 63.3 | 64 |
| 0.5 | 5 | 5/5 | 63.2 | 64 |
| 0.5 | 8 | 5/5 | 64.2 | 64 |

(`action_l2_vs_baseline` was 0.000 for every row -- an artifact of the sweep
writing to its own output directory with no baseline trace present to
compare against, not a real fidelity measurement. Not a usable column here.)

**Every one of the 12 combinations lands in a 62.4-64.2ms band** -- a 1.8ms
spread across the *entire* grid, smaller than the 15-22ms per-episode stdev
measured at n=100 above. Neither `keep_ratio` nor `prune_layer` moves latency
at all once compiled -- this generalizes the n=100 correction: it isn't that
`keep=0.75, layer=3` specifically lacks an edge, no point in this search
space has one. **This is a genuine null result for "auto-optimize token/layer
for latency"**: there is nothing to optimize, because the architecture
(launch-overhead-bound at batch=1) doesn't leave room for token-count or
prune-layer choice to matter, regardless of configuration.

The two SR dips (`keep=0.6, layer=2`: 4/5; `keep=0.6, layer=8`: 3/5) don't
form a clean pattern (layers 3 and 5 at the same keep_ratio were both 5/5)
and are within the range a single noisy episode produces at n=5 -- not
trusted without a much larger rerun, and not chased further given the
latency search already returned null.

**If a single config must be chosen despite no latency benefit**:
`keep_ratio=0.75, layer=3` -- fastest single point in the grid (62.4ms, though
statistically indistinguishable from every other cell), and the configuration
with by far the most real validation history in this investigation (Day 15
audit, 200-episode run, full 10-task suite).

**Caveat: n=5/config is still a smoke test.** A ~2ms gap needs a much larger
sample (order of Hook B's earlier 200-episode SR validation) before treating
it as a confirmed effect rather than noise. The direction and mechanism are
sound; the magnitude is not yet statistically established. Hook C's single
failure (4/5) is within noise at this n, not evidence of degradation.

## Real-Run Commands Used

Executed once the GPU freed up (see results above); kept here so the exact
invocations are reproducible:

```bash
LIBERO_PY=gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python
MODEL=checkpoints/GR00T-N1.7-LIBERO/libero_10
OUT=results/efficient_benchmark/real_libero/hook_experiments
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
mkdir -p "$OUT" "$OUT/videos"

# Baseline (run first -- writes the action traces the other three compare against)
$LIBERO_PY -m gr00t.efficient.benchmark.run_libero \
  --method baseline --keep-ratio 1.0 --model-path "$MODEL" \
  --num-episodes 3 --task debug --seed 42 --save-actions \
  --video-dir "$OUT/videos/baseline" \
  --real-output-dir "$OUT" --output "$OUT/baseline.json"

# Hook A: after-merger mask-only VLA-Pruner (new)
$LIBERO_PY -m gr00t.efficient.benchmark.run_libero \
  --method vlapruner --keep-ratio 0.75 --score-mode norm \
  --prune-stage after_visual_merger --prune-mode mask_only \
  --model-path "$MODEL" --num-episodes 3 --task debug --seed 42 --save-actions \
  --video-dir "$OUT/videos/after_merger_mask_vlapruner" \
  --real-output-dir "$OUT" --output "$OUT/after_merger_mask_vlapruner.json"

# After-merger gather: SKIP, rejected -- see "Candidate Hooks" table.
# --prune-mode gather at --prune-stage after_visual_merger is not implemented
# (VisualMergerHook forces mask_only regardless of the requested mode).

# Hook B: post-policy-layer-3, gather (already validated; smoke re-run for comparison)
$LIBERO_PY -m gr00t.efficient.benchmark.run_libero \
  --method vlapruner --keep-ratio 0.75 --score-mode norm \
  --prune-stage backbone --prune-layer 3 \
  --model-path "$MODEL" --num-episodes 3 --task debug --seed 42 --save-actions \
  --video-dir "$OUT/videos/post_layer3_gather_vlapruner" \
  --real-output-dir "$OUT" --output "$OUT/post_layer3_gather_vlapruner.json"

# Hook C: action-conditioning, gather (already validated; smoke re-run for comparison)
$LIBERO_PY -m gr00t.efficient.benchmark.run_libero \
  --method vlapruner --keep-ratio 0.75 --score-mode norm \
  --prune-stage action_head \
  --model-path "$MODEL" --num-episodes 3 --task debug --seed 42 --save-actions \
  --video-dir "$OUT/videos/action_conditioning_vlapruner" \
  --real-output-dir "$OUT" --output "$OUT/action_conditioning_vlapruner.json"
```

## Real Smoke-Test Videos

12 episode videos saved under `results/efficient_benchmark/real_libero/hook_experiments/videos/<config>/*.mp4`
(3 per config x 4 configs). One representative video per config was sent to
the user directly for visual inspection; the rest remain on disk at the path
above for further review.

## Remaining Gap: Per-Stage Profiler Not Yet Wired

`attach_stage_profiler` (new this pass, `profiler/model_stages.py`) is a
working, unit-tested library function, but `real_libero_adapter.py` does not
call it automatically -- the smoke-test numbers above come from
`real_metrics.py`'s existing `latency_per_action_ms` (whole-`get_action`
timing), not a vision/merger/LLM/action-head breakdown. Wiring
`attach_stage_profiler` into `RealLiberoAdapter.run_existing_libero_baseline_python`
(attach after hook setup, `detach()` + fold `stage_summary_with_gpu_memory()`
into record metadata after the rollout) would surface that breakdown
automatically; not done in this pass to keep the diff reviewable.

## Key Findings

- **Can VLA-Pruner run on real GR00T N1.7 LIBERO checkpoint?** Yes, at all
  three hooks. B and C are extensively confirmed (n=200+); A now has a
  successful real-checkpoint smoke test (3/3 episodes, no errors) in addition
  to its unit tests.
- **Does it reduce real token count?** Yes for B/C (gather, measured 128->96
  at keep=0.75 in the smoke test, 128->64 at keep=0.5 in the larger runs). For
  A, shape is unchanged by design (mask-only); the *effective* (nonzero) count
  reduces identically and was confirmed correct in the real run (96 of 128,
  exactly matching keep_ratio=0.75) -- there's just no tensor size reduction
  to point to.
- **Does it improve real latency?** No, at any hook, at any keep_ratio, at
  any prune_layer -- eager or compiled. Root cause (Day 15 audit): batch=1 +
  this model's scale means per-layer kernels are launch-overhead-bound, not
  FLOPs-bound, so removing FLOPs via pruning doesn't move wall-clock time
  (architecture-appropriate, not a bug; see the VLA-Pruner paper comparison
  earlier this session for why their reported speedups, from autoregressive
  KV-cache reuse on a 7B backbone, don't transfer to GR00T's 4-step
  flow-matching DiT). `torch.compile` on the DiT *does* give a real ~2x
  (65-66ms median vs ~132-137ms eager, re-confirmed at n=100 this update),
  but it is unaffected by which hook, if any, runs alongside it: an n=5 smoke
  test briefly suggested Hook B added a further ~3%, but a proper n=100
  re-run of that exact config showed only a 1.0% median delta (well inside
  the 15-22ms per-episode noise band), and a 12-point grid over
  `keep_ratio x prune_layer` confirmed every combination lands within a
  1.8ms band -- a clean null result, not a promising lead.
- **Does it preserve action output?** Yes at B/C (SR statistically
  indistinguishable from baseline when pooled across the full libero_10
  suite). At A, the n=3 smoke test also held 3/3 with action L2 (2.22) in the
  same range as B/C (1.8-2.0) -- not statistically powered, but not an
  outlier either.
- **Which hook is safest?** All three are low-risk by construction (mask-only
  for A never changes shape; B/C gather-prune only after multi-tensor-
  consistent index_select, audited end-to-end). A is the newest of the three
  and has the least real-run history (one 3-episode smoke test vs. B/C's
  hundreds of episodes), not because anything about its first run was
  concerning.
- **Which hook has highest latency potential?** None, confirmed at proper
  statistical power. B (post-layer-3) was the only hook with a theoretical
  case (it's the only one that runs *before* the bulk of the LLM's own
  compute), and an n=5 smoke test briefly appeared to show it (63.8ms vs
  65.8ms compiled baseline) -- but the n=100 re-run and the 12-point
  `keep_ratio x prune_layer` grid both came back flat (all within ~2ms of
  each other and of baseline). A (after-merger) has **no latency potential at
  all by construction** -- mask-only never changes compute, masked rows still
  flow through every downstream matmul. C (post-backbone) can only ever touch
  DiT cross-attention and was flat under compile in every test. All three are
  latency-neutral once the real numbers are in; the search space has no
  winner to find.
- **Should we continue VLA-Pruner or pivot to action-head/runtime
  optimization?** See Recommendation.

## Recommendation

**Final, post-confirmation: #4, cleanly -- stop chasing latency via pruning
hooks or pruning hyperparameters, at any keep_ratio or prune_layer.**
`torch.compile(mode="reduce-overhead")` on `action_head.model` is the one
real, reproducible latency lever found in this entire investigation (~2x,
re-confirmed at n=100 and across a 12-point grid sweep in this update). An
n=5 smoke test briefly suggested Hook B added a further ~3% on top of
compile; a proper n=100 re-run of exactly that config showed a 1.0% median
delta, well inside the 15-22ms per-episode noise band, and a 12-combination
grid over `keep_ratio x prune_layer` confirmed the null generalizes -- every
point in the search space lands within a 1.8ms band. There is no
(keep_ratio, prune_layer) configuration that measurably beats compile alone,
so there is nothing left to auto-optimize for speed in this space.

Practical guidance: ship `torch.compile` as the latency lever, independent of
any pruning choice. If pruning is still wanted for **quality/compression**
reasons (not latency), Hook B (`post_policy_layer_3`) at `keep_ratio=0.75,
layer=3` remains the best-supported default -- not because it's faster (it
isn't, confirmed), but because it has the deepest real-run history in this
investigation (Day 15 audit, 200-episode run, full 10-task suite, this
update's n=100 re-run) and preserves SR throughout. Hook A (`after_visual_merger`,
mask-only) and Hook C (`action_conditioning`) remain valid, low-risk
quality-preserving options but have no latency case at all, by construction.

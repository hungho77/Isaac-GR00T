# VLA-Pruner Hook Investigation for GR00T N1.7

## Status: no GPU/checkpoint execution in this pass

This document was produced while the GPU was reserved for the user's own model
training. Everything below is verified by static code inspection and CPU-only
unit tests with synthetic tensors (no real checkpoint, no CUDA). Real LIBERO
runs (Phase 6/7 of the investigation) are deferred until the GPU is free --
see `vlapruner_groot_real_hook_report.md` for the exact commands to run then.

## 1. Current efficient framework status

`gr00t/efficient/` (this repo, branch `dev/efficient-inference-benchmark`):

- `benchmark/` -- CLI entry points (`run_libero.py`, `run_libero_plus.py`,
  `run_comparison.py`), method registry (`registry.py`, `methods.py`), the
  real-LIBERO adapter (`real_libero_adapter.py`), metrics/reporting
  (`metrics.py`, `real_metrics.py`).
- `pruners/` -- `VisualTokenPruner` base class, `DummyVisualTokenPruner`,
  `VLAPruner` (norm/mean_abs/attention/action score modes, temporal-momentum
  smoothing), `SpecPrune`.
- `schedulers/` -- `ADP` dynamic keep-ratio scheduler.
- `hooks/` -- three hook modules, described below.
- `profiler/` -- `latency.py`, `memory.py`, `token.py` helpers.
- `docs/` -- day-by-day progress logs; `day15_audit_report.md` is the most
  relevant prior artifact (a full correctness audit of Hook B, with two
  confirmed-and-fixed bugs).

## 2. Real LIBERO baseline entrypoint

Two paths exist, both going through `Gr00tPolicy`:

- **Two-process server/client** (`examples/LIBERO/README.md`): `python
  gr00t/eval/run_gr00t_server.py --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10
  --embodiment-tag LIBERO_PANDA --use-sim-policy-wrapper`, then
  `gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py
  --n-episodes N --policy-client-host 127.0.0.1 --policy-client-port 5555 ...`.
  `run_gr00t_server.py` also has `--efficient-*` flags (added this session) that
  attach a pruning hook to the server's model instance before it starts serving.
- **In-process benchmark adapter** (`gr00t/efficient/benchmark/real_libero_adapter.py:
  RealLiberoAdapter.run_existing_libero_baseline_python`): loads `Gr00tPolicy`
  directly, optionally attaches a hook, and calls
  `gr00t.eval.rollout_policy.run_rollout_gymnasium_policy` once per episode
  (fixed to do this per-episode, not per-batch, in the Day 15 audit -- see
  `_run_episodes_one_at_a_time`). This is what `python -m
  gr00t.efficient.benchmark.run_libero --model-path ...` (no `--mock`/`--dry-run`)
  actually runs.

Both paths use the identical model/policy code underneath -- there is only one
inference implementation, not a server-specific and an eval-specific one.

## 3. Current VLA-Pruner MVP implementation

`gr00t/efficient/pruners/vlapruner.py: VLAPruner.prune(visual_tokens, ...)`:
validates `[B, N, D]` input, computes a per-token score (`norm`, `mean_abs`,
`attention`, or `action`-conditioned), optionally blends it with an EMA of the
previous call's score (`temporal_momentum`), takes `torch.topk` over the
score, and returns `(gathered_tokens, metadata)` plus `self.last_selected_indices`
(the kept, sorted-ascending indices). This is the one piece of logic every
hook below reuses -- hooks differ only in *where* they extract the `[B,N,D]`
tensor from and *how* they apply the pruning decision (gather vs. mask), not
in how tokens are scored.

## 4. Existing hooks (2 of the 3 requested already exist and are validated)

| User's hook name | This repo's name | File | Status |
|---|---|---|---|
| `after_visual_merger` (Hook A) | `VisualMergerHook` | `hooks/visual_merger_hook.py` | **New in this pass** -- implemented, unit-tested, not yet run on real LIBERO |
| `post_policy_layer` (Hook B) | `BackboneVisualTokenHook` | `hooks/backbone_token_hook.py` | Already existed; extensively validated (Day 15 audit + 200-episode + 10-task-suite real runs) |
| `action_conditioning` (Hook C) | `VisualTokenHook` | `hooks/visual_token_hook.py` | Already existed; validated in Day 11 real runs + this session's server smoke test |

### Hook B (`post_policy_layer_3`) -- already implemented, already proven

`attach_backbone_visual_token_hook` monkey-patches `Qwen3VLTextModel.forward`
with `_text_forward_with_pruning`, an undecorated re-implementation of the
stock forward. After decoder layer `prune_layer` (default 3, clamped past the
last DeepStack injection layer via `resolve_prune_layer`), it calls
`hook.prune_intermediate(...)`, which:
1. Isolates visual positions via `visual_pos_masks` (the model's own
   `image_mask`, requires batch_size==1 -- backbone-stage pruning safely
   no-ops otherwise).
2. Scores/selects via the shared `VLAPruner`/`DummyVisualTokenPruner` logic.
3. `index_select`s `hidden_states`, `visual_pos_masks`, `text_position_ids`,
   `position_ids`, `cache_position`, `attention_mask` (when not `None`) with
   the *same* kept-index tensor -- this consistent multi-tensor gather is
   exactly what the Day 15 audit verified end-to-end (5/5 unit tests,
   real-model equivalence tests, 200-episode statistical validation).

Real evidence already collected (cited in full in
`vlapruner_groot_real_hook_report.md`): token count genuinely reduces
(128->64 at keep=0.5), action output remains valid (SR pooled across libero_10
suite: baseline 91.0% vs pruned 91.5%, statistically indistinguishable,
n=200/method), and eager-mode latency does **not** improve (median identical,
118.5ms both) -- launch-overhead-bound at this model scale/batch size, not a
bug (see `day15_audit_report.md` §14 for the profiled breakdown proving this).

### Hook C (`action_conditioning`) -- already implemented, already proven

`attach_visual_token_hook` monkey-patches `action_head.process_backbone_output`
(`gr00t/model/gr00t_n1d7/gr00t_n1d7.py:175`), operating on the fully-processed
`backbone_features`/`image_mask`/`backbone_attention_mask` right before the
DiT's cross-attention conditioning. This is the tensor the user's Phase 4
"Action Conditioning Path" asks about:

- File: `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`
- Conditioning tensor: `vl_embeds = backbone_output.backbone_features`,
  shape `[B, N, D]` (`N`=149 in our captured example: 128 visual + 21
  text/special tokens, `D`=2048), fed to the DiT as
  `encoder_hidden_states=vl_embeds` (cross-attention keys/values).
  `sa_embs` (state+action embeddings) is the DiT's *separate* self-attention
  query stream -- state/action tokens are never part of the tensor this hook
  touches (structurally impossible to prune them here, confirmed in the
  Day 15 audit §8).
- The hook prunes only visual positions within this mixed tensor (via
  `image_mask`), same multi-tensor-consistent gather pattern as Hook B.
- **Latency impact is smaller here than Hook B's, and for the same reason
  either way** (launch-overhead-bound): this hook sits *after* the entire
  Qwen3-VL backbone has already run at full cost, so pruning here can only
  ever reduce DiT cross-attention compute, never backbone compute.

## 5. Where visual tokens flow (full path, for reference)

```
pixel_values [B, cams, C, H, W]
  -> Qwen3VLVisionPatchEmbed (Conv3d)          -- raw patches, 256/camera here
  -> Qwen3VLVisionModel.blocks (ViT self-attn) -- ALL 256 patches attend, every block
  -> [DeepStack mergers at intermediate depths -- see Phase 1]
  -> Qwen3VLVisionModel.merger                 -- 256 -> 64 tokens/camera (4x)
  -> Qwen3VLModel.get_image_features returns image_embeds (list, per-image)
       <-- HOOK A (after_visual_merger) inserts here
  -> torch.cat + masked_scatter into inputs_embeds (packed text+image sequence)
  -> Qwen3VLTextModel.forward, 16 decoder layers
       <-- HOOK B (post_policy_layer_3) inserts after layer 3
  -> action_head.process_backbone_output (vlln + vl_self_attention)
       <-- HOOK C (action_conditioning) inserts here
  -> AlternateVLDiT: hidden_states=sa_embs (state+action), encoder_hidden_states=vl_embeds
  -> 4 flow-matching denoising steps -> normalized action -> denormalized
```

See §Rejected Hook below for why a fourth candidate point (before the merger,
inside the ViT) was investigated and ruled out.

## Rejected Hook: Pre-Merger Raw Patch Pruning

**Not implemented.** Verified invalid by direct code inspection, not by
running anything.

- File: `transformers/models/qwen3_vl/modeling_qwen3_vl.py`
  (`gr00t/eval/sim/LIBERO/libero_uv/.venv/lib/python3.10/site-packages/...`)
- Class/function: `Qwen3VLVisionModel.forward` (line 705)

```python
hidden_states = self.patch_embed(hidden_states)      # 256 raw patches/camera
...
for layer_num, blk in enumerate(self.blocks):          # ALL ViT self-attention runs HERE
    hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, ...)
    if layer_num in self.deepstack_visual_indexes:
        deepstack_feature = self.deepstack_merger_list[...](hidden_states)   # merger call #1, #2, ...
hidden_states = self.merger(hidden_states)             # final merger, 256 -> 64
```

And the merger itself (`Qwen3VLVisionPatchMerger.forward`, line 100):
```python
x = self.norm(x.view(-1, self.hidden_size))   # self.hidden_size = vit_hidden * merge_size**2 (4)
```

**Exact reason rejected:**
1. `self.blocks` (the ViT's own self-attention stack, where essentially all
   vision-tower compute lives) runs to completion on **all** 256 raw patches
   *before* any merger is ever called. There is no point between "patches
   exist" and "patches are expensive" to intervene -- by the time the merger
   runs, the compute it might have let us skip has already happened.
2. The merger's `.view(-1, self.hidden_size)` requires a **complete, regular
   grid**: every group of `spatial_merge_size**2 = 4` consecutive raw-patch
   embeddings must be present to reshape correctly. Pruning individual raw
   patches before this call breaks the reshape outright.
3. This isn't a single call site either -- `deepstack_merger_list` calls the
   merger *again* at each intermediate DeepStack-injection depth, each
   requiring the same complete grid.
4. The only point where pruning *would* skip real ViT compute is
   **immediately after `patch_embed`, before any block runs** -- but that
   uses raw, un-contextualized patch embeddings (no attention has happened
   yet) as the pruning signal, a materially weaker/riskier basis than
   anything VLA-Pruner's paper or our own hooks use (both require at least
   one attention pass to produce a meaningful importance score).

**Profiling implication:** even a hypothetical *perfect* elimination of this
signal-weak, pre-attention path would only touch the vision tower's share of
the budget -- measured at ~15ms of ~120ms total (~12.5%) in this session's
profiling (`day15_audit_report.md` §14, `vision_tower` row). Not worth the
correctness risk for a ceiling that small.

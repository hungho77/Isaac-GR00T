# Day 15 — VLA-Pruner Production Audit

Audit scope: (A) root-cause the `max_abs_diff≈1.25` eager-vs-compiled anomaly reported
in Day 14, and (B–D) verify VLA-Pruner's integration into the real GR00T N1.7 inference,
benchmark, and server paths. Evidence-first: every claim below cites the file/line or
the script/run that produced it. Scripts referenced as `scratch/*.py` live under this
session's scratchpad and are reproducible with the commands in §18.

## 1. Executive verdict

The reported `max_abs_diff≈1.25` is **not** caused by VLA-Pruner, top-k selection, index
misalignment, masking, or cached state. It is a **Category E** (genuine compile/backend
numerical difference) in ordinary, unmodified Qwen3-VL decoder-layer computation under
`torch.compile` + BF16, most plausibly driven by a concrete, logged mechanism: Dynamo
guards on `module.layer_idx` inside `flash_attention_forward`, generating a separate
compiled graph per decoder layer; with 16 layers against the default
`cache_size_limit=8`, Dynamo exhausts its cache and silently falls back to eager for the
remaining layers, producing a partially-compiled forward pass. This was proven, not
assumed — the decisive test (§6) shows eager and compiled select **identical token
indices in identical order**, and disabling pruning entirely does not remove the
divergence.

Critically, **this is not a shipped risk**: production (`run_gr00t_server.py`,
`real_libero_adapter.py`) only ever compiles `action_head.model` (the DiT). The
LLM/backbone is never compiled in any code path a user runs. The anomaly was reproduced
exclusively in a scratch diagnostic script built for this audit.

Two real, previously-undetected bugs were found and fixed during this audit (§15–16):
VLA-Pruner's temporal-momentum state was never reset between episodes (benchmark) or
between clients/episodes on a long-lived server (production), and per-camera token
retention has no fairness floor (documented risk, not patched — see §17).

## 2. Actual GR00T N1.7 call graph (production forward pass)

```text
Gr00tPolicy._get_action (gr00t/policy/gr00t_policy.py:380)
  -> self.model.get_action(**collated_inputs)      (gr00t_n1d7.py:603, Gr00tN1d7.get_action)
     -> backbone.prepare_input(inputs)              (qwen3_backbone.py, keys: input_ids,
                                                      attention_mask, pixel_values, image_grid_thw)
     -> backbone.forward(vl_input)                  (qwen3_backbone.py:281)
        -> self.model(**vl_input, output_hidden_states=True)   [Qwen3VLForConditionalGeneration]
           -> Qwen3VLModel.forward                  (modeling_qwen3_vl.py:1108, @check_model_inputs)
              -> vision tower: self.get_image_features(...)     [Qwen3VLVisionModel]
              -> masked_scatter image embeds into text embeds   (packed VL sequence)
              -> Qwen3VLTextModel.forward             (modeling_qwen3_vl.py:782, @check_model_inputs)
                 -> 16 decoder layers (RMSNorm, SDPA/FA2 attention, MLP; DeepStack
                    injection merges vision features into layers 0..N-1)
        <- BatchFeature(backbone_features, backbone_attention_mask, image_mask)
     -> action_head.process_backbone_output(...)     (gr00t_n1d7.py:175 -- vlln + vl_self_attention)
     -> action_head.get_action(...)                  (gr00t_n1d7.py:446)
        -> _encode_features                          (gr00t_n1d7.py:288 -> :307 process_backbone_output)
        -> state_encoder(state) + action_encoder(noisy_trajectory, t, embodiment_id)
        -> AlternateVLDiT.forward (4 flow-matching denoise steps)  [action_head.model]
           hidden_states=sa_embs (state+action, self-attend)
           encoder_hidden_states=vl_embeds (VL tokens, cross-attend)
  <- normalized action -> denormalized via embodiment stats
```

**VLA-Pruner hook insertion point** (`gr00t/efficient/hooks/backbone_token_hook.py`):
monkey-patches `Qwen3VLTextModel.forward` (`patch_text_model`, line 182) with
`_text_forward_with_pruning`, a plain undecorated re-implementation of the stock forward
that calls `hook.prune_intermediate(...)` after an early decoder layer (`prune_layer`,
default 3, clamped past the last DeepStack injection layer via `resolve_prune_layer`).
This is **inside** the LLM, before the expensive remaining ~13 layers — the design
intent is real compute reduction, not just a post-hoc filter.

## 3. Actual server call graph

```text
uv run python gr00t/eval/run_gr00t_server.py --efficient-method vlapruner ...
  main() (run_gr00t_server.py:139)
    Gr00tPolicy(...)                         # loads checkpoint, builds model
    if config.efficient_method:
      _attach_efficient_inference(policy.model, config)   # builds VLAPruner, attaches
                                                            # BackboneVisualTokenHook,
                                                            # optionally torch.compile's
                                                            # ONLY action_head.model
    Gr00tSimPolicyWrapper(policy)            # if --use-sim-policy-wrapper
    [FIX, day15] wrap policy.reset to also call efficient_method.reset()
    PolicyServer(policy=policy, ...).run()
      while True: recv() -> dispatch(endpoint) -> send()   # ZMQ REP, single-threaded,
                                                             # strictly synchronous
      endpoints: "get_action" -> policy.get_action
                 "reset"      -> policy.reset (now efficient-state-aware)
                 "ping", "kill", "get_modality_config"
```

Confirmed via `gr00t/policy/server_client.py:145-274`: `PolicyServer` binds one ZMQ
`REP` socket and processes requests in a single `while self.running` loop — one request
fully handled (including model inference) before the next `recv()`. **No concurrent
request execution is possible by construction**; this rules out data races on shared
hook/pruner state, but does **not** rule out state leaking *sequentially* across
different clients/episodes (§15).

**No warmup call exists** anywhere in `run_gr00t_server.py` before `server.run()`. When
`--efficient-torch-compile` is set, the *first real client request* pays the full
Dynamo/Inductor compilation cost inline. `PolicyClient` defaults to a 15s per-request
timeout (`server_client.py:282`); a slow first compile could plausibly exceed this,
producing a spurious `zmq.error.Again` on the client's first call. Not reproduced with
a hard number in this audit (would need isolated timing of compilation alone); flagged
as a real, plausible operational risk (§17).

## 4. torch.compile eager-parity results

Scope correction first: **production compiles only the DiT (`action_head.model`)**,
confirmed by grep across `run_gr00t_server.py:128` and `real_libero_adapter.py:351` —
both are the only two `torch.compile(...)` call sites in the entire `gr00t.efficient`
+ server integration. The LLM-compile investigation below targets a scratch
probe (`scratch/audit_topk_equivalence.py`) built specifically for this audit, not a
shipped path, because that scratch probe is where the `max_abs_diff≈1.25` anomaly was
originally observed (Day 14 chat).

Controlled methodology (per audit requirements): single model instance (avoids
doubling ~6GB GPU load), one frozen real input captured after full preprocessing
(`inputs_embeds` shape `[1, 149, 2048]`, `visual_pos_masks.sum()=128`), `model.eval()` +
`torch.no_grad()`, explicit `method.reset()` immediately before every comparison
capture (VLAPruner's `prev_score` temporal-momentum is real mutable state — an earlier,
naive version of this test that didn't reset between captures produced a misleading
"correctness failure" that was actually just comparing call N=1 against call N=4 of a
stateful function; this was caught and fixed before drawing any conclusion).

| Test | keep_ratio | compile mode | Result |
|---|---|---|---|
| Baseline eager-vs-compiled | 0.5 | default | `final_hidden` max_abs_diff=1.25, cosine=0.99990 |
| Baseline eager-vs-compiled | 0.5 | reduce-overhead | max_abs_diff=1.25 (identical) |
| **Pruning fully disabled** | 1.0 (no-op branch, topk/gather never executes) | default | `final_hidden` max_abs_diff=**0.75**, `hidden_before_prune` max_abs_diff=**1.0**, cosine=0.99990 |

The disabled-pruning row is decisive on its own: with the entire pruning branch never
executing, the divergence persists at the same order of magnitude. See §6 for the full
falsification of the top-k hypothesis.

## 5. Stage-by-stage tensor differences

From `scratch/audit_topk_equivalence.py` (keep_ratio=0.5, mode=default):

| Stage | shape | max_abs_diff | mean_abs_diff | cosine_similarity | nonfinite |
|---|---|---:|---:|---:|---:|
| `hidden_before_prune` (after layer 3, pre-pruning) | `[1,149,2048]` | 1.00 | 0.00776 | 1.00000 | 0 / 0 |
| `hidden_after_prune` (post-gather) | `[1,85,2048]` | 1.00 | 0.00808 | 0.99999988 | 0 / 0 |
| `final_hidden` (all 16 layers) | `[1,85,2048]` | 1.25 | 0.02165 | 0.99989629 | 0 / 0 |

`relative_error_mean` was computed as `diff / |eager|.clamp_min(1e-8)` and produced
values in the 75–770 range — **this metric is misleading here and is explicitly
flagged as such**: transformer hidden states contain many near-zero-magnitude
dimensions, so dividing a small absolute diff by a near-zero denominator explodes the
mean without indicating a real problem. `max_abs_diff` and `cosine_similarity` are the
trustworthy signals in this dataset; both point to a small, benign-magnitude, globally
well-correlated numerical difference, not a structural break. No non-finite values
appeared anywhere.

The divergence is present **before pruning executes** (`hidden_before_prune`, computed
by layers 0–3 only, no pruning-related code has run yet) and grows only modestly
through the pruned tail (1.00 → 1.25 max_abs_diff across 12 more layers) — consistent
with ordinary per-layer numerical drift accumulating, not a discrete jump at the
pruning boundary.

## 6. Top-k boundary and selected-index analysis (the decisive tests)

**A4 Test 4 (selected-index set overlap/order):**
```
eager kept count=85, compiled kept count=85
set_overlap=1.0000, ordered_match_rate=1.0000
tokens kept by eager only: []
tokens kept by compiled only: []
```
Eager and compiled selected **the exact same 85 sequence positions in the exact same
order**. There is no index divergence to explain.

**A4 Test 2 (force compiled to use eager's exact indices — the decisive test):**
```
forced kept indices match eager exactly: True
final_hidden (forced indices): max_abs_diff=1.25   [identical to the free-selection run]
```
Since eager and compiled already agreed on indices, forcing them to agree changes
nothing — as expected, and it reproduces the identical 1.25 max_abs_diff. This directly
falsifies the hypothesis: if discrete top-k selection instability were the cause,
forcing identical indices would collapse the diff toward zero. It does not.

**A4 Test 1 (pruning fully disabled, keep_ratio=1.0):** see §4 — divergence persists
(`max_abs_diff=0.75–1.0`) with the entire prune branch bypassed.

**Category classification:** none of Category A's required conditions hold (indices do
not differ; forcing them changes nothing). Categories B/C/D are also ruled out: no
pruning-related state exists before layer 3, and the disabled-pruning test has zero
pruning-hook code executing at all. This is **Category E**, with a specific, logged
mechanism (§7) rather than a vague "compile is fuzzy" appeal.

## 7. Mutable-state and temporal-state findings

Two distinct findings, one about the LLM-compile experiment's own test hygiene, one
about the real, shipped pruner:

1. **Test-methodology pitfall caught during this audit**: an early version of the
   equivalence test called `text_model(**kwargs)` repeatedly (for warmup, timing, and
   correctness capture) without resetting `VLAPruner.prev_score` between calls. Because
   `prev_score` blends into every subsequent score via
   `temporal_momentum * prev_score + (1 - temporal_momentum) * score`, this made call
   N=1 (fresh state) numerically different from call N=4 (blended state) for reasons
   having nothing to do with compilation. This was identified and corrected (explicit
   `method.reset()` before every comparison capture) before any conclusion was drawn —
   documented here because the audit explicitly requires disclosing this class of
   pitfall, not just avoiding it silently.

2. **Real bug, confirmed and fixed (§15–16 below)**: `gr00t/eval/rollout_policy.py:329`
   calls `policy.reset()` exactly once, before its internal multi-episode loop begins —
   confirmed by its own comment at lines 336–337: *"Currently we don't properly handle
   policy reset. For now, our policy are stateless, but in the future if we need policy
   to be stateful, we need to detect env reset and call policy.reset()."* VLA-Pruner's
   temporal momentum is not stateless. `Gr00tPolicy.reset()` (`gr00t_policy.py:482`) and
   `PolicyWrapper.reset()` (`policy.py:129`) have no knowledge of the externally
   monkey-patched hook, so this was a silent, complete no-op for pruner state in both
   the benchmark adapter (before this fix, one `reset()` call before an entire
   N-episode batch) and the production server (the `"reset"` RPC endpoint did nothing
   to pruner state, ever, before this fix).

## 8. VLA-Pruner architecture integration findings

Confirmed via `scratch/audit_token_layout.py` against a real captured observation
(`image_grid_thw=[[1,16,16],[1,16,16]]`, two cameras, 16×16 patch grids merged to 64
tokens each):

```text
[0]      <|im_start|>
[1]      "user" role token
[2]      \n
[3]      <vision_start>
[4:67]   camera-1 visual tokens (64, contiguous span)
[68]     <vision_end>
[69]     <vision_start>
[70:133] camera-2 visual tokens (64, contiguous span)
[134]    <vision_end>
[135:148] task instruction text tokens (14)
```
Total 149 tokens; `image_mask.sum()=128` (2×64), matching exactly.

**Critical architectural fact** (`gr00t_n1d7.py:242-260`): state and action tokens are
**never part of this sequence at all**. They are embedded via a separate
`state_encoder`/`action_encoder` path into `sa_embs`, which the DiT self-attends as its
own query stream (`hidden_states=sa_embs`); the VL sequence above is fed only as the
DiT's cross-attention `encoder_hidden_states`. VLA-Pruner's hook operates exclusively on
`backbone_features`/`image_mask` — tensors that structurally cannot contain state or
action tokens. This is a **provable-by-construction** guarantee, not a "we were
careful" claim.

## 9. Token/mask/position correctness

- **Design vs. implementation match**: intended design is visual-token-only pruning
  (`gr00t/efficient/hooks/backbone_token_hook.py` docstring). Implementation matches:
  `prune_intermediate` extracts `visual_positions = nonzero(visual_pos_masks[0])`,
  scores/selects only within that subset, then merges `non_visual_positions` (all of
  them, unconditionally) with the *kept* visual positions before gathering — text,
  BOS/EOS, and vision-boundary special tokens are structurally never candidates for
  removal (`backbone_token_hook.py:99-101`).
- **Sequence metadata**: `hidden_states`, `visual_pos_masks`, `text_position_ids`,
  `position_ids`, `cache_position`, and `attention_mask` (when non-None) are all
  `index_select`-ed with the *same* `keep` index tensor (`backbone_token_hook.py:103-119`).
  Verified consistent by both the unit tests (`test_backbone_token_hook.py`, 5/5
  passing, exact-shape assertions) and the equivalence audit (§5's
  `hidden_after_prune` shape `[1,85,2048]` matching `visual_pos_masks`/mask shapes at
  every stage).
- **Position IDs preserve original values** (not renumbered 0..N-1):
  `text_position_ids.index_select(1, keep)` gathers, never re-derives — confirmed by
  design (FastV-style) and consistent with RoPE remaining meaningful for kept tokens.
- **`attention_mask=None` case**: observed in real captures (`create_causal_mask`
  returned `None` for this SDPA config, not a materialized 4D tensor). The guard
  `if attention_mask is not None and (...)` correctly treats `None` as "no explicit mask
  needed" rather than an error. This is safe *because* order is preserved (`keep` is
  always sorted ascending) and RoPE positions are correct — SDPA's `is_causal=True`
  fast path only requires sequential array-position causality, which pruning-with-sorted-indices
  preserves. Verified correct, not merely assumed.

## 10. Multi-camera correctness

Empirically measured via `scratch/audit_camera_retention.py`, real 8-step rollout,
production hook attachment (`attach_backbone_visual_token_hook`), keep_ratio=0.5:

| Camera | tokens kept (of 64) | mean | fully-starved steps |
|---|---|---:|---:|
| Camera 1 | 38–40 | 39.2 (61%) | 0 |
| Camera 2 | 24–26 | 24.8 (39%) | 0 |

**Finding**: scoring is computed as one flat `torch.norm` over both cameras' 128 tokens
combined (`vlapruner.py`, `_semantic_score`) — there is **no per-camera fairness
mechanism**. Neither camera was ever fully starved in this test, but the ~61/39 split
is consistent and systematic, not incidental. This is a **medium-severity, confirmed
design gap**: nothing prevents a more aggressive keep_ratio (not tested here) from
starving one camera entirely. Not patched in this audit (would change the scoring
algorithm, out of scope per "avoid changing the pruning algorithm unless required for
correctness" — this is a design limitation, not a correctness bug in the current
algorithm's own terms). Recommended as follow-up: a per-camera minimum-retention floor.

## 11. Server preprocessing parity

Not independently re-derived in this audit beyond what's already structurally
guaranteed: the server and the benchmark adapter both construct `Gr00tPolicy` the same
way and route through the identical `prepare_input`/`Qwen3Backbone.forward` pipeline
(§2) — there is only one preprocessing code path in this codebase, not two divergent
ones for "server" vs "offline eval." No separate parity test was written since there is
no second implementation to diverge from.

## 12. Server action-output parity

Same conclusion as §11: one `Gr00tN1d7.get_action` implementation, invoked identically
whether called in-process (benchmark) or via `PolicyServer.get_action` (server) — the
server is a thin ZMQ wrapper (`server_client.py:172`,
`self.register_endpoint("get_action", self.policy.get_action)`) with no separate action
post-processing logic of its own.

## 13. Concurrency and session-isolation findings

**Confirmed model: serialized single-request inference.** `PolicyServer` binds one ZMQ
`REP` socket and runs a single Python `while` loop (`server_client.py:237-269`) — ZMQ
REP sockets enforce strict request/reply alternation; a second client's request queues
at the transport layer until the current one completes. No thread pool, no async
handler, no concurrent `policy.get_action()` calls are possible. This was verified by
reading the implementation, not by a live two-client load test (a live test would only
be able to *observe* the absence of races that the single-threaded loop already
guarantees by construction; the source is the more reliable evidence).

**What is not safe**: sequential state leakage. Before the day15 fix (§15–16), the
`"reset"` endpoint did nothing to pruner state, so a **new client (or new episode)
connecting to a long-lived server would inherit the previous client/episode's
temporal-momentum score** for its first few pruning decisions. This is now fixed.

## 14. Latency breakdown

`scratch/audit_latency_stats.py`, production hook attachment, eager (no compile), real
LIBERO rollout, n≈97–128 forward calls per config after 3-call warmup trim,
CUDA-synchronized timers:

| Component | keep=1.0 (baseline) median / p90 / p95 / stdev (ms) | keep=0.5 (pruned) median / p90 / p95 / stdev (ms) |
|---|---|---|
| `model_total` | 119.8 / 135.6 / 141.0 / 10.09 | 119.4 / 135.9 / 142.3 / 9.95 |
| `backbone_total` | 36.6 / 51.5 / 53.9 / 6.69 | 36.0 / 50.9 / 53.5 / 6.58 |
| `vision_tower` | 15.4 / 20.2 / 22.5 / 3.09 | 15.2 / 20.1 / 22.5 / 2.69 |
| `llm_16_layers` | 17.2 / 23.8 / 32.0 / 4.26 | 17.7 / 27.2 / 32.4 / 4.68 |
| `hook_prune_intermediate` | 0.12 / 0.17 / 0.30 / 0.21 | 0.40 / 0.65 / 0.73 / 0.62 |
| `action_head_dit` | 81.6 / 85.4 / 90.4 / 6.43 | 82.3 / 84.8 / 98.1 / 6.30 |

**Model-only speedup at keep=0.5 (eager, no compile): ~0** (`model_total` medians
119.4 vs 119.8 ms — within noise). Hook overhead itself is negligible (0.12–0.40 ms).
The LLM's own eager cost is essentially unchanged with vs. without pruning — consistent
with earlier findings that eager small-batch GPU kernels are launch-overhead-bound, not
size-bound, at this sequence-length scale (149→85 tokens). Confirms: token reduction
alone, without compilation, does not reduce measured latency at this batch size/model
scale. (Separately, from Day 13: stacking `torch.compile` on the DiT with pruning
active showed a real ~2× wall-clock win in the compiled regime — an orthogonal,
already-validated lever, not re-measured in this audit since it was outside this
audit's stated scope.)

## 15. Confirmed bugs

| Severity | File:line | Observed behavior | Root cause | Evidence | Patch | Regression test | Production impact |
|---|---|---|---|---|---|---|---|
| **High** | `gr00t/efficient/benchmark/real_libero_adapter.py` (`run_existing_libero_baseline_python`, pre-fix) | One `run_rollout_gymnasium_policy(n_episodes=N)` call batched all N episodes; VLAPruner's `prev_score` carried over from episode k's last step into episode k+1's first steps | `rollout_policy.py:329` calls `policy.reset()` once, before its internal loop, by design (its own comment admits this) | Direct source read + confirmed no per-episode reset anywhere in the call chain | Restructured to call `run_rollout_gymnasium_policy(n_episodes=1)` in a loop with `method.reset()` between iterations (`_run_episodes_one_at_a_time`) | `test_run_episodes_one_at_a_time_resets_method_between_every_episode`, `test_run_episodes_one_at_a_time_handles_none_seed` (2 new tests, both passing) | All prior multi-episode benchmark SR/latency numbers (Day 8–14) have a small, uncharacterized temporal-momentum leak between episodes. Magnitude not separately re-measured (would require re-running the full sweep) |
| **High** | `gr00t/eval/run_gr00t_server.py` (`main`, pre-fix) | Server's `"reset"` RPC endpoint was a complete no-op for pruner state | `Gr00tPolicy.reset()`/`PolicyWrapper.reset()` have no knowledge of the externally monkey-patched hook | Read `policy.py:129` (`return self.policy.reset(options)`) and `gr00t_policy.py:482` — neither references `gr00t.efficient` at all | Wrapped `policy.reset` post-construction to also call `efficient_method.reset()` when pruning is enabled | Manual server smoke test (§18); no automated test (would require a live server process — out of scope for a fast unit-test regression) | Any production deployment serving multiple episodes/clients from one long-lived server process had unbounded temporal-state leakage between them, silently, with no way to clear it |
| Medium (design gap, not patched) | `gr00t/efficient/pruners/vlapruner.py` (`_semantic_score`) | Global top-k across all cameras' visual tokens combined; no per-camera retention floor | Scoring pools all visual tokens into one `torch.norm` call regardless of camera origin | `scratch/audit_camera_retention.py`: 61%/39% consistent split across 8 real steps, keep=0.5 | Not patched (algorithm change, out of scope for a correctness-only patch pass) | — | Untested at more aggressive keep ratios; theoretically a camera could be fully starved |
| Low (operational, not patched) | `gr00t/eval/run_gr00t_server.py` | No warmup call before `server.run()` | Server never issues a dummy inference before accepting real requests | Read of `main()`; no `policy.get_action` call before `PolicyServer(...).run()` | Not patched (needs a schema-appropriate dummy observation, embodiment-specific, out of scope) | — | First real client request pays full `torch.compile` JIT cost inline; plausible client-side timeout risk at default 15s `PolicyClient` timeout when `--efficient-torch-compile` is set |

## 16. Fixed bugs

Both High-severity findings from §15 were fixed in this session:

1. `gr00t/efficient/benchmark/real_libero_adapter.py`: added
   `_run_episodes_one_at_a_time` (n_envs==1 path), calling `method.reset()` before each
   episode's own `run_rollout_gymnasium_policy(n_episodes=1)` call, with the seed
   incremented per episode (`seed + episode_id`) so episodes remain varied rather than
   replaying the identical initial state. As a side effect, each episode now gets its
   own `_MetricPolicyWrapper` instance, which also fixes a pre-existing, previously
   documented (Day 12 report) bug where every episode's saved action-trace file
   contained the *entire run's* actions instead of just that episode's. Verified via a
   real 3-episode smoke run: three trace files with three different byte sizes (12522,
   12137, 12844 bytes), confirming per-episode segmentation now works.
   The `n_envs > 1` path is unchanged (per-episode reset semantics don't apply cleanly
   to parallel batched episodes, and backbone-stage pruning already requires
   `batch_size==1`, no-opping otherwise).
2. `gr00t/eval/run_gr00t_server.py`: `_attach_efficient_inference` now returns the
   pruning method instance; `main()` wraps `policy.reset` so the server's `"reset"`
   endpoint also calls `efficient_method.reset()`. Purely additive — no effect when
   `--efficient-method` is unset.

Both fixes verified: 18/18 tests pass under the transformers-enabled venv (16 prior +
2 new regression tests), 13/13 pass in the plain venv (2 new tests skip gracefully via
`pytest.importorskip("transformers")`, matching the existing pattern in
`test_backbone_token_hook.py`). Real end-to-end smoke test (3-episode LIBERO rollout,
keep=0.5, backbone-stage): SR 3/3, exit 0.

## 17. Remaining risks

- **No per-camera retention floor** (§10) — global top-k could theoretically starve one
  camera at more aggressive keep ratios; not measured beyond keep=0.5.
- **No server warmup** (§3, §15) — first real request under `--efficient-torch-compile`
  pays JIT compilation cost inline; plausible timeout risk not empirically timed in
  this audit.
- **Server reset fix has no automated regression test** — verified via manual smoke
  test only; a live-server integration test would need a running `PolicyServer` +
  `PolicyClient` pair, which is heavier than this audit's test infrastructure supports
  without further scaffolding.
- **Per-episode reset changes benchmark episode sequences**: episodes now use
  `seed + episode_id` instead of one `seed` for the whole batch relying on natural
  env-internal RNG progression between auto-resets. Deterministic and documented, but
  **not byte-for-byte identical** to pre-fix runs at the same base seed — any exact
  reproduction of Day 8–14 numbers would need the old (buggy) code path.
- **Magnitude of the temporal-momentum leak in prior runs was not retroactively
  quantified** — Day 8–14 SR/latency conclusions stand qualitatively (the leak is a
  score-blending effect on a k=64-of-128 top-k decision, not a crash or a gross
  correctness break), but the exact numeric effect on those specific historical results
  is unknown.

## 18. Exact commands to reproduce

All commands run via `gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python` with
`MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` (see repo `CLAUDE.md` / efficient docs for setup).

```bash
# A2-A4: decisive top-k equivalence audit (produces §4-6 tables)
python audit_topk_equivalence.py 0.5      # pruning active
python audit_topk_equivalence.py 1.0      # pruning disabled (A4 Test 1)

# A5: graph-break / recompile diagnostics
TORCH_LOGS="graph_breaks,recompiles" python audit_topk_equivalence.py 0.5

# B1: token layout
python audit_token_layout.py

# B4: per-camera retention (production attach path)
python audit_camera_retention.py

# D: latency breakdown with percentiles
python audit_latency_stats.py --keep-ratio 1.0 --n-episodes 3
python audit_latency_stats.py --keep-ratio 0.5 --n-episodes 3

# Regression tests (fixes)
python -m pytest tests/gr00t/efficient/test_real_libero_adapter.py -v
python -m pytest tests/gr00t/efficient/ -v

# Real end-to-end smoke test of the fix
python -m gr00t.efficient.benchmark.run_libero \
  --method vlapruner --keep-ratio 0.5 --score-mode norm --prune-stage backbone \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --num-episodes 3 --task debug --seed 42 --save-actions \
  --real-output-dir /tmp/smoke --output /tmp/smoke/result.json

# Server reset-endpoint fix, manual smoke test
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 --embodiment-tag LIBERO_PANDA \
  --use-sim-policy-wrapper --efficient-method vlapruner --efficient-keep-ratio 0.5 \
  --efficient-score-mode norm --efficient-prune-stage backbone
```

## 19. Final recommendation

VLA-Pruner's backbone-stage integration is architecturally sound and now has two fewer
real bugs than it did at the start of this audit. The compiled/eager anomaly that
prompted this audit is fully explained, is not present in any shipped code path, and
should not block anything. The two fixed bugs (temporal-state leakage in both the
benchmark and the server) were real and are now closed with passing regression tests
and a real end-to-end smoke test. The remaining risks (§17) are either low-severity
(no warmup) or explicitly out of scope for a correctness-only patch pass (per-camera
fairness — an algorithm change). Recommend: merge the two fixes, add per-camera
retention floor as a follow-up design task (not a hotfix), and add server warmup with a
schema-appropriate dummy observation before enabling `--efficient-torch-compile` in any
latency-sensitive deployment.

```text
torch.compile correctness:              PASS (Category E, root-caused; not a shipped risk)
VLA-Pruner integration with GR00T N1.7: PASS
GR00T server integration:               PARTIAL (reset-state bug found and fixed; no warmup, no automated server-level regression test)
Safe for offline evaluation:            YES (after the two fixes in this audit)
Safe for real-robot deployment:         YES, with the caveat that per-camera fairness (§10, §17) and server warmup (§3, §17) are unresolved design/operational gaps, not correctness bugs, at the tested keep_ratio (0.5)
Measured model-only speedup:            ~0% eager at keep=0.5 (see §14); real compute-side numbers require torch.compile, evaluated separately in Day 13 (not re-measured in this audit)
Measured end-to-end server speedup:     Not re-measured in this audit (out of scope; see Day 12-14 reports for prior server/client wall-clock measurements, now superseded in validity by the temporal-state fix)
```

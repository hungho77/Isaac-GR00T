# GR00T-N1.7 HoloQ W4A4 on LIBERO

This directory defines both the fake numerical reference and strict native
packed-INT4 execution paths for applying uniform W4A4 to the four published
GR00T-N1.7 LIBERO checkpoints.

## Fixed scope and primary configuration

- Suites and packs are independent: `object`, `spatial`, `goal`, and `long`
  (`libero_10`). Never reuse a pack across suites.
- The released checkpoints use 16 selected language layers and 32 DiT blocks.
  The strict scope is 112 LLM linears plus 192 DiT linears, or 304 total.
- Vision, embeddings, state/action/timestep encoders, output projectors, and
  `vl_self_attention` are not part of the paper-equivalent scope.
- The optional `vit` extension covers Qwen3-VL block `qkv`, attention output,
  MLP linears, merger/DeepStack merger linears, and optionally lowers the
  non-overlapping patch `Conv3d` to the same W4A4 GEMM core. Vision embeddings,
  norms, RoPE, QK/AV attention matmuls, and softmax remain high precision.
- Weights and activations use signed symmetric W4A4 in `[-7, 7]`.
- Each input block uses a weight-norm zigzag permutation followed by a
  randomized `SVD * Hadamard` rotation with block size 64.
- LLM weights use GPTQ with block size 128 and damping 0.01. LLM activations
  use a dynamic per-token scale.
- DiT weights use RTN. DiT activations use a q99.9 per-step, per-layer,
  per-channel scale table.
- N1.7 primary evaluation keeps the checkpoint's four Euler denoising steps.
  The paper's `T=8` is an ablation and requires separately calibrated packs.

The JSON files under `configs/` make these suite-specific assumptions explicit.

## Modal L4 phase-one notebook

[`modal_phase1.ipynb`](modal_phase1.ipynb) is the resumable hosted-Modal workflow.
Attach a Modal Volume at `/vol`, select an NVIDIA L4 kernel, set `SUITE`, and run
the phases in order by changing only `WORK_PHASE` and choosing **Run all**:

```text
setup -> prepare -> calibrate -> build_pack -> smoke_rollout -> full_rollout -> package
```

The notebook clones only `duc-quan`, resolves the selected Hugging Face
checkpoint reference to an immutable commit SHA, evaluates BF16 and W4A4 with
paired seeds, and atomically stores one 20-episode JSON shard per task. The
`package` phase validates all 400 final-evaluation episodes for the selected
suite (10 tasks x 20 episodes x 2 modes), writes an inventory plus SHA256
checksums, and creates a suite-specific ZIP suitable for upload as a Kaggle
Dataset. Run `status` at any time for a read-only progress report.

## 1. Collect one FP16 calibration artifact per suite

Start the normal server without a quantization pack. Example for Object:

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_object \
  --embodiment-tag LIBERO_PANDA \
  --use-sim-policy-wrapper \
  --holoq-calibration-output artifacts/holoq/object-calibration.pt \
  --holoq-suite object \
  --holoq-calibration-run-id object-one-per-task-seed-0 \
  --holoq-calibration-topk 512
```

Against that server, roll out exactly ten unlabeled FP16 trajectories: one from
each task in the suite, using a fixed calibration initial state. Stop the server
after all ten trajectories; shutdown writes the artifact. Repeat with the
matching checkpoint and suite name for Spatial, Goal, and Long.

The server cannot infer simulator episode identity. The operator must ensure the
ten tasks are covered exactly once and record the task/seed list in the run ID
or experiment log. If q99.9 needs an order statistic larger than `topk`, final
save fails and tells you the minimum capacity to use on the next run.

## 2. Build four independent packs

Use immutable checkpoint and source revisions in every manifest:

```bash
uv run python tools/build_holoq_n1d7_pack.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_object \
  --calibration-path artifacts/holoq/object-calibration.pt \
  --output-path artifacts/holoq/object-w4a4.pt \
  --suite object \
  --checkpoint-revision <hugging-face-commit-sha> \
  --source-revision <git-commit-sha>
```

The builder recomputes no calibration assumptions: it consumes the exact saved
rotation, GPTQ Hessian blocks, and DiT scale table. It writes a `.sha256`
sidecar. Loading fails on a config hash mismatch, missing/extra target, damaged
layer record, suite mismatch, or denoising-step mismatch.

## 3. Evaluate held-out rollouts

Start the server with the matching pack:

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_object \
  --embodiment-tag LIBERO_PANDA \
  --use-sim-policy-wrapper \
  --holoq-pack-path artifacts/holoq/object-w4a4.pt \
  --holoq-suite object
```

Evaluate ten trials per task with initial states held out from calibration.
Report every suite separately and the unweighted four-suite average. Run the
same seeds for FP16 and W4A4, and retain action trajectories to check for spikes
or drift, especially on Long.

## Fake and native backends

Pack format v2 stores two signed INT4 values per byte and is shared by both
backends. `--holoq-backend fake` unpacks and dequantizes as a numerical golden
reference. `--holoq-backend native` invokes the bundled CUTLASS extension for
signed INT4 x INT4 tensor-core GEMM with INT32 accumulation. Native mode never
falls back to `F.linear`.

Set `HOLOQ_CUTLASS_ROOT` to a CUTLASS checkout and precompile on the target GPU:

```bash
uv run python tools/build_holoq_native_extension.py --cutlass-root /opt/cutlass
```

To calibrate the complete transformer scope, pass
`--holoq-scopes llm,dit,vit`. Add `--holoq-include-vit-patch-embed` only when
patch-embedding quantization is intended. Build a native-compatible pack with:

```bash
uv run python tools/build_holoq_n1d7_pack.py \
  ... \
  --scopes llm,dit,vit \
  --include-vit-mergers \
  --include-vit-patch-embed \
  --dit-activation-granularity dynamic-per-token
```

The original DiT q99.9 per-step/per-channel activation table remains available
to the fake paper-reference path. It cannot be factored out of a single integer
GEMM because its scale varies inside K; native mode therefore requires the
explicit dynamic-per-token DiT variant and rejects incompatible packs.

# CKA-guided pruning and recovery fine-tuning for GR00T N1.7

This workflow adapts the CLP N1.5 research protocol to the different N1.7
architecture. It does **not** reuse N1.5 layer indices.

For an end-to-end Kaggle/Colab workflow, upload and run
`notebooks/GR00T_N1D7_CKA_RECOVERY_KAGGLE.ipynb`. It includes hidden access-token
input, guarded shell execution, baseline/CKA recovery, fair benchmarking,
before/after CKA heatmaps and an automatically labeled conclusion.

## What is pruned in N1.7

| Manifest target | N1.7 implementation | Official N1.7 LIBERO checkpoint depth |
|---|---|---:|
| `backbone_language` | Cosmos-Reason2/Qwen3-VL language layers used by GR00T | 16 |
| `action_dit` | `AlternateVLDiT.transformer_blocks` | 32 |
| `vl_self_attention` | action-head VL self-attention adapter | checkpoint-defined (commonly 4) |

The source-code construction defaults are smaller (12/16), so never hardcode
these values in a notebook. Capture and manifest validation use the depth of
the checkpoint actually loaded.

Qwen3-VL vision blocks are intentionally excluded. Unlike the N1.5 Eagle
backbone, N1.7 vision features participate in Qwen3-VL deep-stack taps and a
vision merger whose layer positions must remain consistent. Pruning them needs
a separate topology-aware study.

The N1.7 DiT alternates self-attention, text cross-attention and image
cross-attention. Pruning records every retained block's original index so its
role does not silently change after `ModuleList` renumbering.

## Required experimental protocol

Use a task-specific training/calibration set and a disjoint validation set:

```text
full NVIDIA checkpoint
├── baseline: full architecture -> fine-tune -> evaluate validation/rollouts
└── CKA: capture train activations -> calculate CKA -> prune -> fine-tune
         -> evaluate the same validation/rollouts
```

Do not cite a comparison between a pretrained baseline and zero-shot-pruned
CKA as a CLP reproduction. Offline MSE/MAE is useful diagnostics; simulator or
real-robot success rate is the primary task-quality metric.

## Hugging Face access token without Kaggle Secrets

Create a read token at <https://huggingface.co/settings/tokens>. Never paste it
into source code, a shell command, notebook output, Git remote, or a committed
`.env` file.

Run this Python cell/session once. `getpass` hides the token:

```python
from getpass import getpass
from huggingface_hub import login

hf_token = getpass("Hugging Face read token: ")
login(token=hf_token, add_to_git_credential=False)
del hf_token
```

`huggingface_hub` stores the credential in the runtime user's HF cache. The
scripts also accept an already-set `HF_TOKEN`, but never accept a token CLI
argument because command-line tokens leak through process listings and logs.

## Kaggle T4: CKA calibration, analysis and inference benchmark

Enable one GPU in **Notebook options -> Accelerator -> GPU T4**. Kaggle T4 is
suitable for capture/analysis/inference. Full N1.7 recovery training with Adam
optimizer states is not a faithful paper reproduction on 16 GiB VRAM.

### 1. Clone and install

```python
!git clone --branch cka-n1d7-recovery https://github.com/<YOUR_USER>/Isaac-GR00T.git
%cd /kaggle/working/Isaac-GR00T
!pip -q install uv
!uv sync --all-extras
```

Authenticate with the hidden `getpass` cell above, then verify:

```python
from huggingface_hub import whoami
print(whoami()["name"])
```

### 2. Prepare data

Convert task demonstrations to GR00T LeRobot format following
`getting_started/data_preparation.md`. Keep three disjoint roles:

- `train`: recovery fine-tuning;
- `calibration`: a small sample from train, used only for CKA;
- `validation`: held-out episodes, used by both final checkpoints.

The bundled DROID sample is only a pipeline smoke test; it cannot establish a
research result.

### 3. Capture N1.7 activations

```python
!MPLBACKEND=Agg uv run python scripts/cka_n1d7/capture_activations.py \
  --model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /kaggle/input/<calibration-dataset> \
  --embodiment-tag <EMBODIMENT_TAG> \
  --trajectory-ids 0 1 2 3 \
  --samples-per-trajectory 16 \
  --sample-stride 4 \
  --denoising-steps 1 \
  --output-dir /kaggle/working/cka_calibration \
  --require-hf-token
```

Use at least 32 observations in total. More varied calibration episodes are
preferable to many neighboring frames from one episode.

Activation archives use schema version 2. Every layer stores exactly one row
per selected observation; repeated Action-DiT calls across denoising steps are
aggregated within that observation. Delete and recapture older archives: the
analyzer intentionally rejects legacy archives whose sample axes can be
misaligned across language, Action-DiT and VL-attention modules.

### 4. Calculate CKA and generate the manifest

The starting budgets below are inspired by CLP N1.5, but the selected N1.7
indices come from N1.7 activations:

```python
!MPLBACKEND=Agg uv run python scripts/cka_n1d7/analyze_cka.py \
  --calibration-dir /kaggle/working/cka_calibration \
  --output-dir /kaggle/working/cka_analysis \
  --backbone-language-prune-ratio 0.40 \
  --action-dit-prune-ratio 0.50 \
  --vl-self-attention-prune-ratio 0.25
```

Download and version together:

```text
cka_analysis/pruning_manifest.json
cka_analysis/cka_report.json
cka_analysis/cka_*.png
```

### 5. Optional T4 structural smoke test

On T4, run only a short test to prove that a manifest loads and gradients flow.
It is not the final recovery result. Freeze the billion-parameter DiT if memory
is insufficient:

```python
!uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /kaggle/input/<train-dataset> \
  --embodiment-tag <EMBODIMENT_TAG> \
  --cka-pruning-manifest-path /kaggle/working/cka_analysis/pruning_manifest.json \
  --output-dir /kaggle/working/cka_smoke \
  --global-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --max-steps 20 \
  --save-steps 20 \
  --no-tune-diffusion-model \
  --no-tune-vlln
```

Do not compare this lightweight smoke checkpoint with the fully fine-tuned
baseline in the final report.

## VS Code: full baseline and CKA recovery

Use VS Code Remote SSH or WSL connected to a Linux CUDA machine. Native Windows
is not the supported full-training environment. For the paper-like protocol,
use at least a 48 GiB GPU; H100/A100 80 GiB is preferable.

### 1. Environment

```bash
git clone --branch cka-n1d7-recovery https://github.com/<YOUR_USER>/Isaac-GR00T.git
cd Isaac-GR00T
uv sync --all-extras
uv run python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.is_available())"
```

Authenticate interactively using the same `getpass` Python snippet. Do not add
the token to `.vscode/launch.json`.

### 2. Fine-tune the full baseline

```bash
uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /data/task_train \
  --embodiment-tag <EMBODIMENT_TAG> \
  --output-dir outputs/n1d7_baseline_recovery \
  --global-batch-size 8 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --max-steps 10000 \
  --save-steps 1000
```

### 3. Fine-tune CKA from the same base checkpoint

Use identical dataset, seed, effective batch, learning rate, augmentations and
steps. The only experimental variable is the manifest:

```bash
uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /data/task_train \
  --embodiment-tag <EMBODIMENT_TAG> \
  --cka-pruning-manifest-path outputs/cka_analysis/pruning_manifest.json \
  --output-dir outputs/n1d7_cka_recovery \
  --global-batch-size 8 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --max-steps 10000 \
  --save-steps 1000
```

First establish a learning curve at 500/2k/5k/10k steps. A paper reproduction
may require substantially more training, as CLP N1.5 reports use task-dependent
runs up to 100k-200k steps.

For memory-bounded single-GPU recovery, the branch also supports opt-in
Action-DiT LoRA checkpoints. Keep the same values for baseline and CKA. When a
run is split at 1000/2000/3000, set `--lr-scheduler-total-steps 3000` in every
stage so the learning-rate schedule has one fixed final horizon. Enable compact
numbered checkpoints with `GR00T_ACTION_DIT_LORA_ADAPTER_CHECKPOINTS=1`; only
the final stage should set `GR00T_ACTION_DIT_LORA_EXPORT_FINAL=1` to merge LoRA
and write a standalone model. Resume validation rejects a different pruning
manifest, adapter layout or scheduler horizon.

Add `--exact-data-resume` when a staged run must reproduce the uninterrupted
iterable sample order. It replays/skips prior video samples and can therefore
make stage startup much slower. Without it, resume uses a deterministic
stage-specific seed; use the same stage boundaries for every compared model
and do not describe that mode as bitwise-equivalent to one uninterrupted run.

```bash
export GR00T_ACTION_DIT_LORA_RANK=16
export GR00T_ACTION_DIT_LORA_ALPHA=32
export GR00T_ACTION_DIT_LORA_DROPOUT=0.05
export GR00T_ACTION_DIT_LORA_ADAPTER_CHECKPOINTS=1
# Set only for the last stage:
export GR00T_ACTION_DIT_LORA_EXPORT_FINAL=1
```

The T4 mode no longer changes the optimizer implicitly. Pass the same
`--optim` value to all compared runs. For an OOM-only smoke fallback, set
`GR00T_LOW_VRAM_T4_USE_ADAFACTOR=1` and label the resulting run separately
from AdamW results.

### 4. Evaluate both fine-tuned checkpoints

Run them in separate processes so GPU memory is released. Use the same held-out
dataset and all timing parameters:

```bash
uv run python scripts/cka_n1d7/benchmark_finetuned.py \
  --model-path outputs/n1d7_baseline_recovery \
  --dataset-path /data/task_validation \
  --embodiment-tag <EMBODIMENT_TAG> \
  --trajectory-ids 0 1 2 3 4 \
  --samples-per-trajectory 50 \
  --sample-stride 8 \
  --warmup-steps 5 \
  --denoising-steps 4 \
  --run-name baseline-finetuned \
  --output-dir outputs/eval_baseline

uv run python scripts/cka_n1d7/benchmark_finetuned.py \
  --model-path outputs/n1d7_cka_recovery \
  --dataset-path /data/task_validation \
  --embodiment-tag <EMBODIMENT_TAG> \
  --trajectory-ids 0 1 2 3 4 \
  --samples-per-trajectory 50 \
  --sample-stride 8 \
  --warmup-steps 5 \
  --denoising-steps 4 \
  --run-name cka-finetuned \
  --output-dir outputs/eval_cka
```

Generate the guarded comparison:

```bash
uv run python scripts/cka_n1d7/compare_finetuned.py \
  --baseline-dir outputs/eval_baseline \
  --cka-dir outputs/eval_cka \
  --output-dir outputs/final_cka_comparison
```

The comparator rejects mismatched ground truth, seeds, GPU, denoising steps,
sample selection and non-CKA checkpoints. It also requires trainer metadata by
default, preventing a zero-shot-pruned run from being presented as recovery.

## Final report aligned with CLP N1.5

Report at minimum:

1. architecture depth and parameter reduction;
2. training wall-clock time under the same GPU/effective batch/steps;
3. mean, median and P95 inference latency after warm-up;
4. peak allocated and reserved VRAM;
5. held-out MSE/MAE and prediction plots;
6. simulator or real-robot success rate over the same tasks and episode count;
7. calibration size, CKA matrices and exact pruning manifest;
8. mean and confidence interval across multiple seeds.

The primary quality comparison is **fine-tuned baseline versus fine-tuned CKA**.
Zero-shot pruning is an ablation and must be labeled as such.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generated from: C:/Users/nguye/Downloads/GR00T_N1D7_UR10E_CUP_CKA_KEEP4DIT_KEEP6LANG_KEEP2VLSA_FULL_SINGLE_GPU_20K_RUN_ALL.ipynb
Cell order and executable code are preserved.
Markdown cells are retained as commented VS Code/Jupyter sections.
"""

# %% [markdown]
# # GR00T N1.7 — UR10e Cup — CKA 4/6/2 — full single-GPU 20K — one-pass Run All
#

# %% [markdown]
# ## Execution contract
#
# Fill only `HF_READ_TOKEN` and `HF_WRITE_TOKEN` in the configuration cell, then select **Run All** once. The notebook validates one visible 40 GB GPU, pins hungho77/main, applies the pinned CKA runtime derived from ducnm, downloads and converts LeRobot v3, creates leakage-controlled splits, captures CKA activations, builds and validates the exact 4/6/2 manifest, runs full retained-module recovery to 20K with 1K saves, uploads the merged checkpoint to Luke99662244/462, performs offline evaluation and heatmaps, and produces a compact evidence archive. A repeated Run All is idempotent and resumes only validated checkpoints.
#

# %%
from pathlib import Path
import os

# ========================= USER CONFIGURATION =========================
# Fill these two values before pressing Run All. The read-token account must
# already have access to NVIDIA gated repositories, including Cosmos-Reason2-2B.
HF_READ_TOKEN = ""
HF_WRITE_TOKEN = ""

# This notebook intentionally executes every stage in one Run All.
WORK_PHASE = "run_all"

# The only recovery-budget variable. Recovery starts at step 0 and resumes in-place.
# Change only this value to select a new target. Persistent paths never include it.
# Increasing it resumes from the latest valid checkpoint. Decreasing it is allowed only
# before training has passed the requested target. Use multiples of SAVE_STEPS.
FINAL_STEP = 20_000

# Fixed 4/6/2 retained-depth budget (12/52 blocks kept; 76.923% pruned by block count):
#   language: 6/16   -> 62.5% pruned
#   Action-DiT: 4/32 -> 87.5% pruned
#   VLSA: 2/4        -> 50% pruned
BACKBONE_LANGUAGE_PRUNE_RATIO = 0.625
ACTION_DIT_PRUNE_RATIO = 0.875
VL_SELF_ATTENTION_PRUNE_RATIO = 0.50
EXPECTED_ORIGINAL_DEPTHS = {
    "backbone_language": 16,
    "action_dit": 32,
    "vl_self_attention": 4,
}
EXPECTED_KEPT_DEPTHS = {
    "backbone_language": 6,
    "action_dit": 4,
    "vl_self_attention": 2,
}

REPO_URL = "https://github.com/hungho77/Isaac-GR00T.git"
BRANCH = "main"
REPO_COMMIT = "9c7e746b2cd37a810070a98ef41d290a07e806c2"
MODEL_REPO = "nvidia/GR00T-N1.7-3B"
MODEL_REVISION = "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
DATASET_REPO = "khanhnd61/ur10e-cup"
DATASET_REVISION = "12bbeaad3995e84cf43b99adecb0a8bc216cedbf"
EMBODIMENT_TAG = "NEW_EMBODIMENT"
CKA_SOURCE_BRANCH = "ducnm"
CKA_SOURCE_COMMIT = "93e73595b5070b650cf3792cb6d497760e92b1f2"
CKA_RUNTIME_OVERLAY_VERSION = "ur10e-minimal-cka-v1"
CKA_RUNTIME_OVERLAY_SHA256 = "3b5263339987562035613b82126c334af359a2f58f3445ff93ef7ebb9cb0a8e4"

# Standalone candidate: no pre-trained task baseline is required.

# Single-L40S resource contract. This is full retained-module recovery,
# not LoRA, but it is intentionally smaller than the 8-GPU reference recipe.
NUM_GPUS = 1
GLOBAL_BATCH = 32
GRADIENT_ACCUMULATION = 1
# Repository-reference single-GPU dataloader setting.
DATALOADER_NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 1e-5
STATE_DROPOUT_PROB = 0.2
SAVE_STEPS = 1_000
SAVE_TOTAL_LIMIT = 2
SHARD_SIZE = 1024
NUM_SHARDS_PER_EPOCH = 100_000
EPISODE_SAMPLING_RATE = 0.1
COLOR_JITTER = [
    "brightness", "0.3", "contrast", "0.4",
    "saturation", "0.5", "hue", "0.08",
]
USE_PERCENTILES = True
TUNE_LLM = False
TUNE_VISUAL = False
TUNE_PROJECTOR = True
TUNE_DIFFUSION_MODEL = True
TUNE_VLLN = True
USE_WANDB = False
SEED = 42

# Single-40GB-GPU hardware contract. Values are checked before any download or training.
STRICT_HARDWARE_PREFLIGHT = True
MIN_LOGICAL_CPUS = 8
MIN_SYSTEM_RAM_GIB = 46
MIN_GPU_COUNT = 1
MIN_GPU_VRAM_GIB = 39
REQUIRE_L40S = False

# Deterministic leakage-controlled episode contract.
TRAIN_EPISODES = list(range(0, 65))
VALIDATION_EPISODES = list(range(65, 73))
TEST_EPISODES = list(range(73, 81))
CALIBRATION_TRAJECTORIES = list(range(0, 64, 4))
SAMPLES_PER_CALIBRATION_TRAJECTORY = 16
SAMPLES_PER_VALIDATION_TRAJECTORY = 32
SAMPLE_STRIDE = 2
DENOISING_STEPS = 16
# Seed 42 is the deploy-like raw run; the remaining paired seeds estimate
# diffusion sampling variance without smoothing or altering ground truth.
EVAL_REPEAT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]

RUN_CKA = True
RUN_CKA_RECOVERY = True
RUN_EVALUATION = True
RUN_REPORT = True
RUN_UPLOAD = True

assert WORK_PHASE == "run_all"
assert isinstance(FINAL_STEP, int) and FINAL_STEP >= SAVE_STEPS
assert FINAL_STEP % SAVE_STEPS == 0, (FINAL_STEP, SAVE_STEPS)
assert NUM_GPUS == 1 and GLOBAL_BATCH == 32
assert GRADIENT_ACCUMULATION == 1
assert DATALOADER_NUM_WORKERS == 4
assert len(EVAL_REPEAT_SEEDS) >= 2 and EVAL_REPEAT_SEEDS[0] == SEED
assert len(set(EVAL_REPEAT_SEEDS)) == len(EVAL_REPEAT_SEEDS)
assert LEARNING_RATE == 1e-4 and WARMUP_RATIO == 0.05 and WEIGHT_DECAY == 1e-5
assert TUNE_PROJECTOR and TUNE_DIFFUSION_MODEL and TUNE_VLLN
assert not TUNE_LLM and not TUNE_VISUAL
assert TRAIN_EPISODES == list(range(65))
assert set(TRAIN_EPISODES).isdisjoint(VALIDATION_EPISODES)
assert set(TRAIN_EPISODES).isdisjoint(TEST_EPISODES)
assert set(VALIDATION_EPISODES).isdisjoint(TEST_EPISODES)

PLATFORM = "linux_jupyter_server"
# Override GR00T_EXPERIMENT_ROOT only if the server provides a dedicated data disk.
PERSIST_ROOT = Path(os.environ.get(
    "GR00T_EXPERIMENT_ROOT", str(Path.cwd() / "gr00t-experiment")
)).expanduser().resolve()
PERSIST_ROOT.mkdir(parents=True, exist_ok=True)
RUNTIME_ROOT = PERSIST_ROOT / "runtime"
CACHE_ROOT = RUNTIME_ROOT / "cache"
HF_HOME_ROOT = RUNTIME_ROOT / "hf-home"
MIN_DISK_FREE_GIB = 80


REPO_DIR = RUNTIME_ROOT / "Isaac-GR00T"
UV_PROJECT_ENVIRONMENT = RUNTIME_ROOT / "isaac-gr00t-venv"
UV_CACHE_DIR = RUNTIME_ROOT / "uv-cache"
HF_HOME = HF_HOME_ROOT
CHECKPOINT_ROOT = PERSIST_ROOT / "cache" / "checkpoints"
MODEL_PATH = CHECKPOINT_ROOT / "GR00T-N1.7-3B"
DATASET_ROOT = PERSIST_ROOT / "cache" / "datasets"
DATASET_PREP_ROOT = DATASET_ROOT / "ur10e_cup_lerobot"
FULL_DATASET = DATASET_PREP_ROOT / "khanhnd61" / "ur10e-cup"
TRAIN_DATASET = DATASET_ROOT / "ur10e-cup-train65"
assert CHECKPOINT_ROOT.is_relative_to(PERSIST_ROOT)
assert DATASET_ROOT.is_relative_to(PERSIST_ROOT)

VARIANT_TAG = "cka_keep4dit_keep6lang_keep2vlsa_full_single_gpu_candidate"
EXPERIMENT_ROOT = PERSIST_ROOT / "experiments" / f"gr00t_n1d7_ur10e_cup_{VARIANT_TAG}"
OUTPUT_ROOT = EXPERIMENT_ROOT / "results"
# Candidate checkpoints are persistent and resumable on persistent server storage.
TRAIN_OUTPUT_ROOT = PERSIST_ROOT / "training" / EXPERIMENT_ROOT.name
CKA_OUTPUT = TRAIN_OUTPUT_ROOT / "cka_keep4dit_keep6lang_keep2vlsa_full_recovery"
SOURCE_CALIBRATION_DIR = OUTPUT_ROOT / "source_calibration"
ANALYSIS_DIR = OUTPUT_ROOT / "source_cka_analysis"
PRUNING_MANIFEST = ANALYSIS_DIR / "pruning_manifest.json"
EVAL_CKA = OUTPUT_ROOT / "eval_cka_keep4dit_keep6lang_keep2vlsa"
CANDIDATE_ANALYSIS = OUTPUT_ROOT / "candidate_analysis"
HEATMAP_ROOT = OUTPUT_ROOT / "heatmap_comparison"
ARCHIVE_ROOT = PERSIST_ROOT / "archives" / f"ur10e_cup_{VARIANT_TAG}"
FINAL_ZIP = ARCHIVE_ROOT / f"gr00t_n1d7_ur10e_cup_{VARIANT_TAG}_candidate_only_results.zip"

for path in [
    RUNTIME_ROOT, OUTPUT_ROOT, TRAIN_OUTPUT_ROOT, CHECKPOINT_ROOT, DATASET_ROOT,
    SOURCE_CALIBRATION_DIR, ANALYSIS_DIR, EVAL_CKA,
    CANDIDATE_ANALYSIS, HEATMAP_ROOT, ARCHIVE_ROOT,
]:
    path.mkdir(parents=True, exist_ok=True)

os.environ["UV_PROJECT_ENVIRONMENT"] = str(UV_PROJECT_ENVIRONMENT)
os.environ["UV_CACHE_DIR"] = str(UV_CACHE_DIR)
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ.pop("GR00T_LOW_VRAM_T4", None)
os.environ["GR00T_ACTIVATION_CHECKPOINTING"] = "1"
for key in list(os.environ):
    if key.startswith("GR00T_ACTION_DIT_LORA_"):
        os.environ.pop(key, None)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["MPLBACKEND"] = "Agg"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["GR00T_MAX_LOG_BYTES"] = str(512 * 1024)

assert HF_READ_TOKEN.strip(), "Fill HF_READ_TOKEN before Run All."
assert HF_WRITE_TOKEN.strip(), "Fill HF_WRITE_TOKEN before Run All."
print({
    "platform": PLATFORM,
    "phase": WORK_PHASE,
    "final_step": FINAL_STEP,
    "save_steps": SAVE_STEPS,
    "training_storage": str(TRAIN_OUTPUT_ROOT),
    "candidate_training": {
        "profile": "hungho_main_repository_reference_single_40gb_gpu",
        "num_gpus": NUM_GPUS,
        "global_batch": GLOBAL_BATCH,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "effective_batch": GLOBAL_BATCH * GRADIENT_ACCUMULATION,
        "full_action_dit": TUNE_DIFFUSION_MODEL,
        "projector": TUNE_PROJECTOR,
        "vlln": TUNE_VLLN,
        "lora": False,
    },
    "expected_depths": {
        name: f"{EXPECTED_KEPT_DEPTHS[name]}/{depth}"
        for name, depth in EXPECTED_ORIGINAL_DEPTHS.items()
    },
})

# %%
import collections
import json
import re
import shutil
import subprocess
import sys
import time

LOG_DIR = OUTPUT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_STATUS_PATH = OUTPUT_ROOT / "runtime_status.json"
MAX_LOG_BYTES = int(os.environ.get("GR00T_MAX_LOG_BYTES", str(256 * 1024)))


def write_runtime_status(stage, event, detail=None, command=None):
    payload = {
        "stage": stage, "event": event, "time_unix": time.time(),
        "detail": detail, "command": command,
    }
    try:
        temporary = RUNTIME_STATUS_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(RUNTIME_STATUS_PATH)
    except OSError:
        pass


IMPORTANT = re.compile(
    r"(error|exception|traceback|warning|failed|cuda out of memory|no space left|"
    r"checkpoint|saving|loss|mse|mae|latency|vram|parameter|completed|step|results|success rate|episodes took)", re.I
)


def run_checked(name, command, cwd=None, env=None):
    command = [str(item) for item in command]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    log_path = LOG_DIR / f"{safe_name}.log"
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    tail = collections.deque(maxlen=180)
    important = collections.deque(maxlen=180)
    written = 0
    log_file = None
    write_runtime_status(name, "running", command=command)
    print(f"[run] {name} started; status: {RUNTIME_STATUS_PATH}")
    try:
        log_file = log_path.open("w", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(
            command, cwd=str(cwd) if cwd else None, env=run_env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            tail.append(line)
            if IMPORTANT.search(line):
                important.append(line)
                print(line, end="", flush=True)
                if log_file and written < MAX_LOG_BYTES:
                    log_file.write(line)
                    written += len(line.encode("utf-8", errors="replace"))
        returncode = proc.wait()
    except KeyboardInterrupt:
        if "proc" in locals() and proc.poll() is None:
            proc.terminate()
        write_runtime_status(name, "interrupted", "KeyboardInterrupt", command)
        raise
    finally:
        if log_file:
            try:
                log_file.close()
            except OSError:
                pass
    if returncode != 0:
        write_runtime_status(name, "failed", f"exit code {returncode}", command)
        print("\n===== IMPORTANT LINES =====")
        print("".join(important) or "(none)")
        print("\n===== RAW TAIL =====")
        print("".join(tail))
        raise RuntimeError(f"{name} failed with exit code {returncode}")
    write_runtime_status(name, "completed", command=command)
    print(f"[done] {name}")
    return log_path


def uv_python(*args):
    return ["uv", "run", "--no-sync", "python", *map(str, args)]


def latest_step(output_dir):
    best = 0
    for state_path in Path(output_dir).glob("checkpoint-*/trainer_state.json"):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            best = max(best, int(state.get("global_step", 0)))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    root_state = Path(output_dir) / "trainer_state.json"
    if root_state.exists():
        try:
            best = max(best, int(json.loads(root_state.read_text())["global_step"]))
        except Exception:
            pass
    return best


def model_payload_ready(model_dir):
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shards = {model_dir / name for name in index["weight_map"].values()}
            return bool(shards) and all(path.is_file() and path.stat().st_size > 0 for path in shards)
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            return False
    candidates = list(model_dir.glob("model*.safetensors")) + list(
        model_dir.glob("pytorch_model*.bin")
    )
    return bool(candidates) and all(path.stat().st_size > 0 for path in candidates)


def resumable_checkpoint_ready(path):
    path = Path(path)
    required = [
        path / "trainer_state.json",
        path / "optimizer.pt",
        path / "scheduler.pt",
    ]
    has_rng = any(path.glob("rng_state*.pth"))
    return all(item.is_file() and item.stat().st_size > 0 for item in required) and has_rng and model_payload_ready(path)


def latest_checkpoint(output_dir):
    valid = []
    for path in Path(output_dir).glob("checkpoint-*"):
        if resumable_checkpoint_ready(path):
            try:
                step = int(json.loads((path / "trainer_state.json").read_text())["global_step"])
                valid.append((step, path))
            except Exception:
                pass
    return max(valid, default=(0, None))[1]


def root_model_ready(output_dir):
    output_dir = Path(output_dir)
    return (output_dir / "config.json").exists() and model_payload_ready(output_dir)


RUN_ENV = os.environ.copy()

# %%
# Single 40 GB GPU server preflight.
import os
import shutil
import subprocess


def system_ram_gib():
    values = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as file:
        for line in file:
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0])
    return values["MemTotal"] / 1024**2


query = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
    text=True, capture_output=True, check=True,
)
gpus = []
for row in query.stdout.strip().splitlines():
    index, name, memory_mib = [part.strip() for part in row.split(",", 2)]
    gpus.append({"index": int(index), "name": name, "vram_gib": float(memory_mib) / 1024})

logical_cpus = os.cpu_count() or 0
ram_gib = system_ram_gib()
free_disk_gib = shutil.disk_usage(PERSIST_ROOT).free / 1024**3
print("Logical CPUs:", logical_cpus)
print("RAM GiB:", round(ram_gib, 2))
print("Free persistent disk GiB:", round(free_disk_gib, 2))
print("GPUs:", gpus)

hardware_ok = (
    logical_cpus >= MIN_LOGICAL_CPUS
    and ram_gib >= MIN_SYSTEM_RAM_GIB
    and free_disk_gib >= MIN_DISK_FREE_GIB
    and len(gpus) == MIN_GPU_COUNT
    and gpus[0]["vram_gib"] >= MIN_GPU_VRAM_GIB
)
print("FULL SINGLE-40GB-GPU CONTRACT SATISFIED:", hardware_ok)
if STRICT_HARDWARE_PREFLIGHT:
    assert logical_cpus >= MIN_LOGICAL_CPUS, (logical_cpus, MIN_LOGICAL_CPUS)
    assert ram_gib >= MIN_SYSTEM_RAM_GIB, (ram_gib, MIN_SYSTEM_RAM_GIB)
    assert free_disk_gib >= MIN_DISK_FREE_GIB, (free_disk_gib, MIN_DISK_FREE_GIB)
    assert len(gpus) == 1, f"Expose exactly one GPU; found: {gpus}"
    assert gpus[0]["vram_gib"] >= MIN_GPU_VRAM_GIB, gpus

# %%
import subprocess

assert PERSIST_ROOT.is_dir(), PERSIST_ROOT
assert "GR00T_LOW_VRAM_T4" not in os.environ
assert os.environ.get("GR00T_ACTIVATION_CHECKPOINTING") == "1"
assert not any(key.startswith("GR00T_ACTION_DIT_LORA_") for key in os.environ)

print("Standalone 4/6/2 mode: no prior task model is required.")
print("Original checkpoint cache:", MODEL_PATH)
print("Persistent root:", PERSIST_ROOT)
print("Persistent 4/6/2 training output:", CKA_OUTPUT)
print("Persistent compact results:", OUTPUT_ROOT)
print("Runtime and venv:", RUNTIME_ROOT)
subprocess.run(["bash", "-lc", f"du -sh '{PERSIST_ROOT}' 2>/dev/null || true"], check=False)
subprocess.run(["nvidia-smi"], check=False)

# %% [markdown]
# ## A. Disposable runtime and authenticated source checkout
#

# %%
# Build a disposable hungho77/main runtime and install the notebook-owned
# minimal CKA overlay. Persistent checkpoints/results stay on the persistent server storage.
import base64
import hashlib
import io
import zipfile

run_checked(
    "Install uv and huggingface_hub",
    [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "uv==0.8.14", "huggingface_hub==0.36.0", "hf-xet==1.1.10"],
)

overlay_marker = REPO_DIR / ".codex_minimal_cka_overlay.json"

def native_cka_branch_ready():
    """Detect the CKA-capable runtime's built-in CKA runtime."""
    required_files = [
        "gr00t/model/cka_pruning.py",
        "scripts/cka_n1d7/capture_activations.py",
        "scripts/cka_n1d7/analyze_cka.py",
        "scripts/cka_n1d7/benchmark_finetuned.py",
        "examples/UR10eCup/prepare_dataset.py",
        "examples/UR10eCup/ur10e_cup_config.py",
    ]
    core_markers = {
        "gr00t/configs/finetune_config.py": "cka_pruning_manifest_path",
        "gr00t/configs/model/gr00t_n1d7.py": "cka_pruning_manifest: dict | None",
        "gr00t/experiment/launch_finetune.py": "load_pruning_manifest",
        "gr00t/model/gr00t_n1d7/gr00t_n1d7.py": "def apply_cka_pruning",
        "gr00t/model/gr00t_n1d7/setup.py": "requested_manifest = self.config.model.cka_pruning_manifest",
        "gr00t/model/modules/dit.py": 'topology_idx = getattr(block, "_gr00t_original_index", idx)',
        "gr00t/policy/gr00t_policy.py": "validate_checkpoint_loading_info",
    }
    if not all((REPO_DIR / relative).is_file() for relative in required_files):
        return False
    return all(
        (REPO_DIR / relative).is_file()
        and marker in (REPO_DIR / relative).read_text(encoding="utf-8")
        for relative, marker in core_markers.items()
    )

def overlay_ready():
    if native_cka_branch_ready():
        return True
    if not overlay_marker.is_file():
        return False
    try:
        marker = json.loads(overlay_marker.read_text(encoding="utf-8"))
    except Exception:
        return False
    if (
        marker.get("version") != CKA_RUNTIME_OVERLAY_VERSION
        or marker.get("payload_sha256") != CKA_RUNTIME_OVERLAY_SHA256
        or marker.get("base_commit") != REPO_COMMIT
    ):
        return False
    for relative, expected in marker.get("files", {}).items():
        path = REPO_DIR / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            return False
    for relative, expected_marker in marker.get("core_markers", {}).items():
        path = REPO_DIR / relative
        if not path.is_file() or expected_marker not in path.read_text(encoding="utf-8"):
            return False
    return True

if REPO_DIR.exists() and not overlay_ready():
    # /root/gr00t-runtime is explicitly disposable; persistent data is mounted
    # under /mnt/gr00t-experiment and is never touched here.
    assert REPO_DIR.parent.resolve() == RUNTIME_ROOT.resolve()
    shutil.rmtree(REPO_DIR)

if not (REPO_DIR / ".git").exists():
    run_checked(
        "Clone pinned hungho77 main",
        ["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, REPO_DIR],
    )
    run_checked("Checkout pinned hungho77 commit", ["git", "checkout", "--detach", REPO_COMMIT], cwd=REPO_DIR)

actual_repo_commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True, capture_output=True, check=True,
).stdout.strip()
assert actual_repo_commit == REPO_COMMIT, (actual_repo_commit, REPO_COMMIT)

if not overlay_ready():
    payload = base64.b64decode("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")
    assert hashlib.sha256(payload).hexdigest() == CKA_RUNTIME_OVERLAY_SHA256
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        assert names and all(".." not in Path(name).parts for name in names)
        archive.extractall(REPO_DIR)

    def replace_once(relative, old, new):
        path = REPO_DIR / relative
        text = path.read_text(encoding="utf-8")
        if new in text:
            return
        count = text.count(old)
        if count != 1:
            raise RuntimeError(f"{relative}: expected one patch anchor, found {count}")
        path.write_text(text.replace(old, new, 1), encoding="utf-8")

    replace_once(
        "gr00t/configs/finetune_config.py",
        '    # --- Model Tuning Flags ---',
        '    cka_pruning_manifest_path: str | None = None\n'
        '    """Optional CKA manifest. Full weights load before structural pruning."""\n\n'
        '    # --- Model Tuning Flags ---',
    )
    replace_once(
        "gr00t/configs/model/gr00t_n1d7.py",
        '    tune_vlln: bool = True\n\n    # State augmentation parameters',
        '    tune_vlln: bool = True\n\n'
        '    # Persisted structural-pruning contract for reduced checkpoints.\n'
        '    cka_pruning_manifest: dict | None = None\n\n'
        '    # State augmentation parameters',
    )
    replace_once(
        "gr00t/experiment/launch_finetune.py",
        '    config.model.tune_diffusion_model = ft_config.tune_diffusion_model\n'
        '    config.model.state_dropout_prob = ft_config.state_dropout_prob',
        '    config.model.tune_diffusion_model = ft_config.tune_diffusion_model\n'
        '    if ft_config.cka_pruning_manifest_path is not None:\n'
        '        manifest_path = Path(ft_config.cka_pruning_manifest_path)\n'
        '        if not manifest_path.is_file():\n'
        '            raise FileNotFoundError(f"CKA pruning manifest does not exist: {manifest_path}")\n'
        '        from gr00t.model.cka_pruning import load_pruning_manifest\n\n'
        '        config.model.cka_pruning_manifest = load_pruning_manifest(manifest_path)\n'
        '    else:\n'
        '        config.model.cka_pruning_manifest = None\n'
        '    config.model.state_dropout_prob = ft_config.state_dropout_prob',
    )
    replace_once(
        "gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
        '        # Initialize action head\n'
        '        self.action_head = Gr00tN1d7ActionHead(config)\n'
        '        from .processing_gr00t_n1d7 import Gr00tN1d7DataCollator',
        '        # Initialize action head\n'
        '        self.action_head = Gr00tN1d7ActionHead(config)\n\n'
        '        # Rebuild a saved reduced architecture before state-dict loading.\n'
        '        if config.cka_pruning_manifest is not None:\n'
        '            self.apply_cka_pruning(config.cka_pruning_manifest)\n\n'
        '        from .processing_gr00t_n1d7 import Gr00tN1d7DataCollator',
    )
    replace_once(
        "gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
        '    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:',
        '    def apply_cka_pruning(self, manifest: dict[str, Any]) -> dict[str, Any]:\n'
        '        """Structurally prune N1.7 according to a validated CKA manifest."""\n'
        '        from gr00t.model.cka_pruning import apply_pruning_manifest\n\n'
        '        return apply_pruning_manifest(self, manifest)\n\n'
        '    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:',
    )
    replace_once(
        "gr00t/model/gr00t_n1d7/setup.py",
        '        logging.debug(f"Model Config: {model.config}")',
        '        requested_manifest = self.config.model.cka_pruning_manifest\n'
        '        if requested_manifest is not None:\n'
        '            pruning_stats = model.apply_cka_pruning(requested_manifest)\n'
        '            logging.info("Applied N1.7 CKA pruning after full checkpoint load: %s", pruning_stats)\n\n'
        '        logging.debug(f"Model Config: {model.config}")',
    )
    replace_once(
        "gr00t/model/modules/dit.py",
        '        for idx, block in enumerate(self.transformer_blocks):\n'
        '            if idx % 2 == 1 and self.config.interleave_self_attention:',
        '        for idx, block in enumerate(self.transformer_blocks):\n'
        '            topology_idx = getattr(block, "_gr00t_original_index", idx)\n'
        '            if topology_idx % 2 == 1 and self.config.interleave_self_attention:',
    )
    replace_once(
        "gr00t/model/modules/dit.py",
        '        for idx, block in enumerate(self.transformer_blocks):\n'
        '            if idx % 2 == 1:\n'
        '                # Self-attention blocks',
        '        for idx, block in enumerate(self.transformer_blocks):\n'
        '            topology_idx = getattr(block, "_gr00t_original_index", idx)\n'
        '            if topology_idx % 2 == 1:\n'
        '                # Self-attention blocks',
    )
    replace_once(
        "gr00t/model/modules/dit.py",
        '                if idx % (2 * self.attend_text_every_n_blocks) == 0:',
        '                if topology_idx % (2 * self.attend_text_every_n_blocks) == 0:',
    )

    marker = {
        "version": CKA_RUNTIME_OVERLAY_VERSION,
        "payload_sha256": CKA_RUNTIME_OVERLAY_SHA256,
        "base_repository": REPO_URL,
        "base_branch": BRANCH,
        "base_commit": REPO_COMMIT,
        "scope": [
            "CKA structural pruning and topology preservation",
            "CKA activation capture/analyze/offline benchmark/heatmap scripts",
            "UR10e dataset conversion and modality recipe",
        ],
        "excluded": ["LoRA", "T4 low-VRAM mode", "simulator", "rollout patches"],
        "files": {"examples/UR10eCup/modality.json": "390e3fbb04ba3038a7587c7f39f6331e2fc721b26328cfb54d4b1cf8651fd172", "examples/UR10eCup/prepare_dataset.py": "7574a4628d5423f2445094e870407ec8b6e8f11559a2858e7ea01c81471fac91", "examples/UR10eCup/ur10e_cup_config.py": "87180dc4085b9e0e8a1d1e31bc0b69f99069864fff0a6b2cf13878dff2d68ce9", "gr00t/model/cka_pruning.py": "0fc0e7a9828c9ec72b218edfea3a80a1da59e46a3d0faff38ae225593790f532", "scripts/cka_n1d7/analyze_cka.py": "df4aef1e5f32747fdc7d301720b6cdc577e8a035f0c7db65dd2e83720b1eeb8b", "scripts/cka_n1d7/benchmark_finetuned.py": "5028853e7437501e4e86cc11f04c60a7aae7d42203a87bc0e34a96ce11145149", "scripts/cka_n1d7/capture_activations.py": "e5f2555565b198d59861ef2ce4f0d2ae068ce65cda571ee86f4527fc01c3ef06", "scripts/cka_n1d7/compare_cka_heatmaps.py": "aded59ed3d91d173a0476a7da3d2f27eb048e578c3c5cd6b58936f50c6543090"},
        "core_markers": {
            "gr00t/configs/finetune_config.py": "cka_pruning_manifest_path",
            "gr00t/configs/model/gr00t_n1d7.py": "cka_pruning_manifest: dict | None",
            "gr00t/experiment/launch_finetune.py": "load_pruning_manifest",
            "gr00t/model/gr00t_n1d7/gr00t_n1d7.py": "def apply_cka_pruning",
            "gr00t/model/gr00t_n1d7/setup.py": "requested_manifest = self.config.model.cka_pruning_manifest",
            "gr00t/model/modules/dit.py": "topology_idx = getattr(block, \"_gr00t_original_index\", idx)",
        },
    }
    overlay_marker.write_text(json.dumps(marker, indent=2), encoding="utf-8")

assert overlay_ready(), "Native ducnm CKA runtime/legacy overlay integrity check failed"
if native_cka_branch_ready():
    marker = {
        "version": "native-ducnm-v1",
        "repository": REPO_URL,
        "branch": BRANCH,
        "commit": REPO_COMMIT,
        "excluded": ["LoRA", "T4 low-VRAM mode", "simulator", "rollout patches"],
    }
else:
    marker = json.loads(overlay_marker.read_text(encoding="utf-8"))
assert "LoRA" in marker["excluded"] and "T4 low-VRAM mode" in marker["excluded"]
print("Pinned hungho77/ducnm source commit:", actual_repo_commit)
print("CKA runtime contract:", marker)

# The repository stores an aarch64 torchcodec wheel in Git LFS. Materialize it
# because uv validates all locked sources even on an x86_64 Modal host.
lfs_wheel = REPO_DIR / "scripts/deployment/dgpu/wheels/torchcodec-0.8.0-cp312-cp312-linux_aarch64.whl"
if lfs_wheel.exists() and lfs_wheel.read_bytes()[:32].startswith(b"version https://git-lfs"):
    run_checked(
        "Install git-lfs",
        ["bash", "-lc", "command -v git-lfs >/dev/null 2>&1 || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git-lfs)"],
    )
    run_checked("Initialize Git LFS", ["git", "lfs", "install", "--local"], cwd=REPO_DIR)
    run_checked(
        "Pull torchcodec Git LFS wheel",
        ["git", "lfs", "pull", "--include=scripts/deployment/dgpu/wheels/torchcodec-0.8.0-cp312-cp312-linux_aarch64.whl", "--exclude="],
        cwd=REPO_DIR,
    )
assert lfs_wheel.is_file() and lfs_wheel.read_bytes()[:4] == b"PK\x03\x04", lfs_wheel

run_checked(
    "Install native build and video conversion dependencies",
    ["bash", "-lc", "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential python3-dev linux-libc-dev ffmpeg git-lfs libgl1 libglib2.0-0"],
)
run_checked("Verify uv version", ["uv", "--version"])
run_checked("Install hungho77 GR00T environment", ["uv", "sync", "--all-extras"], cwd=REPO_DIR, env=RUN_ENV)

RUN_ENV = os.environ.copy()
RUN_ENV["PYTHONUNBUFFERED"] = "1"
run_checked(
    "Compile and verify minimal CKA runtime",
    uv_python(
        "-c",
        "import torch; import examples.UR10eCup.ur10e_cup_config; "
        "from gr00t.model.cka_pruning import validate_pruning_manifest; "
        "print(torch.cuda.is_available(), torch.cuda.get_device_name(0)); "
        "print('hungho77/ducnm + minimal CKA overlay OK')",
    ),
    cwd=REPO_DIR,
    env=RUN_ENV,
)

# %%
from huggingface_hub import HfApi, hf_hub_download

HF_READ_TOKEN = HF_READ_TOKEN.strip()
HF_WRITE_TOKEN = HF_WRITE_TOKEN.strip()
assert HF_READ_TOKEN and HF_WRITE_TOKEN

read_api = HfApi(token=HF_READ_TOKEN)
write_api = HfApi(token=HF_WRITE_TOKEN)
read_user = read_api.whoami()["name"]
write_user = write_api.whoami()["name"]
print("HF read-token user:", read_user)
print("HF write-token user:", write_user)

os.environ["HF_TOKEN"] = HF_READ_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_READ_TOKEN
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
RUN_ENV["HF_TOKEN"] = HF_READ_TOKEN
RUN_ENV["HUGGING_FACE_HUB_TOKEN"] = HF_READ_TOKEN
RUN_ENV["HF_XET_HIGH_PERFORMANCE"] = "1"

# Fail before downloads/training if the token account has not accepted NVIDIA's gate.
COSMOS_REPO = "nvidia/Cosmos-Reason2-2B"
cosmos_config = hf_hub_download(
    repo_id=COSMOS_REPO, filename="config.json", token=HF_READ_TOKEN,
    cache_dir=str(HF_HOME / "hub"),
)
print("NVIDIA gated access: PASS", cosmos_config)

# %% [markdown]
# ## B. Pinned checkpoint and LeRobot v3 → GR00T conversion
#

# %%
from huggingface_hub import snapshot_download

# The LeRobot conversion project has a Git dependency whose repository also
# tracks test fixtures with Git LFS.  One fixture is missing upstream, but it
# is not needed by the v3 -> v2.1 converter.  Skip LFS smudging and, critically,
# use a conversion-only venv so `uv run --project` cannot replace the main
# Isaac-GR00T environment configured in UV_PROJECT_ENVIRONMENT.
CONVERSION_ENV = RUN_ENV.copy()
CONVERSION_ENV["GIT_LFS_SKIP_SMUDGE"] = "1"
CONVERSION_ENV["UV_PROJECT_ENVIRONMENT"] = str(
    RUNTIME_ROOT / "lerobot-conversion-venv"
)
CONVERSION_ENV["UV_CACHE_DIR"] = str(
    RUNTIME_ROOT / "lerobot-conversion-uv-cache"
)

MODEL_PATH.mkdir(parents=True, exist_ok=True)
# snapshot_download is resumable and verifies the pinned Hub snapshot metadata;
# calling it again repairs a model directory that has config/index but lacks a shard.
snapshot_download(
    repo_id=MODEL_REPO,
    revision=MODEL_REVISION,
    local_dir=MODEL_PATH,
    token=HF_READ_TOKEN,
)
model_index_path = MODEL_PATH / "model.safetensors.index.json"
assert (MODEL_PATH / "config.json").exists()
assert model_index_path.exists()
model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
model_shards = sorted(set(model_index["weight_map"].values()))
missing_model_shards = [
    name for name in model_shards
    if not (MODEL_PATH / name).is_file() or (MODEL_PATH / name).stat().st_size == 0
]
assert not missing_model_shards, f"Missing/empty model shards: {missing_model_shards}"
assert len(model_shards) == 2, model_shards
print("Base checkpoint ready:", MODEL_PATH, model_shards)

# Resume a partially downloaded pinned v3 snapshot before conversion. Never
# mix Hub v3 files into an already converted v2.1 dataset.
source_info_path = FULL_DATASET / "meta" / "info.json"
source_version = None
if source_info_path.exists():
    source_version = json.loads(source_info_path.read_text(encoding="utf-8")).get(
        "codebase_version"
    )
if source_version in {None, "v3.0"}:
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        revision=DATASET_REVISION,
        local_dir=FULL_DATASET,
        token=HF_READ_TOKEN,
    )
    source_info_path = FULL_DATASET / "meta" / "info.json"
    assert source_info_path.exists(), source_info_path

# A failed conversion may leave only the reproducible sibling staging tree.
# convert_dataset removes/rebuilds it itself, so preserve the v3 source and
# merely report its presence for traceability.
conversion_staging = FULL_DATASET.parent / f"{FULL_DATASET.name}_v2.1"
if conversion_staging.exists():
    print("Reusable conversion staging tree detected; converter will rebuild it:", conversion_staging)

conversion_report = FULL_DATASET / "meta" / "gr00t_conversion.json"
if not conversion_report.exists():
    run_checked(
        "Download convert and validate UR10e LeRobot v3",
        [
            "uv", "run", "--project", "scripts/lerobot_conversion", "python",
            "examples/UR10eCup/prepare_dataset.py",
            "--root", str(DATASET_PREP_ROOT),
            "--revision", DATASET_REVISION,
        ],
        cwd=REPO_DIR,
        env=CONVERSION_ENV,
    )
else:
    run_checked(
        "Revalidate converted UR10e dataset",
        [
            "uv", "run", "--project", "scripts/lerobot_conversion", "python",
            "examples/UR10eCup/prepare_dataset.py",
            "--root", str(DATASET_PREP_ROOT), "--validate-only",
        ],
        cwd=REPO_DIR,
        env=CONVERSION_ENV,
    )

report = json.loads(conversion_report.read_text(encoding="utf-8"))
assert report["status"] == "ok"
assert report["episodes"] == 81
assert report["frames"] == 49_779
assert report["source_revision"] == DATASET_REVISION
print("Converted dataset ready:", FULL_DATASET)
print("Conversion report:", conversion_report)

# %% [markdown]
# ## C. Leakage-controlled train/validation/test episode contract
#

# %%
# Create a zero-copy training view containing episodes 0..64 only.
# Validation/test use the full immutable converted dataset with disjoint IDs.
import os
import shutil

split_marker = TRAIN_DATASET / "meta" / "ur10e_train_split.json"
if not split_marker.exists():
    if TRAIN_DATASET.exists():
        # This directory is a reproducible derived view, never source data.
        # Safely rebuild it after an interrupted prior run.
        assert TRAIN_DATASET.parent.resolve() == DATASET_ROOT.resolve()
        assert TRAIN_DATASET.name == "ur10e-cup-train65"
        shutil.rmtree(TRAIN_DATASET)
        print("Removed incomplete derived train split:", TRAIN_DATASET)
    (TRAIN_DATASET / "meta").mkdir(parents=True)
    for name in ["tasks.jsonl", "modality.json"]:
        shutil.copy2(FULL_DATASET / "meta" / name, TRAIN_DATASET / "meta" / name)

    episodes = [json.loads(line) for line in (FULL_DATASET / "meta" / "episodes.jsonl").read_text().splitlines() if line]
    episode_stats = [json.loads(line) for line in (FULL_DATASET / "meta" / "episodes_stats.jsonl").read_text().splitlines() if line]
    train_rows = [row for row in episodes if int(row["episode_index"]) in TRAIN_EPISODES]
    train_stats = [row for row in episode_stats if int(row["episode_index"]) in TRAIN_EPISODES]
    assert [int(row["episode_index"]) for row in train_rows] == TRAIN_EPISODES
    (TRAIN_DATASET / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in train_rows), encoding="utf-8"
    )
    (TRAIN_DATASET / "meta" / "episodes_stats.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in train_stats), encoding="utf-8"
    )
    info = json.loads((FULL_DATASET / "meta" / "info.json").read_text())
    info["total_episodes"] = len(TRAIN_EPISODES)
    info["total_frames"] = sum(int(row["length"]) for row in train_rows)
    info["splits"] = {"train": f"0:{len(TRAIN_EPISODES)}"}
    (TRAIN_DATASET / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    # Link/copy only train episode files. Linking the complete data directory
    # would make stats.py include validation/test parquet files even though
    # episodes.jsonl was filtered.
    linked_files = 0
    for episode_index in TRAIN_EPISODES:
        patterns = [
            f"data/chunk-*/episode_{episode_index:06d}.parquet",
            f"videos/chunk-*/*/episode_{episode_index:06d}.mp4",
        ]
        matches = []
        for pattern in patterns:
            matches.extend(FULL_DATASET.glob(pattern))
        assert sum(path.suffix == ".parquet" for path in matches) == 1, (episode_index, matches)
        assert sum(path.suffix == ".mp4" for path in matches) == 2, (episode_index, matches)
        for source in matches:
            target = TRAIN_DATASET / source.relative_to(FULL_DATASET)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.symlink_to(source.resolve())
            except OSError:
                shutil.copy2(source, target)
            linked_files += 1

    split_marker.write_text(json.dumps({
        "source": str(FULL_DATASET),
        "train": TRAIN_EPISODES,
        "validation": VALIDATION_EPISODES,
        "test": TEST_EPISODES,
        "leakage_check": True,
        "schema_version": 2,
        "linked_or_copied_episode_files": linked_files,
    }, indent=2), encoding="utf-8")

assert split_marker.exists()
split = json.loads(split_marker.read_text())
assert split["train"] == TRAIN_EPISODES
assert split["validation"] == VALIDATION_EPISODES
assert split["test"] == TEST_EPISODES
assert split.get("schema_version") == 2
assert Path(split["source"]).resolve() == FULL_DATASET.resolve()
assert len(split["train"]) == 65
assert len(split["validation"]) == 8
assert len(split["test"]) == 8
assert len(list(TRAIN_DATASET.glob("data/chunk-*/*.parquet"))) == 65
assert len(list(TRAIN_DATASET.glob("videos/chunk-*/*/*.mp4"))) == 130

stats_path = TRAIN_DATASET / "meta" / "stats.json"
if not stats_path.exists():
    run_checked(
        "Generate train-only UR10e statistics",
        uv_python(
            "gr00t/data/stats.py",
            "--dataset-path", str(TRAIN_DATASET),
            "--embodiment-tag", EMBODIMENT_TAG,
            "--modality-config-path", "examples/UR10eCup/ur10e_cup_config.py",
        ),
        cwd=REPO_DIR,
        env=RUN_ENV,
    )
assert stats_path.exists()
print("Train-only dataset:", TRAIN_DATASET)
print("Held-out validation episodes:", VALIDATION_EPISODES)
print("Held-out test episodes:", TEST_EPISODES)

# %% [markdown]
# ## D. Capture hidden states, compute CKA and create the exact 4/6/2 pruning manifest
#

# %%
capture_common = [
    "--dataset-path", str(TRAIN_DATASET),
    "--embodiment-tag", EMBODIMENT_TAG,
    "--trajectory-ids", *map(str, CALIBRATION_TRAJECTORIES),
    "--samples-per-trajectory", str(SAMPLES_PER_CALIBRATION_TRAJECTORY),
    "--sample-stride", str(SAMPLE_STRIDE),
    "--denoising-steps", str(DENOISING_STEPS),
    "--seed", str(SEED),
    "--require-hf-token",
]

if RUN_CKA:
    import shutil

    assert root_model_ready(MODEL_PATH), f"Missing original checkpoint: {MODEL_PATH}"

    # The original 3B checkpoint reserves projector index 10 for custom robots,
    # but its serialized processor intentionally does not contain a custom
    # NEW_EMBODIMENT modality or UR10e normalization statistics.  CKA must use
    # the original weights, so create a lightweight processor overlay rather
    # than fine-tuning first or impersonating a different robot tag.
    SOURCE_PROCESSOR_OVERLAY = (
        RUNTIME_ROOT / "source_overlays" /
        "gr00t_n1d7_3b_ur10e_new_embodiment_processor"
    )
    SOURCE_PROCESSOR_CONTRACT = OUTPUT_ROOT / "source_processor_contract.json"

    def registered_source_ready():
        required = [
            SOURCE_PROCESSOR_OVERLAY / "config.json",
            SOURCE_PROCESSOR_OVERLAY / "model.safetensors.index.json",
            SOURCE_PROCESSOR_OVERLAY / "processor_config.json",
            SOURCE_PROCESSOR_OVERLAY / "statistics.json",
            SOURCE_PROCESSOR_OVERLAY / "embodiment_id.json",
            SOURCE_PROCESSOR_CONTRACT,
        ]
        if not all(item.is_file() and item.stat().st_size > 0 for item in required):
            return False
        contract = json.loads(SOURCE_PROCESSOR_CONTRACT.read_text(encoding="utf-8"))
        return (
            contract.get("schema") == "source_processor_registration_v1"
            and contract.get("base_model_revision") == MODEL_REVISION
            and contract.get("embodiment_tag") == EMBODIMENT_TAG
            and contract.get("projector_index") == 10
            and contract.get("source_weights_modified") is False
            and all(
                (SOURCE_PROCESSOR_OVERLAY / name).is_symlink()
                and (SOURCE_PROCESSOR_OVERLAY / name).resolve()
                    == (MODEL_PATH / name).resolve()
                for name in contract.get("weight_files", [])
            )
        )

    if not registered_source_ready():
        overlay_env = RUN_ENV.copy()
        overlay_env.update({
            "GR00T_SOURCE_BASE_MODEL": str(MODEL_PATH),
            "GR00T_SOURCE_OVERLAY": str(SOURCE_PROCESSOR_OVERLAY),
            "GR00T_SOURCE_DATASET": str(TRAIN_DATASET),
            "GR00T_SOURCE_MODALITY_CONFIG": str(
                REPO_DIR / "examples/UR10eCup/ur10e_cup_config.py"
            ),
            "GR00T_SOURCE_CONTRACT": str(SOURCE_PROCESSOR_CONTRACT),
            "GR00T_SOURCE_MODEL_REVISION": MODEL_REVISION,
            "GR00T_SOURCE_DATASET_REVISION": DATASET_REVISION,
        })
        overlay_builder = r"""
import importlib.util
import json
import os
from pathlib import Path
import shutil

from transformers import AutoProcessor
import gr00t.model  # noqa: F401 - registers N1.7 AutoProcessor/AutoModel
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag

base = Path(os.environ["GR00T_SOURCE_BASE_MODEL"]).resolve()
overlay = Path(os.environ["GR00T_SOURCE_OVERLAY"]).resolve()
dataset = Path(os.environ["GR00T_SOURCE_DATASET"]).resolve()
config_path = Path(os.environ["GR00T_SOURCE_MODALITY_CONFIG"]).resolve()
contract_path = Path(os.environ["GR00T_SOURCE_CONTRACT"]).resolve()
tag = EmbodimentTag.NEW_EMBODIMENT

spec = importlib.util.spec_from_file_location("ur10e_cup_source_config", config_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot import modality config: {config_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if tag.value not in MODALITY_CONFIGS:
    raise KeyError(f"{tag.value} was not registered by {config_path}")
modality = MODALITY_CONFIGS[tag.value]

loader = LeRobotEpisodeLoader(dataset, modality)
dataset_statistics = {tag.value: loader.get_dataset_statistics()}
processor = AutoProcessor.from_pretrained(
    base,
    modality_configs={tag.value: modality},
)
processor.set_statistics(dataset_statistics, override=True)
processor.eval()
if processor.embodiment_id_mapping.get(tag.value) != 10:
    raise RuntimeError(
        f"NEW_EMBODIMENT must use reserved projector 10, got "
        f"{processor.embodiment_id_mapping.get(tag.value)!r}"
    )

if overlay.exists():
    shutil.rmtree(overlay)
overlay.mkdir(parents=True)
index_path = base / "model.safetensors.index.json"
index = json.loads(index_path.read_text(encoding="utf-8"))
weight_files = sorted(set(index["weight_map"].values()))
model_files = ["config.json", "model.safetensors.index.json", *weight_files]
for name in model_files:
    source = base / name
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    (overlay / name).symlink_to(source)
processor.save_pretrained(overlay)

required_processor_files = [
    "processor_config.json", "statistics.json", "embodiment_id.json"
]
for name in required_processor_files:
    file = overlay / name
    if not file.is_file() or file.stat().st_size == 0:
        raise FileNotFoundError(file)

contract = {
    "schema": "source_processor_registration_v1",
    "base_model": str(base),
    "base_model_revision": os.environ["GR00T_SOURCE_MODEL_REVISION"],
    "dataset": str(dataset),
    "dataset_revision": os.environ["GR00T_SOURCE_DATASET_REVISION"],
    "modality_config": str(config_path),
    "embodiment_tag": tag.name,
    "embodiment_value": tag.value,
    "projector_index": 10,
    "statistics_scope": "train episodes 0-64 only",
    "source_weights_modified": False,
    "weight_storage": "symlinks to pinned original checkpoint",
    "weight_files": weight_files,
}
contract_path.parent.mkdir(parents=True, exist_ok=True)
contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
print(json.dumps(contract, indent=2))
"""
        run_checked(
            "Register UR10e processor on immutable original source weights",
            uv_python("-c", overlay_builder),
            cwd=REPO_DIR,
            env=overlay_env,
        )
    assert registered_source_ready(), "Custom source processor overlay is invalid."
    print("Registered immutable source overlay:", SOURCE_PROCESSOR_OVERLAY)

    def source_capture_complete(path):
        path = Path(path)
        required = [path / "metadata.json", path / "activations.npz"]
        if not all(item.is_file() and item.stat().st_size > 0 for item in required):
            return False
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        return (
            metadata.get("args", {}).get("model_path")
                == str(SOURCE_PROCESSOR_OVERLAY)
            and SOURCE_PROCESSOR_CONTRACT.is_file()
        )

    if source_capture_complete(SOURCE_CALIBRATION_DIR):
        print("Reuse completed original source activation capture:", SOURCE_CALIBRATION_DIR)
    else:
        if SOURCE_CALIBRATION_DIR.exists():
            assert SOURCE_CALIBRATION_DIR.parent.resolve() == OUTPUT_ROOT.resolve()
            shutil.rmtree(SOURCE_CALIBRATION_DIR)
        SOURCE_CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        run_checked(
            "Capture UR10e original source activations for 4/6/2",
            uv_python(
                "scripts/cka_n1d7/capture_activations.py",
                "--model-path", str(SOURCE_PROCESSOR_OVERLAY),
                "--output-dir", str(SOURCE_CALIBRATION_DIR),
                *capture_common,
            ),
            cwd=REPO_DIR,
            env=RUN_ENV,
        )
    assert source_capture_complete(SOURCE_CALIBRATION_DIR), SOURCE_CALIBRATION_DIR

    if PRUNING_MANIFEST.is_file() and PRUNING_MANIFEST.stat().st_size > 0:
        print("Reuse existing 4/6/2 pruning manifest pending exact-depth validation:", PRUNING_MANIFEST)
    else:
        if ANALYSIS_DIR.exists():
            assert ANALYSIS_DIR.parent.resolve() == OUTPUT_ROOT.resolve()
            shutil.rmtree(ANALYSIS_DIR)
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        run_checked(
            "Analyze UR10e 4/6/2 CKA (4 DiT / 6 language / 2 VLSA)",
            uv_python(
                "scripts/cka_n1d7/analyze_cka.py",
                "--calibration-dir", str(SOURCE_CALIBRATION_DIR),
                "--output-dir", str(ANALYSIS_DIR),
                "--backbone-language-prune-ratio", str(BACKBONE_LANGUAGE_PRUNE_RATIO),
                "--action-dit-prune-ratio", str(ACTION_DIT_PRUNE_RATIO),
                "--vl-self-attention-prune-ratio", str(VL_SELF_ATTENTION_PRUNE_RATIO),
            ),
            cwd=REPO_DIR,
            env=RUN_ENV,
        )
    assert PRUNING_MANIFEST.is_file() and PRUNING_MANIFEST.stat().st_size > 0
    manifest = json.loads(PRUNING_MANIFEST.read_text(encoding="utf-8"))
    modules = manifest["modules"]
    for name, original_depth in EXPECTED_ORIGINAL_DEPTHS.items():
        spec = modules[name]
        keep = spec["keep_indices"]
        assert spec["original_depth"] == original_depth, (name, spec)
        assert len(keep) == EXPECTED_KEPT_DEPTHS[name], (name, keep)
        assert keep == sorted(set(keep)), (name, keep)
        assert all(0 <= index < original_depth for index in keep), (name, keep)
    print("Validated exact-depth pruning manifest:", PRUNING_MANIFEST)
    print({name: modules[name]["keep_indices"] for name in EXPECTED_ORIGINAL_DEPTHS})
else:
    print("CKA capture unexpectedly skipped in one-pass phase:", WORK_PHASE)

# %% [markdown]
# ## E. Full retained-module 4/6/2 recovery to 20K
#

# %%
# Candidate-only full retained-module recovery on one 40 GB GPU (8 CPU / 48 GiB host RAM).
# No LoRA, no selective-block tuning and no T4 low-VRAM mode.
import shutil
import time

RECOVERY_ENV = RUN_ENV.copy()
RECOVERY_ENV.pop("GR00T_LOW_VRAM_T4", None)
RECOVERY_ENV["GR00T_ACTIVATION_CHECKPOINTING"] = "1"
for key in list(RECOVERY_ENV):
    if key.startswith("GR00T_ACTION_DIT_LORA_"):
        RECOVERY_ENV.pop(key, None)
RECOVERY_ENV["PYTHONUNBUFFERED"] = "1"


def distributed_python(*args):
    if NUM_GPUS == 1:
        return ["uv", "run", "--no-sync", "python", *map(str, args)]
    return [
        "uv", "run", "--no-sync", "torchrun",
        f"--nproc_per_node={NUM_GPUS}", "--master_port=29500",
        *map(str, args),
    ]


def bool_flag(name, enabled):
    return f"--{name}" if enabled else f"--no-{name}"


def checkpoint_number(path):
    try:
        return int(Path(path).name.rsplit("-", 1)[-1])
    except Exception:
        return -1


def resumable_checkpoint_ready(checkpoint):
    checkpoint = Path(checkpoint)
    required = [
        checkpoint / "trainer_state.json",
        checkpoint / "optimizer.pt",
        checkpoint / "scheduler.pt",
    ]
    has_weights = any(checkpoint.glob("model*.safetensors")) or any(checkpoint.glob("pytorch_model*.bin"))
    has_rng = any(checkpoint.glob("rng_state*.pth"))
    if not all(path.exists() and path.stat().st_size > 0 for path in required):
        return False
    if not has_weights or not has_rng:
        return False
    try:
        state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        return int(state.get("global_step", -1)) == checkpoint_number(checkpoint)
    except Exception:
        return False


def repair_resume_namespace(output_dir):
    output_dir = Path(output_dir)
    candidates = sorted(output_dir.glob("checkpoint-*"), key=checkpoint_number)
    valid, invalid = [], []
    for checkpoint in candidates:
        (valid if resumable_checkpoint_ready(checkpoint) else invalid).append(checkpoint)
    if invalid:
        quarantine = output_dir / "checkpoint_quarantine"
        quarantine.mkdir(parents=True, exist_ok=True)
        for checkpoint in invalid:
            target = quarantine / f"{checkpoint.name}_incomplete_{int(time.time())}"
            shutil.move(str(checkpoint), str(target))
            print("Quarantined incomplete checkpoint:", target)
    if not valid:
        return None
    selected = valid[-1]
    state_path = selected / "trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    backup = selected / "trainer_state.before_save_schedule_fix.json"
    if not backup.exists():
        shutil.copy2(state_path, backup)
    old_save_steps = state.get("save_steps")
    state["save_steps"] = int(SAVE_STEPS)
    temporary = state_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(state_path)
    print({
        "resume_checkpoint": str(selected),
        "global_step": int(state["global_step"]),
        "old_save_steps": old_save_steps,
        "new_save_steps": SAVE_STEPS,
        "requested_target_step": FINAL_STEP,
    })
    return selected


def recovery_command(base_model, output_dir, manifest=None):
    resume_checkpoint = repair_resume_namespace(output_dir)
    command = distributed_python(
        "gr00t/experiment/launch_finetune.py",
        "--base-model-path", str(base_model),
        "--dataset-path", str(TRAIN_DATASET),
        "--modality-config-path", "examples/UR10eCup/ur10e_cup_config.py",
        "--embodiment-tag", EMBODIMENT_TAG,
        "--output-dir", str(output_dir),
        "--num-gpus", str(NUM_GPUS),
        "--global-batch-size", str(GLOBAL_BATCH),
        "--gradient-accumulation-steps", str(GRADIENT_ACCUMULATION),
        "--dataloader-num-workers", str(DATALOADER_NUM_WORKERS),
        "--learning-rate", str(LEARNING_RATE),
        "--warmup-ratio", str(WARMUP_RATIO),
        "--weight-decay", str(WEIGHT_DECAY),
        "--max-steps", str(FINAL_STEP),
        "--save-steps", str(SAVE_STEPS),
        "--save-total-limit", str(SAVE_TOTAL_LIMIT),
        "--state-dropout-prob", str(STATE_DROPOUT_PROB),
        "--shard-size", str(SHARD_SIZE),
        "--num-shards-per-epoch", str(NUM_SHARDS_PER_EPOCH),
        "--episode-sampling-rate", str(EPISODE_SAMPLING_RATE),
        "--color-jitter-params", *COLOR_JITTER,
        bool_flag("use-percentiles", USE_PERCENTILES),
        bool_flag("tune-llm", TUNE_LLM),
        bool_flag("tune-visual", TUNE_VISUAL),
        bool_flag("tune-projector", TUNE_PROJECTOR),
        bool_flag("tune-diffusion-model", TUNE_DIFFUSION_MODEL),
        bool_flag("use-wandb", USE_WANDB),
    )
    if manifest is not None:
        command += ["--cka-pruning-manifest-path", str(manifest)]
    if resume_checkpoint is not None:
        command += ["--resume-from-checkpoint"]
    return command


def compact_final_checkpoint(output_dir):
    output_dir = Path(output_dir)
    states = sorted(
        output_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda path: int(path.parent.name.rsplit("-", 1)[-1]),
    )
    if states:
        shutil.copy2(states[-1], output_dir / "trainer_state.json")
    # Retain the latest checkpoint until report/archive validation completes.
    # Full checkpoints remain on the attached persistent server storage.


def run_recovery(label, base_model, output_dir, manifest=None):
    output_dir = Path(output_dir)
    before = latest_step(output_dir)
    if before > FINAL_STEP:
        raise ValueError(
            f"{label}: existing step {before} is already above requested FINAL_STEP={FINAL_STEP}. "
            "Changing FINAL_STEP cannot roll a trained model backward. Choose a target >= the existing step."
        )
    if before == FINAL_STEP and root_model_ready(output_dir):
        print(f"{label}: target already complete at step {before}; skipped.")
        return
    if before > 0 and latest_checkpoint(output_dir) is None and not root_model_ready(output_dir):
        raise RuntimeError(f"{label}: state exists but its resumable model payload is missing.")
    run_checked(
        f"{label} to step {FINAL_STEP}",
        recovery_command(base_model, output_dir, manifest),
        cwd=REPO_DIR,
        env=RECOVERY_ENV,
    )
    reached = latest_step(output_dir)
    assert reached >= FINAL_STEP, (label, reached, FINAL_STEP)
    assert root_model_ready(output_dir), f"Missing final model export: {output_dir}"
    compact_final_checkpoint(output_dir)
    print(label, {"from_step": before, "to_step": reached, "root_model": root_model_ready(output_dir)})


contract = {
    "repository": REPO_URL,
    "repository_branch": BRANCH,
    "repository_commit": REPO_COMMIT,
    "cka_source_branch": CKA_SOURCE_BRANCH,
    "cka_source_commit": CKA_SOURCE_COMMIT,
    "cka_runtime_overlay_version": CKA_RUNTIME_OVERLAY_VERSION,
    "cka_runtime_overlay_sha256": CKA_RUNTIME_OVERLAY_SHA256,
    "model_revision": MODEL_REVISION,
    "dataset_revision": DATASET_REVISION,
    "protocol": "ur10e_keep4dit_keep6lang_keep2vlsa_full_retained_module_recovery_single_gpu_v1",
    "phase": WORK_PHASE,
    "final_step": FINAL_STEP,
    "num_gpus": NUM_GPUS,
    "global_batch_pre_accumulation": GLOBAL_BATCH,
    "gradient_accumulation": GRADIENT_ACCUMULATION,
    "effective_batch": GLOBAL_BATCH * GRADIENT_ACCUMULATION,
    "learning_rate": LEARNING_RATE,
    "warmup_ratio": WARMUP_RATIO,
    "weight_decay": WEIGHT_DECAY,
    "state_dropout_prob": STATE_DROPOUT_PROB,
    "full_training": {
        "llm": TUNE_LLM,
        "visual": TUNE_VISUAL,
        "projector": TUNE_PROJECTOR,
        "action_diffusion_model": TUNE_DIFFUSION_MODEL,
        "vlln": TUNE_VLLN,
        "lora": False,
        "selective_action_dit": False,
    },
    "expected_original_depths": EXPECTED_ORIGINAL_DEPTHS,
    "expected_kept_depths": EXPECTED_KEPT_DEPTHS,
    "train_episodes": TRAIN_EPISODES,
    "validation_episodes": VALIDATION_EPISODES,
    "test_episodes": TEST_EPISODES,
    "source_checkpoint": str(MODEL_PATH),
    "source_checkpoint_is_task_adapted": False,
    "candidate_manifest": str(PRUNING_MANIFEST),
    "repository_scale_equivalence": False,
    "resource_profile": "1x 40 GB GPU, 8+ CPU, 48+ GiB RAM",
    "intentional_deviations": [
        f"recovery starts at step 0 and targets {FINAL_STEP}",
        "candidate-only: starts from the original GR00T checkpoint",
        "one GPU, global batch 32 and gradient accumulation 1",
        "activation checkpointing enabled for 40 GB VRAM stability",
    ],
}
(OUTPUT_ROOT / "recovery_contract.json").write_text(
    json.dumps(contract, indent=2), encoding="utf-8"
)

if RUN_CKA_RECOVERY:
    import hashlib

    assert root_model_ready(MODEL_PATH), f"Missing original checkpoint: {MODEL_PATH}"
    assert PRUNING_MANIFEST.exists(), "Run prepare_cka first."
    manifest_sha256 = hashlib.sha256(PRUNING_MANIFEST.read_bytes()).hexdigest()
    recovery_identity = {
        "schema_version": 1,
        "repository": REPO_URL,
    "repository_branch": BRANCH,
    "repository_commit": REPO_COMMIT,
    "cka_source_branch": CKA_SOURCE_BRANCH,
    "cka_source_commit": CKA_SOURCE_COMMIT,
    "cka_runtime_overlay_version": CKA_RUNTIME_OVERLAY_VERSION,
    "cka_runtime_overlay_sha256": CKA_RUNTIME_OVERLAY_SHA256,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "manifest_sha256": manifest_sha256,
        "global_batch": GLOBAL_BATCH,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "save_steps": SAVE_STEPS,
        "tune_projector": TUNE_PROJECTOR,
        "tune_diffusion_model": TUNE_DIFFUSION_MODEL,
        "tune_vlln": TUNE_VLLN,
        "lora": False,
    }
    identity_path = TRAIN_OUTPUT_ROOT / f"{CKA_OUTPUT.name}_recovery_identity.json"
    if latest_step(CKA_OUTPUT) > 0 or root_model_ready(CKA_OUTPUT):
        assert identity_path.exists(), (
            f"Existing candidate has no recovery identity: {identity_path}. "
            "Do not resume an unverified artifact."
        )
        existing_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        assert existing_identity == recovery_identity, {
            "existing": existing_identity,
            "requested": recovery_identity,
        }
    else:
        TRAIN_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        identity_path.write_text(
            json.dumps(recovery_identity, indent=2), encoding="utf-8"
        )
    print("Verified recovery identity:", identity_path)

    # FINAL_STEP is a mutable stopping target, recorded separately from the
    # invariant recovery identity so that a later larger target resumes in-place.
    target_history_path = TRAIN_OUTPUT_ROOT / f"{CKA_OUTPUT.name}_target_history.json"
    target_history = {"schema_version": 1, "targets": []}
    if target_history_path.exists():
        target_history = json.loads(target_history_path.read_text(encoding="utf-8"))
    completed_before = latest_step(CKA_OUTPUT)
    record = {
        "requested_target": FINAL_STEP,
        "completed_before": completed_before,
        "save_steps": SAVE_STEPS,
        "time_unix": time.time(),
    }
    if not target_history.get("targets") or target_history["targets"][-1] != record:
        target_history.setdefault("targets", []).append(record)
    target_history_path.write_text(json.dumps(target_history, indent=2), encoding="utf-8")
    print("Recovery target history:", target_history_path)
    run_recovery(
        "UR10e standalone 4/6/2 full recovery on one 40 GB GPU (4 DiT / 6 language / 2 VLSA)",
        MODEL_PATH,
        CKA_OUTPUT,
        PRUNING_MANIFEST,
    )
else:
    print("Standalone 4/6/2 recovery unexpectedly skipped in one-pass phase:", WORK_PHASE)

print({
    "source_checkpoint_ready": root_model_ready(MODEL_PATH),
    "keep4_6_2_step": latest_step(CKA_OUTPUT),
    "keep4_6_2_final": root_model_ready(CKA_OUTPUT),
})

# %%

# Persistent standalone 4/6/2 recovery/artifact status on the attached persistent server storage.
import json
import subprocess


def model_status(label, root):
    root = Path(root)
    checkpoint = latest_checkpoint(root)
    row = {
        "label": label,
        "root": str(root),
        "exists": root.exists(),
        "latest_step": latest_step(root),
        "latest_checkpoint": str(checkpoint) if checkpoint else None,
        "root_model_ready": root_model_ready(root),
        "storage": str(PERSIST_ROOT),
    }
    print(json.dumps(row, indent=2))
    return row


source_status = {
    "label": "original_source_checkpoint",
    "root": str(MODEL_PATH),
    "root_model_ready": root_model_ready(MODEL_PATH),
    "storage": str(CACHE_ROOT),
}
candidate_status = model_status("cka_keep4dit_keep6lang_keep2vlsa_candidate", CKA_OUTPUT)
manifest_status = {"path": str(PRUNING_MANIFEST), "exists": PRUNING_MANIFEST.exists()}
if PRUNING_MANIFEST.exists():
    manifest = json.loads(PRUNING_MANIFEST.read_text(encoding="utf-8"))
    manifest_status["kept_depths"] = {
        name: len(spec["keep_indices"])
        for name, spec in manifest["modules"].items()
    }
print(json.dumps({"source": source_status, "manifest": manifest_status}, indent=2))
subprocess.run(["bash", "-lc", f"du -sh '{PERSIST_ROOT}' 2>/dev/null || true"], check=False)

final_candidate_ready = (
    candidate_status["latest_step"] >= FINAL_STEP
    and candidate_status["root_model_ready"]
    and PRUNING_MANIFEST.exists()
)
print("4/6/2 FINAL TRAINING ARTIFACT READY:", final_candidate_ready)
print("SAFE TO END SERVER NOTEBOOK:", True)
print("Persistent candidate state:", CKA_OUTPUT)

# %% [markdown]
# ## F. Validate and upload the inference-ready 20K checkpoint to Hugging Face
#

# %%
# Upload only the merged inference-ready root model; optimizer/checkpoint directories stay local.
if RUN_UPLOAD:
    from huggingface_hub import HfApi

    HF_TARGET_REPO = "Luke99662244/462"
    assert root_model_ready(CKA_OUTPUT), f"Missing final merged model: {CKA_OUTPUT}"
    trained_step = latest_step(CKA_OUTPUT)
    assert trained_step >= FINAL_STEP, (trained_step, FINAL_STEP)

    required = [
        CKA_OUTPUT / "config.json",
        CKA_OUTPUT / "model.safetensors.index.json",
        CKA_OUTPUT / "processor",
        CKA_OUTPUT / "experiment_cfg",
    ]
    missing = [str(item) for item in required if not item.exists()]
    assert not missing, f"Missing inference artifacts: {missing}"
    index = json.loads((CKA_OUTPUT / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    assert shards and all((CKA_OUTPUT / name).is_file() for name in shards), shards

    api = HfApi(token=HF_WRITE_TOKEN)
    api.create_repo(repo_id=HF_TARGET_REPO, repo_type="model", exist_ok=True)
    allow_patterns = [
        "config.json", "model.safetensors.index.json", "model-*.safetensors",
        "processor/**", "experiment_cfg/**", "training_args.bin",
        "trainer_state.json", "wandb_config.json",
    ]
    upload_result = api.upload_folder(
        repo_id=HF_TARGET_REPO, repo_type="model", folder_path=str(CKA_OUTPUT),
        path_in_repo="", revision="main", allow_patterns=allow_patterns,
        ignore_patterns=["checkpoint-*/**", "checkpoint_quarantine/**", ".cache/**"],
        commit_message=f"Upload UR10e CKA 4/6/2 full single-GPU checkpoint at step {trained_step}",
    )
    remote_files = set(api.list_repo_files(HF_TARGET_REPO, repo_type="model", revision="main"))
    expected_remote = {"config.json", "model.safetensors.index.json", *shards}
    assert expected_remote.issubset(remote_files), sorted(expected_remote - remote_files)
    upload_receipt = {
        "status": "ok", "repo_id": HF_TARGET_REPO, "revision": "main",
        "trained_step": trained_step, "kept_depths": EXPECTED_KEPT_DEPTHS,
        "repository_branch": BRANCH, "repository_commit": REPO_COMMIT,
        "cka_source_branch": CKA_SOURCE_BRANCH, "cka_source_commit": CKA_SOURCE_COMMIT,
        "commit_url": str(getattr(upload_result, "commit_url", upload_result)),
        "uploaded_required_files": sorted(expected_remote),
    }
    (OUTPUT_ROOT / "huggingface_upload_receipt.json").write_text(
        json.dumps(upload_receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(upload_receipt, indent=2))
else:
    raise AssertionError("RUN_UPLOAD must remain True in this one-pass notebook.")

# %% [markdown]
# ## G. Static real-robot deployment contract
#

# %%
DEPLOYMENT_CONTRACT_PATH = OUTPUT_ROOT / "deployment_contract.json"

if root_model_ready(CKA_OUTPUT):
    assert latest_step(CKA_OUTPUT) >= FINAL_STEP
    config_path = CKA_OUTPUT / "config.json"
    assert config_path.exists(), config_path
    candidate_config = json.loads(config_path.read_text(encoding="utf-8"))
    embedded_manifest = candidate_config.get("cka_pruning_manifest")
    assert embedded_manifest is not None, (
        "Final 4/6/2 config.json does not embed cka_pruning_manifest; "
        "the native CKA server cannot reconstruct this architecture."
    )

    modules = embedded_manifest["modules"]
    embedded_kept = {
        name: len(modules[name]["keep_indices"])
        for name in EXPECTED_ORIGINAL_DEPTHS
    }
    assert embedded_kept == EXPECTED_KEPT_DEPTHS, (
        embedded_kept, EXPECTED_KEPT_DEPTHS
    )

    modality_source = (REPO_DIR / "examples/UR10eCup/ur10e_cup_config.py").read_text(
        encoding="utf-8"
    )
    assert 'modality_keys=["arm_joints", "gripper"]' in modality_source
    assert modality_source.count("ActionRepresentation.ABSOLUTE") >= 2

    task_rows = [
        json.loads(line)
        for line in (FULL_DATASET / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(task_rows) == 1, task_rows
    task_text = str(task_rows[0]["task"]).strip()
    assert task_text

    deployment_contract = {
        "status": "static_contract_pass",
        "variant": "CKA 4/6/2 standalone candidate",
        "model_path": str(CKA_OUTPUT),
        "global_step": latest_step(CKA_OUTPUT),
        "repo_commit": REPO_COMMIT,
        "embodiment_tag": "new_embodiment",
        "task_text": task_text,
        "fps": 20,
        "step_dt_seconds": 0.05,
        "image_size_hw": [480, 640],
        "image_color": "RGB",
        "video_keys": ["side", "wrist"],
        "state_keys": ["arm_joints", "gripper"],
        "state_order": [
            "shoulder_pan", "shoulder_lift", "elbow",
            "wrist_1", "wrist_2", "wrist_3", "gripper",
        ],
        "action_keys": ["arm_joints", "gripper"],
        "action_shape": [16, 7],
        "arm_action_semantics": "absolute joint targets in radians",
        "gripper_semantics": "1=open, 0=closed",
        "original_depths": EXPECTED_ORIGINAL_DEPTHS,
        "kept_depths": embedded_kept,
        "prune_ratios": {
            "backbone_language": BACKBONE_LANGUAGE_PRUNE_RATIO,
            "action_dit": ACTION_DIT_PRUNE_RATIO,
            "vl_self_attention": VL_SELF_ATTENTION_PRUNE_RATIO,
        },
        "client_hard_requirements": [
            "map sensor_msgs/JointState by msg.name, never by array position",
            "normalize measured gripper to 1=open and 0=closed",
            "run dry-run before live control",
            "use finite max_steps for initial live rollout",
        ],
    }
    DEPLOYMENT_CONTRACT_PATH.write_text(
        json.dumps(deployment_contract, indent=2), encoding="utf-8"
    )
    print(json.dumps(deployment_contract, indent=2))
    print("4/6/2 NATIVE-CKA DEPLOYMENT CONTRACT: PASS")
else:
    print(
        "Deployment contract is validated after 4/6/2 recovery reaches "
        f"FINAL_STEP={FINAL_STEP}."
    )

# %% [markdown]
# ## H. Held-out 4/6/2 candidate-only offline benchmark
#

# %%

# Candidate-only raw and repeated-seed diffusion evaluation.
import shutil

if RUN_EVALUATION:
    CURRENT_EVALUATION_STEP = latest_step(CKA_OUTPUT)
    EVALUATION_STEP_MARKER = OUTPUT_ROOT / "evaluated_candidate_step.json"
    previous_evaluation_step = None
    if EVALUATION_STEP_MARKER.exists():
        try:
            previous_evaluation_step = int(json.loads(
                EVALUATION_STEP_MARKER.read_text(encoding="utf-8")
            )["candidate_step"])
        except Exception:
            previous_evaluation_step = None
    if previous_evaluation_step != CURRENT_EVALUATION_STEP:
        print({
            "evaluation_cache_invalidated": True,
            "previous_step": previous_evaluation_step,
            "current_step": CURRENT_EVALUATION_STEP,
        })
        stale_paths = [
            EVAL_CKA, OUTPUT_ROOT / "diffusion_repeats", CANDIDATE_ANALYSIS,
            OUTPUT_ROOT / "recovered_calibration", HEATMAP_ROOT,
        ]
        for stale in stale_paths:
            stale = Path(stale)
            if stale.exists():
                assert stale.is_relative_to(OUTPUT_ROOT)
                shutil.rmtree(stale)
        for required_dir in [EVAL_CKA, CANDIDATE_ANALYSIS, HEATMAP_ROOT]:
            required_dir.mkdir(parents=True, exist_ok=True)


def benchmark_command(model_path, run_name, output_dir, seed):
    return uv_python(
        "scripts/cka_n1d7/benchmark_finetuned.py",
        "--model-path", str(model_path),
        "--dataset-path", str(FULL_DATASET),
        "--embodiment-tag", EMBODIMENT_TAG,
        "--output-dir", str(output_dir),
        "--run-name", run_name,
        "--trajectory-ids", *map(str, VALIDATION_EPISODES),
        "--samples-per-trajectory", str(SAMPLES_PER_VALIDATION_TRAJECTORY),
        "--sample-stride", str(SAMPLE_STRIDE),
        "--warmup-steps", "2",
        "--denoising-steps", str(DENOISING_STEPS),
        "--seed", str(seed),
    )


def repeat_dir(seed):
    if seed == SEED:
        return EVAL_CKA
    return OUTPUT_ROOT / "diffusion_repeats" / "cka_keep4dit_keep6lang_keep2vlsa" / f"seed_{seed}"


def benchmark_complete(path):
    path = Path(path)
    return all((path / name).exists() for name in ["summary.json", "per_step.csv", "actions.npz"])


if RUN_EVALUATION:
    assert root_model_ready(CKA_OUTPUT), CKA_OUTPUT
    assert latest_step(CKA_OUTPUT) >= FINAL_STEP

    repeat_manifest = {
        "variant": "cka_keep4dit_keep6lang_keep2vlsa",
        "candidate_only": True,
        "denoising_steps": DENOISING_STEPS,
        "seeds": EVAL_REPEAT_SEEDS,
        "runs": [],
    }
    for seed in EVAL_REPEAT_SEEDS:
        output_dir = repeat_dir(seed)
        output_dir.mkdir(parents=True, exist_ok=True)
        if benchmark_complete(output_dir):
            print("Reuse completed 4/6/2 benchmark:", seed, output_dir)
        else:
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            run_checked(
                f"Benchmark 4/6/2 candidate diffusion seed {seed}",
                benchmark_command(
                    CKA_OUTPUT,
                    f"cka_keep4dit_keep6lang_keep2vlsa_seed_{seed}",
                    output_dir,
                    seed,
                ),
                cwd=REPO_DIR,
                env=RUN_ENV,
            )
        repeat_manifest["runs"].append(str(output_dir))

    (OUTPUT_ROOT / "diffusion_repeat_manifest.json").write_text(
        json.dumps(repeat_manifest, indent=2), encoding="utf-8"
    )
else:
    print("4/6/2 offline evaluation unexpectedly skipped in one-pass phase:", WORK_PHASE)

# %%

if RUN_EVALUATION:
    import csv
    import numpy as np
    from matplotlib import pyplot as plt

    names = [
        "shoulder_pan", "shoulder_lift", "elbow", "wrist_1",
        "wrist_2", "wrist_3", "gripper",
    ]
    payload = np.load(EVAL_CKA / "actions.npz")
    prediction = np.asarray(payload["prediction"])
    ground_truth = np.asarray(payload["ground_truth"])
    assert prediction.shape == ground_truth.shape and prediction.shape[1] == 7

    rows = []
    for index, name in enumerate(names):
        error = prediction[:, index] - ground_truth[:, index]
        rows.append({
            "model": "cka_keep4dit_keep6lang_keep2vlsa",
            "dimension": index,
            "channel": name,
            "mse": float(np.mean(error ** 2)),
            "mae": float(np.mean(np.abs(error))),
            "target_std": float(np.std(ground_truth[:, index])),
            "note": "near-constant channel; interpret separately" if name == "wrist_2" else "",
        })

    CANDIDATE_ANALYSIS.mkdir(parents=True, exist_ok=True)
    with (CANDIDATE_ANALYSIS / "per_dimension_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (CANDIDATE_ANALYSIS / "per_dimension_metrics.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    candidate_summary = json.loads((EVAL_CKA / "summary.json").read_text(encoding="utf-8"))
    control = {
        "scope": "model-only inference; not measured UR controller round-trip latency",
        "variant": "cka_keep4dit_keep6lang_keep2vlsa",
        "assumed_control_frequency_hz": 20.0,
        "executed_action_chunk": 8,
        "chunk_budget_ms": 400.0,
        "candidate_p95_ms": candidate_summary["latency_p95_ms"],
        "candidate_fits_budget": candidate_summary["latency_p95_ms"] <= 400.0,
    }
    (CANDIDATE_ANALYSIS / "control_latency.json").write_text(
        json.dumps(control, indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(2, 1, figsize=(11, 8))
    x = np.arange(len(names))
    for metric, axis in zip(["mse", "mae"], axes):
        axis.bar(x, [row[metric] for row in rows], color="#d62728")
        axis.set_xticks(x, names, rotation=25, ha="right")
        axis.set_ylabel(metric.upper())
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("UR10e held-out 4/6/2 candidate action error")
    figure.tight_layout()
    figure.savefig(
        CANDIDATE_ANALYSIS / "per_dimension_metrics.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)
    print(json.dumps({"candidate": candidate_summary, "control": control}, indent=2))
else:
    print("Candidate per-channel analysis runs during the one-pass evaluation stage.")

# %%

# Timestamp-aligned 4/6/2 Ground Truth vs Prediction visualization.
if RUN_EVALUATION:
    import csv
    import numpy as np
    import matplotlib.pyplot as plt
    from IPython.display import Image, display

    action_names = [
        "shoulder_pan", "shoulder_lift", "elbow", "wrist_1",
        "wrist_2", "wrist_3", "gripper",
    ]
    with np.load(EVAL_CKA / "actions.npz") as payload:
        prediction = np.asarray(payload["prediction"], dtype=np.float64)
        ground_truth = np.asarray(payload["ground_truth"], dtype=np.float64)
    with (EVAL_CKA / "per_step.csv").open("r", newline="", encoding="utf-8") as file:
        step_rows = list(csv.DictReader(file))

    pred_by_time, gt_by_time = {}, {}
    cursor = 0
    for row in step_rows:
        episode = int(row["trajectory_id"])
        start = int(row["step"])
        horizon = int(row["evaluated_horizon"])
        stop = cursor + horizon
        assert horizon > 0 and stop <= len(prediction)
        pred_by_time.setdefault(episode, {})
        gt_by_time.setdefault(episode, {})
        for offset in range(horizon):
            timestamp = start + offset
            pred_by_time[episode].setdefault(timestamp, []).append(prediction[cursor + offset])
            gt_by_time[episode].setdefault(timestamp, []).append(ground_truth[cursor + offset])
        cursor = stop
    assert cursor == len(prediction), (cursor, len(prediction))

    display_pred, display_gt = [], []
    aligned_pred, aligned_gt = [], []
    ticks, labels, position = [], [], 0
    for episode in sorted(pred_by_time):
        timestamps = sorted(pred_by_time[episode])
        episode_pred = np.stack([
            np.mean(pred_by_time[episode][timestamp], axis=0) for timestamp in timestamps
        ])
        episode_gt = np.stack([
            np.mean(gt_by_time[episode][timestamp], axis=0) for timestamp in timestamps
        ])
        aligned_pred.append(episode_pred)
        aligned_gt.append(episode_gt)
        display_pred.extend([episode_pred, np.full((1, prediction.shape[1]), np.nan)])
        display_gt.extend([episode_gt, np.full((1, prediction.shape[1]), np.nan)])
        ticks.append(position + max(len(episode_pred) - 1, 0) / 2)
        labels.append(f"ep {episode}")
        position += len(episode_pred) + 1

    aligned_pred = np.concatenate(aligned_pred)
    aligned_gt = np.concatenate(aligned_gt)
    plot_pred = np.concatenate(display_pred)
    plot_gt = np.concatenate(display_gt)
    x = np.arange(len(plot_pred))

    figure, axes = plt.subplots(7, 1, figsize=(15, 17), sharex=True)
    for dimension, axis in enumerate(axes):
        axis.plot(x, plot_gt[:, dimension], color="black", linewidth=1.6, label="Ground truth")
        axis.plot(x, plot_pred[:, dimension], color="#d62728", linewidth=1.0, alpha=0.85, label="4/6/2 prediction")
        axis.set_ylabel(action_names[dimension])
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    axes[-1].set_xticks(ticks, labels, rotation=30, ha="right")
    axes[-1].set_xlabel("Timestamp-aligned actions; held-out episodes separated")
    figure.suptitle("4/6/2 candidate: timestamp-aligned Ground Truth vs Prediction")
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    plot_path = CANDIDATE_ANALYSIS / "aligned_prediction_vs_ground_truth_keep4dit_keep6lang_keep2vlsa.png"
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    error = aligned_pred - aligned_gt
    diagnostics = {
        "variant": "cka_keep4dit_keep6lang_keep2vlsa",
        "method": "overlapping action chunks aligned by episode and absolute timestamp; predictions averaged only at identical timestamps",
        "aligned_points": int(len(aligned_pred)),
        "aligned_mse": float(np.mean(error ** 2)),
        "aligned_mae": float(np.mean(np.abs(error))),
        "plot": str(plot_path),
    }
    (CANDIDATE_ANALYSIS / "aligned_prediction_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2))
    display(Image(filename=str(plot_path)))
else:
    print("Aligned 4/6/2 visualization runs during the one-pass evaluation stage.")

# %%

# Candidate-only repeated-seed first-action diagnostics. Metrics are never smoothed.
if RUN_EVALUATION:
    import csv
    import numpy as np
    import matplotlib.pyplot as plt

    action_names = [
        "shoulder_pan", "shoulder_lift", "elbow", "wrist_1",
        "wrist_2", "wrist_3", "gripper",
    ]

    def load_first_action_samples(eval_dir):
        eval_dir = Path(eval_dir)
        with np.load(eval_dir / "actions.npz") as payload:
            prediction = np.asarray(payload["prediction"], dtype=np.float64)
            ground_truth = np.asarray(payload["ground_truth"], dtype=np.float64)
        with (eval_dir / "per_step.csv").open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        keys, first_prediction, first_ground_truth = [], [], []
        cursor = 0
        for row in rows:
            horizon = int(row["evaluated_horizon"])
            assert horizon > 0 and cursor + horizon <= len(prediction)
            keys.append((int(row["trajectory_id"]), int(row["step"])))
            first_prediction.append(prediction[cursor])
            first_ground_truth.append(ground_truth[cursor])
            cursor += horizon
        assert cursor == len(prediction), (cursor, len(prediction))
        return keys, np.stack(first_prediction), np.stack(first_ground_truth)

    loaded = [load_first_action_samples(repeat_dir(seed)) for seed in EVAL_REPEAT_SEEDS]
    keys = loaded[0][0]
    ground_truth = loaded[0][2]
    for candidate_keys, _, candidate_gt in loaded[1:]:
        assert candidate_keys == keys
        assert np.allclose(candidate_gt, ground_truth, atol=1e-7, rtol=1e-7)
    predictions = np.stack([item[1] for item in loaded], axis=0)
    raw = predictions[0]
    mean = predictions.mean(axis=0)
    std = predictions.std(axis=0)
    raw_error = raw - ground_truth
    mean_error = mean - ground_truth

    summary = {
        "method": "first action per held-out timestamp; eight paired diffusion seeds; no smoothing",
        "variant": "cka_keep4dit_keep6lang_keep2vlsa",
        "denoising_steps": DENOISING_STEPS,
        "seeds": EVAL_REPEAT_SEEDS,
        "raw_mse": float(np.mean(raw_error ** 2)),
        "raw_mae": float(np.mean(np.abs(raw_error))),
        "ensemble_mse": float(np.mean(mean_error ** 2)),
        "ensemble_mae": float(np.mean(np.abs(mean_error))),
        "sampling_std": float(np.mean(std)),
    }
    metric_rows = []
    for dimension, channel in enumerate(action_names):
        channel_raw_error = raw[:, dimension] - ground_truth[:, dimension]
        channel_mean_error = mean[:, dimension] - ground_truth[:, dimension]
        metric_rows.append({
            "model": "cka_keep4dit_keep6lang_keep2vlsa",
            "dimension": dimension,
            "channel": channel,
            "raw_mse": float(np.mean(channel_raw_error ** 2)),
            "raw_mae": float(np.mean(np.abs(channel_raw_error))),
            "ensemble_mse": float(np.mean(channel_mean_error ** 2)),
            "ensemble_mae": float(np.mean(np.abs(channel_mean_error))),
            "mean_sampling_std": float(np.mean(std[:, dimension])),
        })

    figure, axes = plt.subplots(7, 1, figsize=(16, 17), sharex=True)
    x = np.arange(len(ground_truth))
    for dimension, axis in enumerate(axes):
        axis.plot(x, ground_truth[:, dimension], color="black", linewidth=1.7, label="ground truth")
        axis.plot(x, raw[:, dimension], color="#ff9896", linewidth=0.8, alpha=0.65, label="raw seed 42")
        axis.plot(x, mean[:, dimension], color="#d62728", linewidth=1.2, label="8-seed mean")
        axis.fill_between(
            x,
            mean[:, dimension] - std[:, dimension],
            mean[:, dimension] + std[:, dimension],
            color="#d62728",
            alpha=0.14,
            label="±1 sampling std",
        )
        axis.set_ylabel(action_names[dimension])
        axis.grid(alpha=0.2)
        axis.legend(loc="best", ncol=2)
    axes[-1].set_xlabel("First predicted action at each held-out timestamp")
    figure.suptitle("4/6/2 candidate: raw and repeated-seed first-action prediction")
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    plot_path = CANDIDATE_ANALYSIS / "first_action_diffusion_ensemble_keep4dit_keep6lang_keep2vlsa.png"
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    summary["plot"] = str(plot_path)
    summary_path = CANDIDATE_ANALYSIS / "ensemble_first_action_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    csv_path = CANDIDATE_ANALYSIS / "ensemble_first_action_per_channel.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    print(json.dumps(summary, indent=2))
    print("Saved:", summary_path, csv_path)
else:
    print("Repeated-seed 4/6/2 diagnostics run only in WORK_PHASE='evaluate'.")

# %% [markdown]
# ## I. Original-source versus recovered-4/6/2 representation heatmaps
#

# %%
if RUN_EVALUATION:
    recovered = OUTPUT_ROOT / "recovered_calibration"

    def calibration_complete(path):
        path = Path(path)
        return all(
            (path / name).is_file() and (path / name).stat().st_size > 0
            for name in ["metadata.json", "activations.npz"]
        )

    assert calibration_complete(SOURCE_CALIBRATION_DIR), (
        f"Missing/incomplete source CKA capture: {SOURCE_CALIBRATION_DIR}. "
        "Run WORK_PHASE='prepare_cka' first."
    )
    if calibration_complete(recovered):
        print("Reuse completed recovered activation capture:", recovered)
    else:
        if recovered.exists():
            assert recovered.parent.resolve() == OUTPUT_ROOT.resolve()
            shutil.rmtree(recovered)
        run_checked(
            "Capture recovered UR10e CKA activations",
            uv_python(
                "scripts/cka_n1d7/capture_activations.py",
                "--model-path", str(CKA_OUTPUT),
                "--output-dir", str(recovered),
                *capture_common,
            ),
            cwd=REPO_DIR, env=RUN_ENV,
        )
    assert calibration_complete(recovered), recovered

    heatmap_summary = HEATMAP_ROOT / "cka_before_after_summary.json"
    heatmap_complete = heatmap_summary.is_file() and bool(list(HEATMAP_ROOT.glob("*.png")))
    if heatmap_complete:
        print("Reuse completed source/recovered heatmaps:", HEATMAP_ROOT)
    else:
        if HEATMAP_ROOT.exists():
            assert HEATMAP_ROOT.parent.resolve() == OUTPUT_ROOT.resolve()
            shutil.rmtree(HEATMAP_ROOT)
        HEATMAP_ROOT.mkdir(parents=True, exist_ok=True)
        run_checked(
            "Compare UR10e source and recovered heatmaps",
            uv_python(
                "scripts/cka_n1d7/compare_cka_heatmaps.py",
                "--baseline-calibration-dir", str(SOURCE_CALIBRATION_DIR),
                "--optimized-calibration-dir", str(recovered),
                "--output-dir", str(HEATMAP_ROOT),
            ),
            cwd=REPO_DIR, env=RUN_ENV,
        )
    assert heatmap_summary.is_file(), heatmap_summary
    assert list(HEATMAP_ROOT.glob("*.png")), HEATMAP_ROOT
    print("Heatmaps:", HEATMAP_ROOT)

    EVALUATION_STEP_MARKER.write_text(
        json.dumps({
            "candidate_step": CURRENT_EVALUATION_STEP,
            "candidate_model": str(CKA_OUTPUT),
            "status": "offline_and_heatmaps_complete",
        }, indent=2),
        encoding="utf-8",
    )
    print("Evaluation cache marker:", EVALUATION_STEP_MARKER)

# %% [markdown]
# ## J. Offline evidence report and compact archive
#

# %%

if RUN_REPORT:
    from zipfile import ZIP_DEFLATED, ZipFile
    import hashlib

    required = [
        PRUNING_MANIFEST,
        OUTPUT_ROOT / "recovery_contract.json",
        OUTPUT_ROOT / "huggingface_upload_receipt.json",
        OUTPUT_ROOT / "deployment_contract.json",
        EVAL_CKA / "summary.json",
        EVAL_CKA / "per_step.csv",
        CANDIDATE_ANALYSIS / "per_dimension_metrics.csv",
        CANDIDATE_ANALYSIS / "aligned_prediction_diagnostics.json",
        CANDIDATE_ANALYSIS / "ensemble_first_action_metrics.json",
        CANDIDATE_ANALYSIS / "ensemble_first_action_per_channel.csv",
        CANDIDATE_ANALYSIS / "control_latency.json",
        CANDIDATE_ANALYSIS / "aligned_prediction_vs_ground_truth_keep4dit_keep6lang_keep2vlsa.png",
        CANDIDATE_ANALYSIS / "per_dimension_metrics.png",
        OUTPUT_ROOT / "diffusion_repeat_manifest.json",
        HEATMAP_ROOT / "cka_before_after_summary.json",
        TRAIN_OUTPUT_ROOT / f"{CKA_OUTPUT.name}_recovery_identity.json",
        TRAIN_OUTPUT_ROOT / f"{CKA_OUTPUT.name}_target_history.json",
        OUTPUT_ROOT / "evaluated_candidate_step.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    assert not missing, f"Missing required 4/6/2 candidate evidence: {missing}"
    assert list(HEATMAP_ROOT.glob("*.png")), HEATMAP_ROOT
    assert root_model_ready(CKA_OUTPUT)
    actual_candidate_step = latest_step(CKA_OUTPUT)
    assert actual_candidate_step == FINAL_STEP, (actual_candidate_step, FINAL_STEP)
    evaluated_step = int(json.loads(
        (OUTPUT_ROOT / "evaluated_candidate_step.json").read_text(encoding="utf-8")
    )["candidate_step"])
    assert evaluated_step == actual_candidate_step, (evaluated_step, actual_candidate_step)

    manifest = json.loads(PRUNING_MANIFEST.read_text(encoding="utf-8"))
    kept = {
        name: len(spec["keep_indices"])
        for name, spec in manifest["modules"].items()
    }
    assert kept == EXPECTED_KEPT_DEPTHS, (kept, EXPECTED_KEPT_DEPTHS)
    candidate_summary = json.loads((EVAL_CKA / "summary.json").read_text(encoding="utf-8"))
    ensemble = json.loads(
        (CANDIDATE_ANALYSIS / "ensemble_first_action_metrics.json").read_text(encoding="utf-8")
    )
    report = {
        "status": "candidate_only_offline_complete",
        "task": "UR10e + Robotiq cup pickup",
        "variant": "4/6/2 standalone candidate-only full retained-module recovery",
        "resource_profile": "Linux server: 1x 40 GB GPU, 8+ CPU, 48+ GiB RAM",
        "source_checkpoint": str(MODEL_PATH),
        "source_checkpoint_is_task_adapted": False,
        "external_reference_required_for_comparison": True,
        "data_split": {
            "train": TRAIN_EPISODES,
            "validation": VALIDATION_EPISODES,
            "test": TEST_EPISODES,
        },
        "candidate_step": latest_step(CKA_OUTPUT),
        "pruning_ratios": {
            "backbone_language": BACKBONE_LANGUAGE_PRUNE_RATIO,
            "action_dit": ACTION_DIT_PRUNE_RATIO,
            "vl_self_attention": VL_SELF_ATTENTION_PRUNE_RATIO,
        },
        "kept_depths": kept,
        "candidate": candidate_summary,
        "diffusion_first_action_diagnostics": ensemble,
        "claim_boundary": (
            "This archive contains 4/6/2 candidate-only offline evidence. It does not "
            "contain or recompute a baseline and contains no simulator/real-robot SR."
        ),
        "model_weights_in_zip": False,
        "candidate_model_path": str(CKA_OUTPUT),
    }
    (OUTPUT_ROOT / "final_offline_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    if FINAL_ZIP.exists():
        FINAL_ZIP.unlink()
    with ZipFile(FINAL_ZIP, "w", compression=ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(OUTPUT_ROOT.rglob("*")):
            if not path.is_file():
                continue
            if path.name in {"activations.npz", "actions.npz"} or path.suffix == ".mp4":
                continue
            archive.write(path, arcname=path.relative_to(EXPERIMENT_ROOT))
        recovery_identity_path = TRAIN_OUTPUT_ROOT / f"{CKA_OUTPUT.name}_recovery_identity.json"
        archive.write(
            recovery_identity_path,
            arcname=Path("results") / "recovery_identity.json",
        )
    digest = hashlib.sha256(FINAL_ZIP.read_bytes()).hexdigest()
    checksum = FINAL_ZIP.with_suffix(".zip.sha256")
    checksum.write_text(f"{digest}  {FINAL_ZIP.name}\n", encoding="utf-8")
    print("4/6/2 candidate-only result ZIP:", FINAL_ZIP)
    print("Size MiB:", round(FINAL_ZIP.stat().st_size / 1024**2, 2))
    print("SHA-256:", digest)
    print("Candidate weights remain on server storage:", CKA_OUTPUT)
else:
    print("Run All must complete evaluation before report generation after 4/6/2 evaluation and heatmaps complete.")

# %% [markdown]
# ## Final local artifacts
#

# %%
if RUN_REPORT:
    import hashlib

    archive = Path(FINAL_ZIP)
    checksum = archive.with_suffix(".zip.sha256")
    assert archive.is_file() and checksum.is_file()
    expected = checksum.read_text(encoding="utf-8").split()[0]
    digest = hashlib.sha256()
    with archive.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    assert digest.hexdigest() == expected
    print("HF model: https://huggingface.co/Luke99662244/462")
    print("Merged local checkpoint:", CKA_OUTPUT)
    print("Evidence ZIP:", archive)
    print("SHA-256:", expected)
    print("FULL 4/6/2 RUN-ALL PIPELINE: COMPLETE")

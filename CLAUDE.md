# CLAUDE.md — Isaac GR00T N1.7

## Project overview

Isaac GR00T N1.7 is an open vision-language-action (VLA) model for generalized humanoid robot skills.
The repo contains the model, training pipeline, evaluation harness, and deployment tooling.

- **Language:** Python 3.12 (dGPU, Thor, DGX Spark); Python 3.10 (Orin — see deployment dir)
- **Package manager:** [uv](https://docs.astral.sh/uv/)
- **Build system:** setuptools (see `pyproject.toml`)

## Quick-start commands

```bash
# Install (dev mode with all extras)
uv sync --all-extras

# Lint and format (uses ruff via pre-commit)
pre-commit run --all-files

# Run CPU tests
python -m pytest tests/ -m "not gpu" -v --timeout=300

# Run GPU tests
python -m pytest tests/ -m gpu -v --timeout=300

# Build package
uv build

# Validate lockfile
uv lock --locked
```

## Code style

- Formatter: `ruff format` (double quotes, spaces, line-length 100)
- Linter: `ruff check` with rules E, F, I (ignores E501)
- Config lives in `pyproject.toml` under `[tool.ruff]`
- Run `pre-commit run --all-files` before committing

## Directory layout

```
gr00t/              # Main package
  configs/          #   Training, data, and model configs
  data/             #   Data loading, embodiment tags, dataset processing
  deployment/       #   Shared deployment mode enums (modes.py)
  efficient/        #   Efficient inference benchmark (pruning, scheduling — see below)
  eval/             #   Evaluation (run_gr00t_server.py)
  experiment/       #   Training pipeline (launch_finetune.py, trainer.py)
  model/            #   Model architecture (N1.7, base, modules)
  policy/           #   Policy inference (Gr00tPolicy, server/client)
  utils/            #   Determinism, video, and initial-action helpers
examples/           # Per-embodiment example configs and READMEs
scripts/            # Deployment, conversion, and utility scripts
  deployment/       #   Platform install scripts (dgpu, orin, thor, spark)
tests/              # pytest suite (markers: gpu, not gpu)
results/            # Benchmark result JSON/CSV outputs (efficient_benchmark/)
getting_started/    # User-facing guides and notebooks
```

`AGENTS.md` at the repo root is a symlink to this file — edits here apply to both.

## Key entry points

- **Fine-tune:** `bash examples/finetune.sh --base-model-path <path> --dataset-path <path> --embodiment-tag <tag> --output-dir <dir>`
- **Inference server:** `python gr00t/eval/run_gr00t_server.py --model-path <path> --embodiment-tag <tag>`
- **ONNX export:** `python scripts/deployment/export_onnx_n1d7.py`
- **TensorRT build:** `python scripts/deployment/build_trt_pipeline.py`
- **Benchmark:** `python scripts/deployment/benchmark_inference.py`

## Efficient inference benchmark (`gr00t/efficient/`)

Self-contained framework for measuring inference-efficiency methods (visual token pruning, dynamic scheduling) on GR00T N1.7 against LIBERO. **It must not change model, training, policy, or deployment behavior outside `gr00t/efficient/`** — pruning is applied via an opt-in `VisualTokenHook` attached to the loaded model's backbone output, never by editing model code.

Architecture:

- `benchmark/registry.py` — method registry; methods: `baseline`, `dummy`, `vlapruner`, `specprune`, `adp`, `adp_vlapruner`. New methods are classes in `benchmark/methods.py` registered in `_register_default_methods()`.
- `pruners/` — `VisualTokenPruner` base class (`keep_ratio` in (0, 1], safe no-op default) and implementations (dummy, VLA-Pruner, SpecPrune)
- `schedulers/` — dynamic scheduling (ADP)
- `hooks/visual_token_hook.py` — opt-in hook that applies a method to `[B, N, D]` visual tokens
- `benchmark/run_libero.py` / `run_libero_plus.py` — main CLIs with three modes: `--dry-run` (no model), `--mock` (synthetic tokens, no LIBERO), and real (requires `--model-path`, uses `real_libero_adapter.py`)
- `benchmark/run_comparison.py` → `collect_results.py` → `generate_report.py` — multi-method comparison pipeline
- `profiler/` — latency, memory, and token-count helpers
- Results go under `results/efficient_benchmark/`; configs under `gr00t/efficient/configs/`

Typical mock run:

```bash
python -m gr00t.efficient.benchmark.run_libero \
  --mock --method vlapruner --keep-ratio 0.75 --score-mode norm \
  --visual-token-count 256 --num-episodes 3 --task debug \
  --output results/efficient_benchmark/<name>.json
```

Full command catalog (dry-run/mock/real, per-method flags) lives in `gr00t/efficient/docs/README.md`; per-area ownership boundaries in `gr00t/efficient/docs/AGENTS.md`. LIBERO-Plus is eval-only — never used for training or fine-tuning.

## Testing

- Test markers: `gpu` (requires GPU), default is CPU-safe
- Run a single test: `python -m pytest tests/<path>/test_<file>.py::<test_name> -v`
- Fixtures live in `tests/fixtures/` and `demo_data/`
- CI runs CPU and GPU tests in separate jobs with 300s timeout

## Deployment platforms

- **dGPU (H100, A100, RTX):** CUDA 12.8 — install via `scripts/deployment/dgpu/install_deps.sh`, container via top-level `docker/Dockerfile` (supports x86_64 and aarch64)
- **Jetson Orin:** CUDA 12.6 — install via `scripts/deployment/orin/install_deps.sh`, container via `scripts/deployment/orin/Dockerfile`
- **Jetson Thor:** CUDA 13.0 — install via `scripts/deployment/thor/install_deps.sh`, container via `scripts/deployment/thor/Dockerfile`
- **DGX Spark:** CUDA 13.0 — install via `scripts/deployment/spark/install_deps.sh`, container via `scripts/deployment/spark/Dockerfile`

Each Jetson/Spark platform ships an `activate_*.sh` helper (`scripts/activate_orin.sh`, `scripts/activate_spark.sh`, `scripts/activate_thor.sh`) that exports platform-specific library paths. For dGPU, the standard `source .venv/bin/activate` is sufficient.

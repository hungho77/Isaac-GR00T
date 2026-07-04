# Efficient Benchmark Agents

## Repo Setup Agent
- Responsibility: Maintain package scaffolding, branch hygiene, config placement, and result directories.
- Owns: `gr00t/efficient/__init__.py`, package `__init__.py` files, `gr00t/efficient/configs/`, `results/efficient_benchmark/`.
- Must not change: Existing GR00T model, training, policy, deployment, or evaluation behavior.

## Benchmark Agent
- Responsibility: Add benchmark entry points, dry-run behavior, result schemas, and future LIBERO wiring.
- Owns: `gr00t/efficient/benchmark/`.
- Must not change: LIBERO dependencies, GR00T inference internals, or environment setup scripts without explicit approval.

## Profiler Agent
- Responsibility: Provide lightweight latency, memory, and token-count helpers.
- Owns: `gr00t/efficient/profiler/`.
- Must not change: CUDA runtime configuration, model precision, or TensorRT export logic.

## Pruner Agent
- Responsibility: Define pruning interfaces and safe test pruners.
- Owns: `gr00t/efficient/pruners/`.
- Must not change: GR00T visual encoders, action heads, checkpoint loading, or production inference paths.

## Evaluation Agent
- Responsibility: Define benchmark metrics, compare methods to baselines, and record failure types.
- Owns: `gr00t/efficient/benchmark/metrics.py` and future evaluation summaries under `results/efficient_benchmark/`.
- Must not change: Existing test fixtures, eval wrappers, or simulation integrations outside `gr00t/efficient/`.

## Report Agent
- Responsibility: Maintain benchmark documentation, target milestones, and comparison reports.
- Owns: `gr00t/efficient/docs/`.
- Must not change: Public docs outside the efficient benchmark area unless requested.


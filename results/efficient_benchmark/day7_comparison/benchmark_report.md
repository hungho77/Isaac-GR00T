# GR00T N1.7 Efficient Inference Benchmark — Day 7 Report

## Scope
- Mock LIBERO comparison.
- Methods tested: `baseline`, `dummy`, `vlapruner`, `specprune`, `adp`, `adp_vlapruner`.
- Metrics: success rate, latency, episode time, GPU memory, visual tokens, keep ratio, token reduction, and action L2.

## Methods
- `baseline`
- `dummy`
- `vlapruner`
- `specprune`
- `adp`
- `adp_vlapruner`

## Summary Table
| Method | Records | Success | Latency/action ms | Tokens After | Keep Ratio | Token Reduction | Action L2 | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| adp | 5 | 1.000 | 46.000 | 164.000 | 0.640 | 0.359 | 0.000 | 0.978x |
| adp_vlapruner | 5 | 1.000 | 46.000 | 164.000 | 0.640 | 0.359 | 0.000 | 0.978x |
| baseline | 5 | 1.000 | 45.000 | 256.000 | 1.000 | 0.000 | 0.000 | 1.000x |
| dummy | 15 | 1.000 | 46.000 | 158.000 | 0.617 | 0.383 | 0.000 | 0.978x |
| specprune | 15 | 1.000 | 46.000 | 158.000 | 0.617 | 0.383 | 0.000 | 0.978x |
| vlapruner | 15 | 1.000 | 46.000 | 158.000 | 0.617 | 0.383 | 0.000 | 0.978x |

## Key Observations
- Fastest method in mock: `baseline` at 45.000 ms/action.
- Lowest action L2 in mock: `adp` at 0.000.
- Best token reduction in mock: `dummy` at 0.383.
- Mock speedup >= 1.2x: no; best was `baseline` at 1.000x.
- Action L2 remains low in mock: yes; max was 0.000.

## Blockers
- Real LIBERO is not connected yet unless discovered otherwise.
- Real GR00T visual token/action-state hooks are not connected yet.
- Mock metrics are for pipeline validation only.

## Recommendation
- Next: connect real LIBERO baseline.
- Then: connect real visual token hook.
- Then: evaluate VLA-Pruner on real episodes.
- Then: run LIBERO-Plus robustness benchmark.

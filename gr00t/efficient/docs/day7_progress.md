# Day 7 — Full Comparison + Report

## Goal
Run full mock comparison and generate report.

## Completed
- [x] Added run_comparison.py
- [x] Improved collect_results.py
- [x] Added generate_report.py
- [x] Ran full mock comparison
- [x] Generated aggregated records
- [x] Generated method summary
- [x] Generated benchmark report
- [x] Added final recommendation doc
- [x] Updated README

## Methods

| Method | Status | Notes |
|---|---|---|
| baseline | complete | no pruning |
| dummy | complete | first-mode token pruning at keep ratios 0.75, 0.6, 0.5 |
| vlapruner | complete | MVP norm scoring at keep ratios 0.75, 0.6, 0.5 |
| specprune | complete | MVP mask reuse at keep ratios 0.75, 0.6, 0.5 |
| adp | complete | dynamic keep-ratio scheduler |
| adp_vlapruner | complete | ADP scheduler + VLA-Pruner selector |

## Metrics

| Method | Speedup | Token Reduction | Action L2 | Notes |
|---|---:|---:|---:|---|
| baseline | 1.000x | 0.000 | 0.000 | 5 mock records |
| dummy | 0.978x | 0.383 | 0.000 | averaged across keep ratios 0.75, 0.6, 0.5 |
| vlapruner | 0.978x | 0.383 | 0.000 | averaged across keep ratios 0.75, 0.6, 0.5 |
| specprune | 0.978x | 0.383 | 0.000 | averaged across keep ratios 0.75, 0.6, 0.5 |
| adp | 0.978x | 0.359 | 0.000 | dynamic average keep ratio 0.64 |
| adp_vlapruner | 0.978x | 0.359 | 0.000 | dynamic average keep ratio 0.64 |

## Blockers
- Real LIBERO execution not connected yet.
- Real GR00T visual token hook not connected yet.
- Real action_state/robot_state hook not connected yet.

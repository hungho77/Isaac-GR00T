# GR00T-N1.7 LIBERO Object Reference Comparison

This report compares compact reference ZIPs for GR00T-N1.7 CKA pruning experiments.

## Configuration

- COMPARISON: object
- Suite ID: libero_object
- USE_FREEZE: False
- Source: Kaggle private dataset mounted at /kaggle/input/datasets/nguyentrangntkt/object
- No training, offline inference, or rollout is rerun in this notebook.

## Main Comparison

| Variant   | Params   | Param reduction   | Mean latency ms   | Speedup   | VRAM GiB   | MSE      | MAE      | SR     |   Successes |   Episodes |
|:----------|:---------|:------------------|:------------------|:----------|:-----------|:---------|:---------|:-------|------------:|-----------:|
| Baseline  | 3.144B   | 0.00%             | N/A               | N/A       | N/A        | N/A      | N/A      | 99.00% |         198 |        200 |
| P25       | 2.675B   | 14.91%            | 169.310           | N/A       | 5.063      | 0.001262 | 0.013641 | 99.50% |         199 |        200 |
| P37       | 2.439B   | 22.42%            | 151.862           | N/A       | 4.620      | 0.002756 | 0.022598 | 99.50% |         199 |        200 |
| P50       | 2.205B   | 29.88%            | 134.896           | N/A       | 4.179      | 0.009182 | 0.047172 | 99.00% |         198 |        200 |
| P75       | 1.731B   | 44.94%            | 104.170           | N/A       | 3.290      | 0.013892 | 0.060439 | 96.00% |         192 |        200 |

## Quick Takeaways

- Fastest latency: **P75** at 104.170 ms.
- Highest success rate: **P25** at 99.50%.
- Best latency-preserving-SR tradeoff in this table: **P50**.

Small success-rate differences such as 1-2 failures over 200 episodes should be treated carefully. They are useful signals, but not enough alone to claim one close variant is definitively more accurate than another.

## Output Files

- comparison.json
- tables/variant_summary.csv
- tables/task_success_rates.csv
- tables/task_success_rate_pivot.csv
- tables/normalized_vs_baseline.csv
- plots/normalized_comparison.png
- plots/latency_success_tradeoff.png
- plots/params_latency_tradeoff.png
- plots/offline_mse_mae.png
- plots/success_rate_bar.png
- plots/per_task_success_rate.png
- plots/kept_layers.png

## Limitation

The compact reference ZIPs preserve metrics and CSV summaries only. They do not include model weights, raw rollout videos, or full heatmap PNGs. For visual heatmap figures, use the original full result archives.

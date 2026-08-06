# GR00T-N1.7 LIBERO Long / LIBERO-10 Reference Comparison

This report compares compact reference ZIPs for GR00T-N1.7 CKA pruning experiments.

## Configuration

- COMPARISON: long
- Suite ID: libero_10
- USE_FREEZE: False
- Source: Kaggle private dataset mounted at /kaggle/input/datasets/nguyentrangntkt/longgg
- No training, offline inference, or rollout is rerun in this notebook.

## Main Comparison

| Variant   | Params   | Param reduction   |   Mean latency ms | Speedup   |   VRAM GiB |      MSE |      MAE | SR     |   Successes |   Episodes |
|:----------|:---------|:------------------|------------------:|:----------|-----------:|---------:|---------:|:-------|------------:|-----------:|
| Baseline  | 3.144B   | 0.00%             |           268.458 | 1.000x    |      5.952 | 0.001738 | 0.015405 | 98.00% |         196 |        200 |
| P25       | 2.675B   | 14.91%            |           117.54  | 2.284x    |      5.065 | 0.002292 | 0.019334 | 92.00% |         184 |        200 |
| P37       | 2.439B   | 22.42%            |           159.964 | 1.678x    |      4.623 | 0.003838 | 0.02735  | 85.50% |         171 |        200 |
| P50       | 2.205B   | 29.88%            |           130.239 | 2.061x    |      4.182 | 0.010837 | 0.053356 | 63.50% |         127 |        200 |

## Quick Takeaways

- Fastest latency: **P25** at 117.540 ms.
- Highest success rate: **Baseline** at 98.00%.
- Best latency-preserving-SR tradeoff in this table: **Baseline**.

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

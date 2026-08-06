# GR00T-N1.7 LIBERO Goal Reference Comparison

This report compares compact reference ZIPs for GR00T-N1.7 CKA pruning experiments.

## Configuration

- COMPARISON: longgg
- Suite ID: libero_goal
- USE_FREEZE: False
- Source: Kaggle private dataset mounted at /kaggle/input/datasets/nguyentrangntkt/goallll
- No training, offline inference, or rollout is rerun in this notebook.

## Main Comparison

| Variant   | Params   | Param reduction   |   Mean latency ms | Speedup   |   VRAM GiB |      MSE |      MAE | SR     |   Successes |   Episodes |
|:----------|:---------|:------------------|------------------:|:----------|-----------:|---------:|---------:|:-------|------------:|-----------:|
| Baseline  | 3.144B   | 0.00%             |           192.578 | 1.000x    |      5.947 | 0.000644 | 0.011211 | 99.50% |         199 |        200 |
| P25       | 2.677B   | 14.86%            |           166.656 | 1.156x    |      5.064 | 0.003535 | 0.025609 | 97.50% |         195 |        200 |
| P37       | 2.442B   | 22.32%            |           169.731 | 1.135x    |      4.625 | 0.005524 | 0.034058 | 94.00% |         188 |        200 |
| P50       | 2.206B   | 29.83%            |           128.384 | 1.500x    |      4.18  | 0.008925 | 0.048202 | 83.00% |         166 |        200 |

## Quick Takeaways

- Fastest latency: **P50** at 128.384 ms.
- Highest success rate: **Baseline** at 99.50%.
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

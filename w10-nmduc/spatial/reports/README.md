# GR00T-N1.7 LIBERO Spatial Reference Comparison

This report compares compact reference ZIPs for GR00T-N1.7 CKA pruning experiments.

## Configuration

- COMPARISON: spatial
- Suite ID: libero_spatial
- USE_FREEZE: False
- Source: Kaggle private dataset mounted at /kaggle/input/datasets/nguyentrangntkt/spatial
- No training, offline inference, or rollout is rerun in this notebook.

## Main Comparison

| Variant   | Params   | Param reduction   |   Mean latency ms | Speedup   |   VRAM GiB |      MSE |      MAE | SR     |   Successes |   Episodes |
|:----------|:---------|:------------------|------------------:|:----------|-----------:|---------:|---------:|:-------|------------:|-----------:|
| Baseline  | 3.144B   | 0.00%             |           197.594 | 1.000x    |      5.952 | 0.000479 | 0.010883 | 99.00% |         198 |        200 |
| P25       | 2.675B   | 14.91%            |           168.298 | 1.174x    |      5.065 | 0.00299  | 0.02673  | 99.50% |         199 |        200 |
| P37       | 2.441B   | 22.37%            |           149.644 | 1.320x    |      4.627 | 0.007807 | 0.043522 | 95.50% |         191 |        200 |
| P50       | 2.205B   | 29.88%            |           127.056 | 1.555x    |      4.179 | 0.011685 | 0.05353  | 87.00% |         174 |        200 |

## Quick Takeaways

- Fastest latency: **P50** at 127.056 ms.
- Highest success rate: **P25** at 99.50%.
- Best latency-preserving-SR tradeoff in this table: **P25**.

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

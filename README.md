# GR00T N1.7 — CKA-Guided Structural Pruning

Fork của [NVIDIA Isaac GR00T N1.7](https://developer.nvidia.com/isaac/gr00t), áp dụng
phương pháp CLP (CKA-guided layer pruning) — vốn thiết kế cho GR00T-N1.5 — sang kiến
trúc N1.7 (backbone Qwen3-VL/Cosmos-Reason2-2B + action head `AlternateVLDiT`).

**Kết quả chính: cắt xuống 43.2% tham số và giảm 62% latency mà không mất accuracy —
200/200 rollout trên full LIBERO Object suite, ngang model gốc.**

## Tóm tắt

| Checkpoint | DiT | Backbone | vlsa | Success | Params | % params | Latency¹ | Control freq |
|---|---|---|---|---|---|---|---|---|
| base (NVIDIA gốc) | 32 | 16 | 4 | 200/200 | 3,455,180,928 | 100.0% | 64.24 ms | 125 Hz |
| anchor | 24 | 16 | 4 | 200/200 | 2,873,348,224 | 83.2% | 55.41 ms | 144 Hz |
| v6 | 12 | 10 | 4 | 200/200 | 2,165,330,560 | 62.7% | 40.76 ms | 196 Hz |
| v7 | 8 | 7 | 3 | 200/200 | 1,828,630,400 | 52.9% | 30.95 ms | 258 Hz |
| v8 | 4 | 7 | 3 | 200/200 | 1,693,296,512 | 49.0% | 26.54 ms | 301 Hz |
| v9 | 4 | 6 | 3 | 200/200 | 1,642,960,512 | 47.6% | 26.48 ms | 302 Hz |
| **v10** | **4** | **4** | **2** | **200/200** | **1,491,930,240** | **43.2%** | **24.32 ms** | **329 Hz** |

¹ Tổng latency đã chuẩn hoá thành phần tiền xử lý CPU, đo trên RTX 4090. Xem
[PRUNING_RESULTS.md](PRUNING_RESULTS.md) để có breakdown theo thành phần, cột GPU-side,
và các cảnh báo về độ tin cậy của phép đo.

Không cấu hình nào trong 7 checkpoint cho thấy suy giảm accuracy — kể cả v10 chỉ còn
4/32 DiT block (12.5% độ sâu gốc), 4/16 backbone layer, 2/4 vlsa layer.

## Tài liệu

| File | Nội dung |
|---|---|
| [PRUNING_RESULTS.md](PRUNING_RESULTS.md) | Kết quả đầy đủ: success rate, tham số theo component, latency chi tiết, giới hạn lý thuyết của phương pháp |
| [PRUNING_REPORT.md](PRUNING_REPORT.md) | Thay đổi code so với bản gốc NVIDIA + phân tích các bug đã gặp và cách sửa |

## Sử dụng

### Train một cấu hình pruned

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True uv run python gr00t/experiment/launch_finetune.py \
    --base_model_path checkpoints/GR00T-N1.7-LIBERO/libero_object \
    --dataset_path examples/LIBERO/converted/IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot/ \
    --embodiment_tag LIBERO_PANDA \
    --num_gpus 1 \
    --output_dir <output_dir> \
    --save_steps 1000 --save_total_limit 5 --max_steps 8000 \
    --warmup_ratio 0.05 --weight_decay 1e-5 --learning_rate 1e-4 \
    --global_batch_size 2 --gradient_accumulation_steps 16 \
    --dataloader_num_workers 4 --shard_size 1024 --num_shards_per_epoch 100000 \
    --episode_sampling_rate 0.1 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --prune_model \
    --kept_layer_idx_list_backbone 0,1,9,15 \
    --kept_layer_idx_list_dit 0,1,2,3 \
    --kept_layer_idx_list_vl_self_attn 0,2
```

Thêm `--lora_rank <r>` để dùng LoRA thay vì full-finetune (khi đó cần chạy
`scripts/merge_lora_checkpoint.py` trước khi eval). Không có `--lora_rank` = full
finetune; không có `--tune_llm` = backbone đóng băng.

### Eval

```bash
# Server (main venv)
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path <checkpoint_dir> --embodiment-tag LIBERO_PANDA --use-sim-policy-wrapper

# Client (LIBERO venv) -- full 10-task suite
GR00T_GRIPPER_SIGNED=1 bash gr00t/eval/sim/LIBERO/run_object_suite.sh --log-dir <log_dir>
```

> **Quan trọng:** `GR00T_GRIPPER_SIGNED=1` là **bắt buộc** với mọi checkpoint finetune
> trên dataset LIBERO đã convert (LeRobot v3→v2), và **phải bỏ đi** với checkpoint gốc
> NVIDIA. Hai bên dùng convention gripper ngược dấu nhau; đặt sai cờ khiến gripper mở ra
> đúng lúc cần kẹp lại và success rate về 0%. Chi tiết ở
> [PRUNING_REPORT.md](PRUNING_REPORT.md) mục 1.4.

### Đo tham số và latency

```bash
uv run python scripts/count_checkpoint_params.py <checkpoint_dir> [<checkpoint_dir> ...]

uv run python scripts/deployment/benchmark_inference.py \
    --model-path <checkpoint_dir> --dataset-path demo_data/libero_demo \
    --embodiment-tag libero_sim --num-iterations 50 --warmup 10 --skip-compile
```

## Tài liệu gốc NVIDIA

[FAQ.md](FAQ.md) · [CONTRIBUTING.md](CONTRIBUTING.md) · [ATTRIBUTIONS.md](ATTRIBUTIONS.md)

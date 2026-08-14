# GR00T N1.7 — CKA-Guided Structural Pruning

Fork của [NVIDIA Isaac GR00T N1.7](https://developer.nvidia.com/isaac/gr00t), áp dụng
phương pháp CLP (CKA-guided layer pruning) — vốn thiết kế cho GR00T-N1.5 — sang kiến
trúc N1.7 (backbone Qwen3-VL/Cosmos-Reason2-2B + action head `AlternateVLDiT`).

**Kết quả chính: cắt xuống 43.2% tham số và giảm ~62% latency. Trên Object, Spatial và
Goal, accuracy giữ nguyên (100% / 100% / 99.83%). Trên LIBERO-10 (Long) — suite
long-horizon — accuracy giảm 4 điểm (95.00% → 91.00%).**

## Tóm tắt theo suite (cấu hình v10: DiT=4, backbone=4, vlsa=2 — 43.2% tham số)

| Suite | Base | v10cfg | Δ |
|---|---:|---:|---:|
| Object | 100.00% | 100.00% | 0 |
| Spatial | 100.00% | 100.00% | 0 |
| Goal | 100.00% | 99.83% | −0.17 |
| **10 (Long)** | **95.00%** | **91.00%** | **−4.00** |

Trên LIBERO-10, tăng DiT lên 12 (v12cfg, 52.5% tham số) lấy lại phần lớn: **93.50%**.

## Tóm tắt theo checkpoint (LIBERO Object)

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

Trên Object, không cấu hình nào trong 7 checkpoint cho thấy suy giảm — kể cả v10 chỉ còn
4/32 DiT block (12.5% độ sâu gốc), 4/16 backbone layer, 2/4 vlsa layer. Điểm suy giảm
đầu tiên chỉ xuất hiện trên LIBERO-10; bảng đầy đủ cho Long/Goal/Spatial ở
[PRUNING_RESULTS.md](PRUNING_RESULTS.md).

**Lưu ý khi eval:** checkpoint finetune trên dataset Object (đã qua
`convert_v3_to_v2.py`) cần `GR00T_GRIPPER_SIGNED=1`; Spatial/Goal/Long tải thẳng ở định
dạng LeRobot v2.1 nên **không** dùng cờ này.

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

### UR10e

Cấu hình v10 áp lên dataset UR10e + Robotiq (81 episode,
7 kênh: 6 khớp + gripper nhị phân). Config đã có sẵn: `examples/UR10e/modality.json`
và `examples/UR10e/ur10e_config.py`.

#### Checkpoint UR10e

| Checkpoint | DiT | Backbone | vlsa | Params | % params | Open-loop MSE | Open-loop MAE | HF |
|---|---|---|---|---|---|---|---|---|
| `ur10e_v10` | 4 | 4 | 2 | 1,491,930,240 | 43.2% | 0.000940 | 0.003603 | [Yin142/ur10e_cup_v10](https://huggingface.co/Yin142/ur10e_cup_v10) |
| `ur10e_v12` | 12 | 4 | 3 | 1,812,956,288 | 52.5% | 0.000773 | 0.003527 | [Yin142/ur10e_v12](https://huggingface.co/Yin142/ur10e_v12) |
| `ur10e_v13` | 4 | 6 | 2 | 1,592,602,240 | 46.1% | 0.000868 | 0.003449 | [Yin142/ur10e_v13](https://huggingface.co/Yin142/ur10e_v13) |

% params tính trên base NVIDIA gốc (3,455,180,928, xem bảng LIBERO Object ở trên).
MSE/MAE đo bằng `gr00t/eval/open_loop_eval.py`, trung bình trên các trajectory
held-out (66-80), so action dự đoán với action ground-truth trong dataset.

Layer giữ lại (chỉ số 0-based, đọc từ `config.json` mỗi checkpoint):

| Checkpoint | `kept_layer_idx_list_dit` | `kept_layer_idx_list_backbone` | `kept_layer_idx_list_vl_self_attn` |
|---|---|---|---|
| `ur10e_v10` | 0, 1, 2, 3 | 0, 1, 9, 15 | 0, 2 |
| `ur10e_v12` | 0, 1, 2, 3, 4, 5, 6, 7, 28, 29, 30, 31 | 0, 1, 9, 15 | 0, 1, 2 |
| `ur10e_v13` | 0, 1, 2, 3 | 0, 2, 7, 10, 14, 15 | 0, 2 |

Base gốc: DiT có 32 layer, backbone có 16 layer, vlsa có 4 layer

```bash
# 1. Tach held-out (15/81 episode) -- repo khong ho tro validation split
uv run python scripts/split_lerobot_dataset.py \
    --dataset-path <dataset_dir> --output-dir <train_dir> --holdout 15

# 2. Train tu base tong quat (KHONG dung checkpoint LIBERO)
NUM_GPUS=1 MAX_STEPS=10000 GLOBAL_BATCH_SIZE=32 USE_WANDB=0 \
uv run bash examples/finetune.sh \
    --base-model-path checkpoints/GR00T-N1.7-3B \
    --dataset-path <train_dir> \
    --modality-config-path examples/UR10e/ur10e_config.py \
    --embodiment-tag NEW_EMBODIMENT \
    --output-dir <output_dir> \
    -- --prune_model \
       --kept_layer_idx_list_backbone 0,1,9,15 \
       --kept_layer_idx_list_dit 0,1,2,3 \
       --kept_layer_idx_list_vl_self_attn 0,2

# 3. Eval open-loop tren dataset GOC, chi episode held-out
uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path <dataset_dir> --embodiment-tag NEW_EMBODIMENT \
    --model-path <output_dir> \
    --traj-ids 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 \
    --execution-horizon 16 --steps 400 --save-plot-path <plot_dir>
```

### Host model lên server

```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path <checkpoint_dir> \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/UR10e/ur10e_config.py \
    --host 0.0.0.0 --port 5555
```

`--modality-config-path` là **bắt buộc** với `NEW_EMBODIMENT`: repo không có config
dựng sẵn cho embodiment này, thiếu cờ thì server không biết bố cục 7 kênh / 2 camera.
Không dùng `--use-sim-policy-wrapper` (cờ đó chỉ để nối vào sim LIBERO).

Client gửi observation khớp `modality.json`, nhận về action chunk 16 bước:

```python
obs = {
    "video.side":  np.uint8 (T, 480, 640, 3),
    "video.wrist": np.uint8 (T, 480, 640, 3),
    "state.single_arm": np.float32 (T, 6),   # goc khop, radian
    "state.gripper":    np.float32 (T, 1),   # 0/1
    "annotation.human.task_description": ["pick up the cup"],
}
# -> action.single_arm (16, 6), action.gripper (16, 1)
```

> `single_arm` được cấu hình `RELATIVE`: giá trị trả về là **delta so với state hiện
> tại**, phải cộng vào góc khớp hiện tại trước khi gửi xuống controller. `gripper` là
> `ABSOLUTE`, dùng thẳng. Mẫu client tham khảo: `gr00t/eval/real_robot/SO100`.

### Đo tham số và latency

```bash
uv run python scripts/count_checkpoint_params.py <checkpoint_dir> [<checkpoint_dir> ...]

uv run python scripts/deployment/benchmark_inference.py \
    --model-path <checkpoint_dir> --dataset-path demo_data/libero_demo \
    --embodiment-tag libero_sim --num-iterations 50 --warmup 10 --skip-compile
```

## Tài liệu gốc NVIDIA

[FAQ.md](FAQ.md) · [CONTRIBUTING.md](CONTRIBUTING.md) · [ATTRIBUTIONS.md](ATTRIBUTIONS.md)

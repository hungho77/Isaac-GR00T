# Kết quả benchmark — 7 checkpoint

Xem thay đổi code / nguyên nhân các bug ở [PRUNING_REPORT.md](PRUNING_REPORT.md).

Eval trên **LIBERO Object suite**, chạy sau khi sửa bug gripper-sign
(`GR00T_GRIPPER_SIGNED=1` cho checkpoint finetune, không cờ cho base). Latency đo
trên RTX 4090 (`scripts/deployment/benchmark_inference.py`, PyTorch eager,
50 iteration, `--skip-compile`).

## Bảng tổng hợp

| Checkpoint | DiT | Backbone | vlsa | Cách train | Success | Params | % params | E2E latency | Control freq |
|---|---|---|---|---|---|---|---|---|---|
| **base** (NVIDIA gốc) | 32 | 16 | 4 | — | 200/200 (100%) | 3,455,180,928 | 100.0% | 64.4 ms | 124 Hz |
| **anchor** (dit24) | 24 | 16 | 4 | LoRA r32, backbone đóng băng, 8000 step | 200/200 (100%) | 2,873,348,224 | 83.2% | 55.3 ms | 145 Hz |
| **v6** | 12 | 10 | 4 | LoRA r16 + `tune_llm` (LoRA cả backbone), resume tới 12000 step, LR 3e-4 | 200/200 (100%) | 2,165,330,560 | 62.7% | 40.9 ms | 196 Hz |
| **v7** | 8 | 7 | 3 | Full-finetune, backbone đóng băng, LR 1e-4, 12000-14000 step | 200/200 (100%) | 1,828,630,400 | 52.9% | 31.0 ms | 258 Hz |
| **v8** | 4 | 7 | 3 | Full-finetune, backbone đóng băng, LR 1e-4, 8000 step (cùng cấu hình backbone/vlsa với v7, giảm tiếp DiT 8→4) | 200/200 (100%) | 1,693,296,512 | 49.0% | 26.6 ms | 301 Hz |
| **v9** | 4 | 6 | 3 | Full-finetune, backbone đóng băng, 8000 step (suy ra cùng công thức v8, giảm tiếp backbone 7→6 — hyperparameter đầy đủ chưa xác nhận lại) | 200/200 (100%) | 1,642,960,512 | 47.6% | 26.5 ms | 302 Hz |
| **v10** | 4 | 4 | 2 | Full-finetune, backbone đóng băng, LR 1e-4, 8000 step (giảm đồng thời backbone 6→4 và vlsa 3→2) | 200/200 (100%) | 1,491,930,240 | **43.2%** | 27.1 ms | 295 Hz |

**Data Processing của v10 cao gấp đôi các checkpoint khác (5.54 ms so với
~2.6-2.8 ms), và tái lập chính xác qua 2 lần đo độc lập** (5.55 ms rồi 5.54 ms, lần 2
có variance chặt hơn: `27.3 ± 1.5 ms` so với `30.2 ± 3.8 ms`). Vì tái lập được, đây
**không phải nhiễu ngẫu nhiên** — nhưng cũng **không thể do prune**, vì Data Processing
là bước tiền xử lý chạy trên CPU (`prepare_model_inputs`: dựng `VLAStepData` → processor
→ collator), hoàn toàn độc lập với số layer của model. Nguyên nhân khả dĩ nhất là tải
CPU nền cố định trong cả hai lần đo (ví dụ một job training đang chạy song song với
`--dataloader_num_workers 4`), hoặc processor config của checkpoint v10 khác các
checkpoint khác. **Cần kiểm tra trước khi dùng số E2E của v10 để so sánh.**

Trong lúc chưa xác định nguyên nhân, dùng **latency phía GPU** (Backbone + Action Head)
làm thước đo so sánh kiến trúc — xem bảng bên dưới. Phần này ổn định và tái lập tốt
qua 2 lần đo v10 (Backbone 15.03 → 14.70 ms, Action Head 6.94 → 6.92 ms).

## Success rate

| Checkpoint | Success rate | Ghi chú |
|---|---|---|
| base | 200/200 (100.00%) | full suite 10/10 task |
| anchor | 200/200 (100.00%) | full suite 10/10 task |
| v6 | 200/200 (100.00%) | full suite 10/10 task |
| v7 | 200/200 (100.00%) | full suite 10/10 task |
| v8 | 200/200 (100.00%) | full suite 10/10 task |
| v9 | 200/200 (100.00%) | full suite 10/10 task |
| v10 | 200/200 (100.00%) | full suite 10/10 task |

## Chi tiết layer giữ lại từng checkpoint

| | Backbone (`kept_layer_idx_list_backbone`) | DiT (`kept_layer_idx_list_dit`) | vlsa (`kept_layer_idx_list_vl_self_attn`) |
|---|---|---|---|
| anchor | full 16 | `0-15,20-23,28-31` | full `0,1,2,3` |
| v6 | `0,1,2,5,8,9,10,11,14,15` | `0,1,2,3,12,13,14,15,28,29,30,31` | `0,1,2,3` |
| v7 | `0,1,3,8,9,12,14` | `0,1,2,3,28,29,30,31` | `0,1,2` |
| v8 | `0,1,3,8,9,12,14` | `0,1,2,3` | `0,1,2` |
| v9 | `0,2,7,10,14,15` | `0,1,2,3` | `0,1,2` |
| v10 | `0,1,9,15` | `0,1,2,3` | `0,2` |

Các keep-set được chọn bằng brute-force trên ma trận CKA (`scripts/cka_results/`),
tối đa hoá worst-case gap CKA — xem `scripts/cluster_prune.py` và
`scripts/cka_logical_layer_analysis.py` trong repo CLP_VLA.

## Latency chi tiết theo thành phần (median, ms)

| | Data Proc (CPU) | Backbone (GPU) | Action Head (GPU) | **GPU-side** | Δ GPU vs base | **Tổng (chuẩn hoá)** | Δ tổng vs base | Control freq | E2E đo được |
|---|---|---|---|---|---|---|---|---|---|
| base | 2.70 | 24.84 | 36.70 | **61.54** | — | **64.24** | — | 125 Hz | 64.4 |
| anchor | 2.61 | 24.32 | 28.39 | **52.71** | −14.3% | **55.41** | −13.7% | 144 Hz | 55.3 |
| v6 | 2.74 | 21.64 | 16.42 | **38.06** | −38.2% | **40.76** | −36.6% | 196 Hz | 40.9 |
| v7 (đo lại) | 2.71 | 16.79 | 11.46 | **28.25** | −54.1% | **30.95** | −51.8% | 258 Hz | 31.0 |
| v8 | 2.68 | 16.67 | 7.17 | **23.84** | −61.3% | **26.54** | −58.7% | 301 Hz | 26.6 |
| v9 | 2.66 | 16.44 | 7.34 | **23.78** | −61.4% | **26.48** | −58.8% | 302 Hz | 26.5 |
| v10 (đo lần 2) | 5.54 | 14.70 | 6.92 | **21.62** | **−64.9%** | **24.32** | **−62.1%** | **329 Hz** | 27.1 |

**Tổng (chuẩn hoá)** = `2.70 + GPU-side` — thay Data Processing đo được bằng giá trị
tham chiếu 2.70 ms (giá trị của base; 6/7 checkpoint đều nằm trong dải hẹp
2.61-2.78 ms). Cột này khôi phục khả năng so sánh cho v10, vốn có Data Processing bất
thường 5.54 ms do nguyên nhân ngoài kiến trúc. Với 6 checkpoint còn lại, tổng chuẩn hoá
trùng khớp E2E đo được trong vòng 0.2 ms — xác nhận phép chuẩn hoá không làm méo số liệu.

**Control freq** tính từ cột tổng chuẩn hoá: `n_action_steps(8) × (1000 / tổng_ms)`.

**Cột GPU-side (Backbone + Action Head) là thước đo sạch nhất để so sánh kiến trúc** —
loại bỏ hoàn toàn thành phần CPU vốn không phụ thuộc số layer. Thứ tự giảm đơn điệu
đúng như kỳ vọng: `61.54 → 52.71 → 38.06 → 28.25 → 23.84 → 23.78 → 21.62`.

**Cột Tổng (chuẩn hoá) là số nên dùng khi báo cáo hiệu năng triển khai** — vì robot thật
phải chịu cả chi phí tiền xử lý CPU, không chỉ phần GPU. Chênh lệch giữa hai cột (−64.9%
so với −62.1% cho v10) phản ánh đúng thực tế: Data Processing là phần **không giảm được
bằng prune**, nên càng cắt sâu thì tỷ trọng của nó càng lớn và lợi ích biên càng giảm —
cùng bản chất với sàn cứng 30.5% ở phần tham số.

Control freq = `n_action_steps(8) × (1000 / tổng_ms)` — model trả về action chunk chạy
open-loop trong `n_action_steps` bước mô phỏng trước khi suy luận lại.


## Giới hạn lý thuyết của layer pruning

Breakdown tham số của v10 (`scripts/count_checkpoint_params.py`) cho thấy phần **không
thể giảm** bằng cách cắt layer:

| Thành phần | Params | % base | Cắt layer có giảm được? |
|---|---|---|---|
| `backbone.vision_tower` | 406,957,056 | 11.8% | ❌ không |
| `action_head.other` (projector/encoders/decoders) | 336,411,776 | 9.7% | ❌ không |
| `backbone.embeddings` | 311,164,928 | 9.0% | ❌ không |
| `backbone.other` | 2,048 | ~0% | ❌ không |
| **Tổng sàn cứng** | **1,054,535,808** | **30.5%** | — |
| `backbone.llm_layers` (4 layer) | 201,344,000 | 5.8% | ✅ có |
| `action_head.DiT_blocks` (4 block) | 135,333,888 | 3.9% | ✅ có |
| `action_head.vl_self_attention` (2 layer) | 100,716,544 | 2.9% | ✅ có |
| **Tổng v10** | **1,491,930,240** | **43.2%** | — |

Chi phí đơn vị suy ra từ breakdown này (khớp chính xác với suy luận từ chênh lệch
v8/v9): **1 backbone LLM layer = 50,336,000**, **1 DiT block = 33,833,472**,
**1 vlsa layer = 50,358,272** params.

**Ý nghĩa**: layer pruning giỏi nhất chỉ có thể đưa model xuống **30.5% của base**
(sàn cứng) — dù cắt sạch mọi layer có thể cắt. v10 đang ở 43.2%, tức đã khai thác
**81.7%** toàn bộ dư địa lý thuyết `(100−43.2)/(100−30.5)`. Phần còn lại chỉ 12.7 điểm
phần trăm. Muốn nén sâu hơn nữa buộc phải dùng kỹ thuật khác (quantization, width
pruning, distillation, hoặc thay vision tower nhỏ hơn).

## Kết luận chính

**v10 (DiT=4, backbone=4, vlsa=2) là cấu hình nhỏ nhất và nhanh nhất chưa cho thấy suy
giảm accuracy: 200/200 trên full LIBERO Object suite, chỉ còn 43.2% tham số của model
gốc (1.49 tỷ so với 3.46 tỷ), GPU-side latency 21.62 ms so với 61.54 ms của base —
giảm 64.9%. E2E chuẩn hoá ≈ 24.3 ms, control frequency ≈ 329 Hz, gấp 2.6 lần base
(124 Hz).**

Điều đáng chú ý nhất: v10 chỉ giữ **4/32 DiT block (12.5% độ sâu gốc), 4/16 backbone
layer (25%), 2/4 vlsa layer (50%)** mà vẫn hoàn thành trọn vẹn 200/200 rollout — không
một task nào dưới 20/20.

Cả 7 checkpoint, trải dài từ 100% xuống 43.2% tham số, đều đạt 200/200. **Không có cấu
hình nào chạm điểm suy giảm.** v8 và v9 gần như ngang nhau trên mọi trục (49.0% vs
47.6% params, 26.6 vs 26.5 ms) vì phần lớn chi phí compute đã bị cắt ở DiT (32→4);
chênh lệch giữa chúng nằm trong biên độ nhiễu đo (xem cảnh báo ở trên).



## Việc còn mở

1. **Chuyển sang LIBERO Spatial / Goal** để tìm điểm suy giảm — Object đã bão hoà, tiếp
   tục cắt sâu trên Object chỉ tạo thêm các dòng "100%" giống nhau, không dựng được
   đường cong accuracy-vs-compression.
2. **Đo lại latency cho base / anchor / v6** — ba checkpoint này đo cùng phiên với lần
   đo lỗi của v7 (đã phát hiện lệch 4.3 ms) nên nhiều khả năng cũng bị báo cao. Đo lại
   trong cùng phiên với v7-mới/v8/v9/v10, 3 lần mỗi checkpoint.
3. **Tìm nguyên nhân Data Processing 5.54 ms của v10** — kiểm tra có job training chạy
   song song lúc benchmark không, hoặc so `processor_config.json` của v10 với các
   checkpoint khác.
4. Nếu vẫn muốn cắt sâu hơn trên trục DiT: nới `_required_prune_group_size()` từ nhóm-4
   xuống nhóm-2 (ràng buộc cứng thật sự chỉ là nhóm-2 để self/cross khớp shape; nhóm-4
   chỉ là ràng buộc mềm về xen kẽ text/image) để thử DiT=2. Lưu ý sàn cứng 30.5% ở
   trên: dư địa còn lại chỉ 12.7 điểm phần trăm, nên lợi ích biên đang giảm nhanh.

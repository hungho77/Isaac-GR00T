# CLP-Style Structural Pruning cho GR00T-N1.7 — Báo cáo kỹ thuật

Repo: fork riêng của `Isaac-GR00T` (NVIDIA), áp dụng phương pháp CKA-guided layer pruning
(CLP) — vốn được thiết kế cho GR00T-N1.5 — sang kiến trúc GR00T-N1.7
(backbone Qwen3-VL/Cosmos-Reason2-2B + action head `AlternateVLDiT`).

---

## 1. Thay đổi so với bản gốc NVIDIA

### 1.1 File mới (3)

| File | Vai trò |
|---|---|
| `gr00t/utils/peft.py` | `get_lora_model()` — bọc LoRA (HuggingFace `peft`) lên các Linear layer khớp pattern attention/FFN/AdaLN, có thể giới hạn chỉ trong `action_head` hoặc mở rộng cả backbone |
| `scripts/merge_lora_checkpoint.py` | Dựng lại kiến trúc pruned từ base checkpoint, load adapter LoRA, `merge_and_unload()` thành checkpoint đầy đủ để eval |
| `gr00t/eval/sim/LIBERO/run_object_suite.sh` | Chạy tuần tự cả 10 task của LIBERO Object suite, in success rate từng task + pooled |

### 1.2 File sửa (10)

| File | Thay đổi |
|---|---|
| `gr00t/model/modules/qwen3_backbone.py` | Thêm `prun_layers(kept_layer_idx_list)` — subset `ModuleList` layer theo danh sách index giữ lại |
| `gr00t/model/modules/dit.py` | Thêm `prun_layers()` + `_required_prune_group_size()` cho `DiT`/`AlternateVLDiT` và `SelfAttentionTransformer`. Ràng buộc: đơn vị cắt tối thiểu là nhóm 4 block liên tiếp (do `AlternateVLDiT` alternate self/cross-attention theo `idx % 2` **và** text/image theo `idx % 4`, khớp sai nhóm gây shape-mismatch hoặc gán nhầm vai trò một cách âm thầm) |
| `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` | `Gr00tN1d7.prun_layers()` — gọi prune xuống cả 3 sub-module. **Không** tự động gọi trong `__init__` |
| `gr00t/model/gr00t_n1d7/setup.py` | Truyền các cờ prune làm override kwargs vào `AutoModel.from_pretrained`; gọi `model.prun_layers()` **sau khi** state_dict đã load xong vào kiến trúc đầy đủ |
| `gr00t/policy/gr00t_policy.py` | Khi eval checkpoint đã prune: dựng kiến trúc pruned trước (`prun_layers()`), rồi mới load state_dict nhỏ — thứ tự **ngược lại** so với lúc train |
| `gr00t/configs/model/gr00t_n1d7.py`, `gr00t/configs/finetune_config.py`, `gr00t/configs/training/training_config.py` | Thêm field `prune_model`, `kept_layer_idx_list_backbone/_dit/_vl_self_attn`, `lora_rank/_alpha/_dropout/_action_head_only`, `resume_from_checkpoint` |
| `gr00t/experiment/launch_finetune.py`, `gr00t/experiment/experiment.py` | Nối CLI → config → model; gọi `get_lora_model()` khi `lora_rank > 0` |
| `gr00t/eval/rollout_policy.py` | Thêm `--policy-client-timeout-ms` (mặc định 15000) |
| `gr00t/eval/sim/LIBERO/libero_env.py` | Xem mục 1.4 |

### 1.3 Hai lỗi thứ tự prune (bug quan trọng, đã sửa)

Cả hai đều cùng một loại lỗi: **thứ tự giữa "load trọng số" và "cắt kiến trúc" bị đảo**, gây mismatch key khi load `state_dict`.

- **Phía train**: nếu prune trước khi load → checkpoint gốc (kiến trúc đầy đủ) không khớp kiến trúc đã bị thu nhỏ → hàng loạt "unexpected keys". Sửa: `setup.py` load full trước, prune sau (mục 1.2).
- **Phía eval**: nếu dựng kiến trúc đầy đủ rồi mới load checkpoint đã prune (nhỏ) → thiếu key cho các layer không tồn tại → load random-init garbage vào các phần bị thiếu, không báo lỗi rõ ràng. Sửa: `gr00t_policy.py` dựng kiến trúc pruned trước, load sau (ngược chiều với train).

### 1.4 Bug gripper-sign 

**Mô tả**: dataset LIBERO Object convert từ LeRobot v3.0→v2.1 (`scripts/lerobot_conversion/convert_v3_to_v2.py`) lưu kênh gripper theo convention `[-1,+1]` (+1=đóng), trong khi checkpoint gốc NVIDIA dùng convention `[0,1]` (0=đóng). `gr00t/eval/sim/LIBERO/libero_env.py` áp cố định 2 phép biến đổi cho convention gốc (`normalize_gripper_action` + `invert_gripper_action`), nên với model train trên dataset đã convert, gripper bị **lật dấu hai lần thay vì một** — robot mở kẹp đúng lúc lẽ ra phải đóng.

**Bằng chứng**: `scripts/diff_statistics.py` so `statistics.json` của checkpoint finetune với base — std gripper khớp tuyệt đối (`0.9976 = 2 × 0.4988`), mean ngược dấu chính xác (`+0.0714` thay vì `−0.0714`), không thể là trùng hợp.

**Vì sao khó phát hiện**: loss training vẫn giảm đẹp (model học đúng dữ liệu nó được đưa, lỗi nằm hạ nguồn của loss); mọi cấu hình kiến trúc đều cho đúng 0% (lỗi dấu cố định, không liên quan capacity); base checkpoint (đúng convention) vẫn chạy 100% nên càng khiến nghi ngờ đổ về phía model đã prune.

**Sửa** (`libero_env.py`): thêm biến môi trường `GR00T_GRIPPER_SIGNED=1` — bật thì đi thẳng (bỏ qua 2 phép biến đổi cũ), tắt thì giữ nguyên đường gốc. Base checkpoint chạy **không** cờ, mọi checkpoint finetune trên dataset đã convert chạy **có** cờ.

Hai can thiệp thử nghiệm khác trong cùng file, kết quả:
- `gripper_debounce_n` (hysteresis chống dao động nhanh open/close): trung tính, không cải thiện — nguyên nhân thật là lỗi dấu, không phải thiếu quyết đoán.
- `pose_ema_alpha` (làm mượt EMA cho 6 chiều pose liên tục): **có hại** — vì `state.x/y/z/gripper` được feedback làm input bước sau, làm mượt/trễ action khiến state thật lệch khỏi phân phối lúc train, đẩy policy ra ngoài phân phối (biểu hiện: tay co về gần thân, nhận diện vật sai). Đã tắt (`alpha=1.0`, debounce=1), giữ lại trong code kèm comment giải thích để không ai vô tình bật lại.

---

## 2. Kết quả benchmark

Xem [PRUNING_RESULTS.md](PRUNING_RESULTS.md) — bảng 4 checkpoint (base/anchor/v6/v7)
với success rate full 200 episode, tham số thật, latency theo từng thành phần.

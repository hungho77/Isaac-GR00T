# BÁO CÁO TRIỂN KHAI VÀ ĐÁNH GIÁ BỐN CHECKPOINT GR00T N1.7 TRÊN UR10e

**Nhiệm vụ:** UR10e + Robotiq Gripper - Pick up the cup  
**Người thực hiện:** Nguyễn Minh Đức  
**Ngày thực hiện:** 19/08/2026

---

## 1. Giới thiệu chung

Báo cáo tổng hợp cách triển khai, huấn luyện, recovery và đánh giá bốn checkpoint GR00T N1.7 cho nhiệm vụ gắp cốc bằng UR10e và Robotiq Gripper. Mục tiêu là xác định cấu hình CKA pruning có trade-off tốt nhất giữa độ chính xác trên robot thật và thời gian inference phía server RTX 5090.

Kết quả chính: **checkpoint 4, giữ 6/4/2 lớp, đạt 20/20 lần thành công với độ trễ 33 ms**. Đây là checkpoint tối ưu trong bốn cấu hình đã thực nghiệm: chỉ chậm hơn checkpoint 3 khoảng 3 ms nhưng cải thiện 30 điểm phần trăm success rate.

### 1.1. Thiết lập dữ liệu và giao diện điều khiển

| Thành phần | Thiết lập |
|---|---|
| Mô hình nền | NVIDIA GR00T N1.7-3B |
| Robot | UR10e, Robotiq Gripper |
| Dataset | `khanhnd61/ur10e-cup`, LeRobot v3; chuyển đổi sang GR00T dataset format trước khi train |
| Quy mô dữ liệu | 81 episodes, 49.779 frames |
| Chia dữ liệu | Train 0-64; validation 65-72; test 73-80 |
| Quan sát | Camera side, camera wrist; trạng thái sáu joint cánh tay và gripper |
| Action | `shoulder_pan`, `shoulder_lift`, `elbow`, `wrist_1`, `wrist_2`, `wrist_3`, `gripper` |
| Quy ước action | Joint target tuyệt đối theo radian; gripper 1 = mở, 0 = đóng |
| Action horizon | 16 x 7 |
| Seed | 42 |

### 1.2. Cấu hình theo branch deploy

Branch [`nmduc`](https://github.com/hungho77/Isaac-GR00T/tree/nmduc) mở rộng main để hỗ trợ pipeline CKA trong điều kiện tài nguyên hạn chế:

- Chuyển đổi UR10e dataset, capture activation, tính linear CKA và tạo pruning manifest.
- Pruning `action_dit`, `backbone_language` và `vl_self_attention`.
- Lưu `original_depth`, `keep_indices`, `_gr00t_original_index` và thông tin topology.
- Hỗ trợ Action-DiT LoRA, staged resume, T4 mode và metadata/fingerprint.
- Đánh giá MSE, MAE, prediction-vs-ground-truth, latency, VRAM, kích thước mô hình và heatmap.

Branch [`ducnm`](https://github.com/hungho77/Isaac-GR00T/tree/ducnm) dùng cho triển khai native CKA:

- Nhúng `cka_pruning_manifest` vào config và tái dựng kiến trúc pruned trước khi load weight.
- Giữ nguyên chỉ số layer gốc và các ràng buộc topology.
- Có strict native-checkpoint preflight và server fail-closed.
- Chuẩn hóa UR10e modality, processor, statistics và hợp đồng action 16 x 7.
- Kiểm tra thứ tự joint, radian, absolute action và chuẩn hóa gripper.

> **Lưu ý tương thích:** Với checkpoint clone `hungho77/main`, notebook đã bổ sung phần triển khai CKA tương thích để checkpoint có thể được load và deploy đúng bằng branch `ducnm`.

---

## 2. Checkpoint 1 - LoRA recovery trên cấu hình Kaggle/T4

| Thuộc tính | Giá trị |
|---|---|
| Model | [`duc996/checkpoint1`](https://huggingface.co/duc996/checkpoint1) |
| Branch clone/train | `hungho77/Isaac-GR00T - nmduc` |
| Branch deploy | `nmduc`, legacy deployment path |
| Kiến trúc giữ lại | Action-DiT 24/32; language backbone 12/16; VLSA 4/4 |
| Mức cắt tổng | 23,08% theo tổng số lớp không trọng số, loại 12/52 lớp |
| Recovery | 6.000 steps; matched Action-DiT LoRA |
| LoRA | Rank 16; alpha 32; dropout 0,05; 12.566.528 tham số LoRA |
| Tổng tham số trainable | 21.617.664 |
| Batch | Global 1; gradient accumulation 8; effective batch 8 |
| Optimizer config | LR 5e-5; weight decay 1e-5; warmup ratio 0,05 |
| Precision | FP16 |
| Checkpoint | Save mỗi 500 steps; giữ 1 checkpoint |
| Dataloader workers | 0 |
| Regularization | State dropout 0,2; không color jitter |
| Trainable/frozen | Projector và diffusion small modules trainable; VLLN/LLM/vision frozen |

### Kết quả

- **Success rate:** Không có phép đo hợp lệ; dừng thử nghiệm vì robot rung lắc rất mạnh.
- **Lý do fail:** Điều khiển không ổn định, không thể tiếp tục rollout an toàn.
- **Inference time:** 58 ms phía server RTX 5090.

Checkpoint 1 chứng minh pipeline có thể tạo và phục vụ checkpoint pruned, nhưng recovery tối ưu cho T4/Kaggle và deploy legacy chưa đủ ổn định cho robot thật. Vì không có rollout hợp lệ, checkpoint này chỉ là mốc kỹ thuật ban đầu, không phải mốc độ chính xác.

---

## 3. Checkpoint 2 - Full retained-module recovery 6K

| Thuộc tính | Giá trị |
|---|---|
| Model | [`Luke99662244/checkpoint2`](https://huggingface.co/Luke99662244/checkpoint2) |
| Branch clone/train | `hungho77/Isaac-GR00T - nmduc` |
| Branch deploy | `nmduc`, legacy deployment path |
| Kiến trúc giữ lại | Action-DiT 16/32; language backbone 8/16; VLSA 4/4 |
| Mức cắt tổng | 46,15% theo tổng số lớp không trọng số, loại 24/52 lớp |
| Recovery | 6.000 steps; full recovery trên module còn giữ; không LoRA |
| Batch | Global 1; gradient accumulation 8; effective batch 8 |
| Optimizer config | LR 1e-4; weight decay 1e-5; warmup ratio 0,05 |
| Precision | BF16 + TF32 |
| Checkpoint | Save mỗi 250 steps; giữ 1 checkpoint |
| Dataloader workers | 0 |
| Regularization | State dropout 0,2; color jitter 0,3/0,4/0,5/0,08 |
| Trainable/frozen | Projector, diffusion và VLLN trainable; LLM/vision frozen |

### Kết quả và cải thiện so với checkpoint 1

- **Success rate:** Không có phép đo hợp lệ.
- **Lý do fail:** Robot không hạ cánh tay xuống cốc, chỉ đung đưa nhẹ ở trên cao.
- **Inference time:** 49 ms phía server RTX 5090.
- **Cải thiện:** Không còn rung lắc mạnh; độ trễ giảm 9 ms, tương đương 15,5% so với checkpoint 1.

Checkpoint 2 cải thiện rõ độ ổn định động học nhưng chưa hoàn thành pha tiếp cận cốc. Việc chuyển sang full retained-module recovery và tăng pruning giúp giảm độ trễ, nhưng pipeline legacy vẫn chưa tạo ra hành vi hoàn chỉnh trên robot thật.

---

## 4. Checkpoint 3 - Native CKA, pruning rất sâu 4/2/1

| Thuộc tính | Giá trị |
|---|---|
| Model | [`Luke99662244/single-gpu`](https://huggingface.co/Luke99662244/single-gpu) |
| Branch clone/train | `hungho77/main@9c7e746`, kèm minimal CKA overlay trong notebook |
| Branch deploy | `ducnm@e2b098294deeb5c46002f9acbc7232031aa8fab4`, native CKA |
| Kiến trúc giữ lại | Action-DiT 4/32; language backbone 2/16; VLSA 1/4 |
| Mức cắt tổng | 86,54% theo tổng số lớp không trọng số, loại 45/52 lớp |
| Recovery | 10.000 steps; full retained-module recovery; không LoRA |
| Batch | Global 32; gradient accumulation 1; effective batch 32 |
| Optimizer config | LR 1e-4; weight decay 1e-5; warmup ratio 0,05 |
| Precision | BF16 + TF32 |
| Checkpoint | Save mỗi 1.000 steps; giữ 2 checkpoint |
| Dataloader workers | 4 |
| Regularization | State dropout 0,2; color jitter 0,3/0,4/0,5/0,08 |
| Trainable/frozen | Projector, diffusion và VLLN trainable; LLM/vision frozen |
| Phần cứng | 1 x NVIDIA L40S, lớp VRAM từ 40 GiB |

> **Ghi chú:** Trong notebook đã có thêm phần triển khai CKA tương thích để đúng với branch deploy `ducnm`, dù repository huấn luyện được clone từ `hungho77/main`.

### Kết quả và cải thiện so với checkpoint 2

- **Success rate:** 14/20, tương đương 70%.
- **Lý do fail:** Sáu trường hợp robot không gắp được cốc.
- **Inference time:** 30 ms phía server RTX 5090.
- **Cải thiện:** Lần đầu có rollout robot thật hợp lệ; khắc phục lỗi cánh tay không hạ; giảm 19 ms, tương đương 38,8% so với checkpoint 2.

Checkpoint 3 cho thấy native CKA và full single-GPU recovery tạo ra mô hình rất gọn nhưng vẫn thực thi được nhiệm vụ. Tuy nhiên, pruning 86,54% làm suy giảm độ tin cậy của pha gắp: 30% số thử nghiệm thất bại dù độ trễ là thấp nhất.

---

## 5. Checkpoint 4 - Native CKA 6/4/2, exact resume 12K

| Thuộc tính | Giá trị |
|---|---|
| Model | [`Luke99662244/642`](https://huggingface.co/Luke99662244/642) |
| Branch clone/train | `hungho77/main@9c7e746`, kèm minimal CKA overlay trong notebook |
| Branch deploy | `ducnm@e2b098294deeb5c46002f9acbc7232031aa8fab4`, native CKA |
| Kiến trúc giữ lại | Action-DiT 6/32; language backbone 4/16; VLSA 2/4 |
| Keep indices | DiT `[0,1,2,3,4,31]`; language `[0,1,9,15]`; VLSA `[0,3]` |
| Cách chọn | `adjacent_linear_cka_topk`; bảo vệ lớp đầu/cuối và giữ topology Action-DiT |
| Mức cắt tổng | 76,92% theo tổng số lớp không trọng số, loại 40/52 lớp |
| Recovery | Full recovery 12K, thực hiện hai giai đoạn bằng exact resume tại step 6K; không LoRA |
| Exact resume | Weight, optimizer, scheduler, global step, RNG state và cùng pruning manifest |
| Batch | Global 32; gradient accumulation 1; effective batch 32 |
| Optimizer config | LR 1e-4; weight decay 1e-5; warmup ratio 0,05 |
| Precision | BF16 + TF32 |
| Checkpoint | Save mỗi 1.000 steps; giữ 2 checkpoint |
| Dataloader workers | 4 |
| Regularization | State dropout 0,2; color jitter 0,3/0,4/0,5/0,08 |
| Trainable/frozen | Projector, diffusion và VLLN trainable; LLM/vision frozen |
| Phần cứng | 1 x NVIDIA L40S, full single-GPU configuration |

> **Ghi chú:** Trong notebook đã có thêm phần triển khai CKA tương thích để đúng với branch deploy `ducnm`, dù repository huấn luyện được clone từ `hungho77/main`. Recovery 12K tương đương chạy from scratch về mặt kết quả vì hai giai đoạn dùng exact resume và toàn bộ cấu hình tổng thể được giữ nguyên.

### Kết quả và cải thiện so với checkpoint 3

- **Success rate:** 20/20, tương đương 100% trong tập thử nghiệm hiện tại.
- **Lý do fail:** Không ghi nhận trường hợp thất bại.
- **Inference time:** 33 ms phía server RTX 5090.
- **Cải thiện:** Thêm sáu lần thành công và tăng 30 điểm phần trăm SR; latency chỉ tăng 3 ms so với checkpoint 3.

Việc giữ thêm hai lớp Action-DiT, hai lớp language backbone và một lớp VLSA, đồng thời kéo dài recovery lên 12K, đem lại cải thiện lớn về độ tin cậy. Độ trễ 33 ms vẫn thấp hơn checkpoint 1 khoảng 43,1% và thấp hơn checkpoint 2 khoảng 32,7%.

---

## 6. So sánh bốn checkpoint

Mức cắt tổng dưới đây được tính trên 52 lớp, gồm 32 Action-DiT, 16 language backbone và 4 VLSA. Đây là tỷ lệ số lớp, không phải tỷ lệ tham số hoặc FLOPs. Chỉ checkpoint 3 và 4 có rollout robot thật hợp lệ; checkpoint 1 và 2 không được diễn giải là 0% success rate.

| Checkpoint | Keep DiT/Lang/VLSA | Mức cắt tổng | Recovery | Kết quả | Inference RTX 5090 |
|---:|---:|---:|---|---|---:|
| 1 | 24/12/4 | 23,08% | 6K, LoRA | Không có SR hợp lệ | 58 ms |
| 2 | 16/8/4 | 46,15% | 6K, full retained | Không có SR hợp lệ | 49 ms |
| 3 | 4/2/1 | 86,54% | 10K, full retained | 14/20, 70% | **30 ms** |
| **4** | **6/4/2** | **76,92%** | **12K, exact resume 6K** | **20/20, 100% quan sát** | **33 ms** |

### 6.1. Đánh giá trade-off

- Checkpoint 1 và 2 giảm latency nhưng chưa đạt điều kiện triển khai vì rollout không hợp lệ.
- Checkpoint 3 nhanh nhất, 30 ms, nhưng pruning 86,54% chỉ đạt 14/20. Cấu hình 4/2/1 đã vượt quá ngưỡng ổn định cho nhiệm vụ gắp cốc.
- Checkpoint 4 tăng latency 3 ms so với checkpoint 3, tương đương 10%, nhưng tăng success rate từ 70% lên 100% trong 20 lần thử.
- So với checkpoint 1, checkpoint 4 nhanh hơn 43,1%, loại bỏ rung lắc và đạt đủ 20/20 lần thành công.

**Checkpoint tối ưu hiện tại:** checkpoint 4, giữ 6/4/2 lớp và recovery 12K. Đây là điểm trade-off tốt nhất trong bốn cấu hình đã thử: độ chính xác quan sát cao nhất, native CKA deploy ổn định và latency chỉ cao hơn cấu hình nhanh nhất 3 ms.

### 6.2. Insight tổng hợp về phương pháp CKA của paper

Pipeline áp dụng ý tưởng cốt lõi của phương pháp trong paper: dùng **linear CKA giữa các biểu diễn của layer** để nhận diện các layer có biểu diễn tương đồng cao, coi đó là tín hiệu dư thừa, sau đó loại bớt layer và recovery mô hình. Trong triển khai hiện tại, `adjacent_linear_cka_topk` ưu tiên loại các layer nội bộ có CKA liền kề cao, đồng thời bảo vệ layer biên và giữ topology cần thiết.

Các thực nghiệm cho phép rút ra các insight sau:

1. **CKA xác định dư thừa biểu diễn tốt hơn pruning theo vị trí cố định.** Keep list không chỉ lấy đều hoặc chỉ giữ layer đầu/cuối; nó dựa vào activation thực tế trên dữ liệu UR10e và được lưu trong manifest để tái lập.
2. **Tỷ lệ pruning không tỷ lệ tuyến tính với chất lượng.** Cắt sâu hơn giúp latency thấp hơn, nhưng từ 76,92% lên 86,54% chỉ tiết kiệm 3 ms trong khi success rate quan sát giảm từ 20/20 xuống 14/20.
3. **Language backbone có ảnh hưởng lớn đến độ tin cậy task-level.** Kết quả bổ sung trước đây cho thấy 4/6/2 đạt 19/20, 4/4/2 đạt 18/20, trong khi 12/4/3 chỉ đạt 16/20. Giữ nhiều Action-DiT hơn không tự động bù được khi language backbone bị cắt quá mạnh.
4. **Action-DiT vẫn là thành phần chính của quỹ đạo điều khiển.** Cấu hình 6/4/2 tốt hơn 4/2/1 vì giữ thêm năng lực sinh và hiệu chỉnh trajectory, đặc biệt ở pha tiếp cận và đóng gripper.
5. **VLSA có thể cắt sâu nhưng không nên chỉ giữ một lớp trong cấu hình cực đoan.** Giữ 2/4 lớp cung cấp biên an toàn tốt hơn cho liên kết thị giác-ngôn ngữ-hành động; dữ liệu hiện tại chưa đủ để kết luận 1/4 luôn không đạt.
6. **Recovery là một phần bắt buộc, không phải bước phụ.** CKA chỉ quyết định kiến trúc/keep list; khả năng hoạt động sau pruning phụ thuộc mạnh vào full retained-module recovery, batch đủ lớn, precision ổn định và resume chính xác.
7. **Manifest và native-load contract quan trọng ngang thuật toán chọn layer.** Checkpoint 1 và 2 cho thấy checkpoint có thể đúng về weight nhưng vẫn thất bại nếu branch deploy không tái dựng đúng kiến trúc, action contract hoặc normalization.
8. **Điểm hợp lý hiện tại nằm quanh 75%-77% tổng số layer.** Với dữ liệu hiện có, 6/4/2 là lựa chọn mạnh nhất; vùng 4/4/2 đến 6/6/2 là vùng nên tiếp tục khảo sát nếu muốn giảm thêm latency hoặc tăng robustness.

### 6.3. Ưu điểm của phương pháp

- **Data-driven:** quyết định pruning dựa trên activation của dữ liệu mục tiêu thay vì chỉ dựa vào magnitude của weight.
- **Diễn giải được:** heatmap, similarity score, keep indices và pruning manifest cho phép truy vết layer nào bị loại và tại sao.
- **Phù hợp với layer pruning có cấu trúc:** giảm trực tiếp độ sâu kiến trúc, có khả năng giảm latency inference thực tế hơn pruning thưa không cấu trúc.
- **Linh hoạt theo module:** có thể đặt ngân sách riêng cho Action-DiT, language backbone và VLSA.
- **Tái lập được:** manifest, original indices và fingerprint giúp train/deploy cùng một kiến trúc.
- **Cho trade-off tốt trong thực nghiệm:** checkpoint 4 nhanh hơn checkpoint 1 khoảng 43,1% nhưng đạt 20/20 lần thành công quan sát.

### 6.4. Nhược điểm và giới hạn

- **CKA đo độ giống biểu diễn, không đo trực tiếp task success.** Hai layer giống nhau trên calibration set vẫn có thể khác nhau ở tín hiệu nhỏ nhưng quan trọng cho gắp cốc.
- **Nhạy với calibration data và sampling.** Keep list có thể thay đổi nếu episode, frame, seed hoặc phân bố camera thay đổi.
- **Adjacent CKA bỏ qua tương tác xa.** So sánh layer liền kề không mô hình hóa đầy đủ quan hệ giữa các layer cách xa nhau hoặc ảnh hưởng liên module.
- **Ngân sách theo số layer không tương đương FLOPs/tham số.** Mức cắt tổng 76,92% là unweighted layer count; lợi ích phần cứng cần được xác nhận bằng latency, VRAM và kích thước thực tế.
- **Cần recovery tốn tài nguyên.** Pruning sâu không thể dùng trực tiếp; phải recovery hàng nghìn steps trên GPU mạnh.
- **Có rủi ro mismatch train/deploy.** Nếu branch deploy không hiểu manifest hoặc action contract, robot có thể rung, đứng cao hoặc sinh hành động sai dù offline metric tốt.
- **Cỡ mẫu robot thật còn nhỏ.** 20/20 là kết quả quan sát, chưa chứng minh success rate thật tuyệt đối 100%.

### 6.5. Hướng cải tiến tiếp theo

- Lặp lại checkpoint 4 với nhiều seed, vị trí cốc, ánh sáng và trạng thái reset; báo cáo khoảng tin cậy cho success rate.
- Ablation riêng từng module quanh vùng tốt: 4/4/2, 6/4/2, 6/6/2 và 8/4/2, giữ nguyên toàn bộ hyperparameter khác.
- Kết hợp CKA với độ nhạy loss hoặc gradient để tránh loại layer có CKA cao nhưng quan trọng cho task.
- Tính thêm parameter reduction, FLOPs, VRAM và end-to-end control latency thay vì chỉ dùng tỷ lệ số layer.
- Dùng calibration set đa dạng hơn, bao gồm cả các pha tiếp cận, đóng gripper và nâng cốc.
- Giữ native checkpoint preflight và action-contract validation là điều kiện bắt buộc trước rollout thật.

---

## 7. Kết luận cuối

Chuỗi bốn checkpoint thể hiện quá trình cải tiến từ pipeline T4/Kaggle và deploy legacy sang full single-GPU recovery với native CKA. Checkpoint 1 rung lắc mạnh; checkpoint 2 ổn định hơn nhưng không tiếp cận cốc; checkpoint 3 lần đầu hoàn thành nhiệm vụ với latency thấp nhất; checkpoint 4 đạt 20/20 với độ trễ gần tương đương.

Kết luận kỹ thuật quan trọng là **không nên tối đa hóa tỷ lệ pruning một cách đơn lẻ**. Với UR10e pick-up-cup, cấu hình 6/4/2 và recovery 12K tạo trade-off tốt hơn rõ rệt so với 4/2/1: giữ thêm năm lớp trên ba nhóm module làm latency tăng rất nhỏ nhưng khôi phục đáng kể độ tin cậy của chuyển động tiếp cận và gắp.

**Checkpoint 4 - `Luke99662244/642` - là checkpoint tối ưu tính đến ngày 19/08/2026** và nên được dùng làm nền cho các vòng kiểm thử mở rộng hoặc triển khai tiếp theo. 

### Nguồn tham chiếu

- [Báo cáo tuần 11](https://github.com/khanhnd61-vr/modelopt-weekly-report/blob/main/26ai.ducnm/Week11-Duc.md)
- [Checkpoint 1](https://huggingface.co/duc996/checkpoint1)
- [Checkpoint 2](https://huggingface.co/Luke99662244/checkpoint2)
- [Checkpoint 3](https://huggingface.co/Luke99662244/single-gpu)
- [Checkpoint 4](https://huggingface.co/Luke99662244/642)
- [Branch nmduc](https://github.com/hungho77/Isaac-GR00T/tree/nmduc)
- [Branch ducnm](https://github.com/hungho77/Isaac-GR00T/tree/ducnm)
- [Branch main](https://github.com/hungho77/Isaac-GR00T/tree/main)

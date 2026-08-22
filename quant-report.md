# Đánh giá HoloQ W4A4 trên GR00T N1.7 với LIBERO

**Ngày tổng hợp:** 22/08/2026  
**Phạm vi:** LIBERO simulation — `object`, `spatial`, `goal`, `long`  

## 1. Bối cảnh chung và mục tiêu

### 1.1. Bối cảnh

Work này mở rộng GR00T N1.7 để kiểm tra khả năng áp dụng W4A4 cho policy thao tác robot trong LIBERO. Mã triển khai được giữ riêng trên branch [`duc-quan`](https://github.com/hungho77/Isaac-GR00T/tree/duc-quan), không thay đổi branch `main` hoặc các branch khác. Bốn gói kết quả đều ghi nhận cùng source commit [`d0d78b72`](https://github.com/hungho77/Isaac-GR00T/commit/d0d78b72833c52e1cf2c52c60ca5a57af0f5f98b) và cùng checkpoint [`nvidia/GR00T-N1.7-LIBERO`](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO) tại revision `2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21`.

Hình thức hiện tại là **post-training fake quantization W4A4** theo HoloQ-style pipeline: trọng số và activation được lượng tử hóa về miền 4-bit để mô phỏng sai số, sau đó dequantize trước phép `F.linear`. Vì chưa có fused/native INT4 kernel, kết quả có ý nghĩa trực tiếp cho tính khả thi thuật toán, quality retention, task sensitivity và VRAM của runtime tham chiếu; không được dùng để khẳng định hiệu năng native.

### 1.2. Thiết lập chính

| Thành phần | Thiết lập |
|---|---|
| Model/checkpoint | GR00T N1.7 LIBERO `libero_10`, baseline BF16 |
| Quantization | W4A4 fake-quant |
| Phạm vi | 112 LLM Linear + 192 DiT Linear = 304 Linear |
| LLM | GPTQ weight solver; activation A4 động theo token |
| DiT | RTN weight solver; activation A4 theo từng denoising step |
| Denoising | 4 inference steps |
| Calibration | 10 trajectory/suite, một floating-point rollout cho mỗi task với initial state cố định |
| Evaluation | 20 episode/task/mode, 10 task/suite, initial state held out khỏi calibration |
| Pairing | BF16 và W4A4 dùng cùng seed theo task |
| Action horizon | 8 action steps |
| Episode horizon | tối đa 720 simulator steps |
| Accelerator Phase 1 | NVIDIA L4 |
| Quy trình | `setup → prepare → calibrate → build_pack → smoke_rollout → full_rollout → benchmark → package` |

### 1.3. Các thành phần đã bổ sung trên branch

So với nền GR00T N1.7 trước phần quantization, branch bổ sung hoặc tích hợp 28 file, tập trung vào:

- Hạ tầng quantization trong `gr00t/quantization/`: xác định scope, thu thập calibration, context theo denoising step,   builder, packing, runtime fake-quant và kiểm tra manifest.
- Công cụ `tools/build_holoq_n1d7_pack.py` để dựng pack W4A4 cho N1.7.
- Tích hợp pack/runtime vào `gr00t_n1d7.py`, `gr00t_policy.py` và inference server mà không thay cấu trúc nền của model N1.7.
- Pipeline LIBERO trong `examples/LIBERO/quantization/`: bốn config suite, runner Phase 1 có resume, rollout shard,   smoke/full rollout, recovery cell, notebook Modal và đóng gói checksum/inventory.
- Protocol full rollout 20 episode/task và paired seed cho BF16/W4A4, kèm test bảo vệ protocol.
- Các bản vá setup headless, cấu hình robosuite không tương tác và chuẩn hóa observation contract của LIBERO.
- Test cho quantization core, pack manifest, runtime, LIBERO observation và quy trình Modal.

### 1.4. Mục tiêu của work

1. Kiểm chứng GR00T N1.7 còn duy trì success rate ở mức hữu dụng khi 304 Linear được áp dụng W4A4.
2. So sánh BF16 và W4A4 bằng rollout ghép cặp trên đủ bốn suite LIBERO.
3. Phân tích theo task để tìm các nhiệm vụ đặc biệt nhạy với lượng tử hóa.
4. Đo mức giảm peak VRAM của runtime W4A4 tham chiếu.
5. Ước lượng mức giảm dung lượng logic của phần được lượng tử hóa và dung lượng hybrid khi có native backend.

## 2. Kết quả rollout theo từng suite

Mỗi suite dùng 10 task × 20 episode/task × 2 mode = 400 episode thô. Trên toàn bộ bốn suite, báo cáo phân tích 1.600 episode thô, tương ứng 800 cặp BF16/W4A4. `True` được hiểu là episode hoàn thành điều kiện success của LIBERO.

### 2.1. LIBERO OBJECT

Suite `object` gồm 10 task dưới đây. Mỗi task được rollout 20 episode cho BF16 và 20 episode cho W4A4; vì vậy một suite có 400 episode thô và 200 cặp episode để so sánh. Hai mode dùng cùng seed theo task và các initial state đánh giá được giữ tách khỏi calibration.

| # | LIBERO task | BF16 | W4A4 | Chênh lệch |
|---:|---|---:|---:|---:|
| 0 | pick up the alphabet soup and place it in the basket | 20/20 (100%) | 20/20 (100%) | +0.0 pp |
| 1 | pick up the cream cheese and place it in the basket | 20/20 (100%) | 20/20 (100%) | +0.0 pp |
| 2 | pick up the salad dressing and place it in the basket | 20/20 (100%) | 20/20 (100%) | +0.0 pp |
| 3 | pick up the bbq sauce and place it in the basket | 18/20 (90%) | 18/20 (90%) | +0.0 pp |
| 4 | pick up the ketchup and place it in the basket | 20/20 (100%) | 20/20 (100%) | +0.0 pp |
| 5 | pick up the tomato sauce and place it in the basket | 20/20 (100%) | 19/20 (95%) | -5.0 pp |
| 6 | pick up the butter and place it in the basket | 20/20 (100%) | 19/20 (95%) | -5.0 pp |
| 7 | pick up the milk and place it in the basket | 19/20 (95%) | 19/20 (95%) | +0.0 pp |
| 8 | pick up the chocolate pudding and place it in the basket | 20/20 (100%) | 19/20 (95%) | -5.0 pp |
| 9 | pick up the orange juice and place it in the basket | 20/20 (100%) | 20/20 (100%) | +0.0 pp |

**Kết quả suite:** BF16 đạt **197/200 (98.5%)**; W4A4 đạt **194/200 (97.0%)**; chênh lệch **-1.5 pp** và retention **98.48%**.

**Các case yếu:** Các điểm yếu đều nhỏ và cùng dạng pick-and-place: tomato sauce, butter và chocolate pudding mỗi task giảm từ 20/20 xuống 19/20. Task BBQ sauce giữ 18/20 ở cả hai mode, cho thấy phần khó ở đây không chỉ do lượng tử hóa.

**Đánh giá mức chấp nhận:** Ở mức chứng minh tính khả thi, kết quả **chấp nhận được**: W4A4 chỉ mất 3 success trên 200 episode, 7/10 task giữ nguyên và ba task còn lại chỉ giảm 1/20. Tuy nhiên đây vẫn là đánh giá mô phỏng, chưa phải tiêu chuẩn tương đương tuyệt đối cho triển khai robot thật.

### 2.2. LIBERO SPATIAL

Suite `spatial` gồm 10 task dưới đây. Mỗi task được rollout 20 episode cho BF16 và 20 episode cho W4A4; vì vậy một suite có 400 episode thô và 200 cặp episode để so sánh. Hai mode dùng cùng seed theo task và các initial state đánh giá được giữ tách khỏi calibration.

| # | LIBERO task | BF16 | W4A4 | Chênh lệch |
|---:|---|---:|---:|---:|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 19/20 (95%) | 20/20 (100%) | +5.0 pp |
| 1 | pick up the black bowl next to the ramekin and place it on the plate | 20/20 (100%) | 20/20 (100%) | +0.0 pp |
| 2 | pick up the black bowl from table center and place it on the plate | 19/20 (95%) | 20/20 (100%) | +5.0 pp |
| 3 | pick up the black bowl on the cookie box and place it on the plate | 20/20 (100%) | 20/20 (100%) | +0.0 pp |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 20/20 (100%) | 20/20 (100%) | +0.0 pp |
| 5 | pick up the black bowl on the ramekin and place it on the plate | 20/20 (100%) | 18/20 (90%) | -10.0 pp |
| 6 | pick up the black bowl next to the cookie box and place it on the plate | 20/20 (100%) | 20/20 (100%) | +0.0 pp |
| 7 | pick up the black bowl on the stove and place it on the plate | 20/20 (100%) | 19/20 (95%) | -5.0 pp |
| 8 | pick up the black bowl next to the plate and place it on the plate | 19/20 (95%) | 19/20 (95%) | +0.0 pp |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate | 20/20 (100%) | 19/20 (95%) | -5.0 pp |

**Kết quả suite:** BF16 đạt **197/200 (98.5%)**; W4A4 đạt **195/200 (97.5%)**; chênh lệch **-1.0 pp** và retention **98.98%**.

**Các case yếu:** Task 5 (black bowl on the ramekin) là điểm yếu rõ nhất, từ 20/20 xuống 18/20; task 7 và 9 giảm 20/20 xuống 19/20. Ngược lại task 0 và 2 tăng từ 19/20 lên 20/20, cho thấy tác động không đồng đều theo trạng thái khởi tạo.

**Đánh giá mức chấp nhận:** Kết quả **chấp nhận được cho nghiên cứu và thử nghiệm tiếp theo**: mức giảm toàn suite chỉ 1,0 pp, hai task còn tăng 1 success. Task 5 cần được đưa vào regression set vì giảm 2/20 dù baseline đạt 20/20.

### 2.3. LIBERO GOAL

Suite `goal` gồm 10 task dưới đây. Mỗi task được rollout 20 episode cho BF16 và 20 episode cho W4A4; vì vậy một suite có 400 episode thô và 200 cặp episode để so sánh. Hai mode dùng cùng seed theo task và các initial state đánh giá được giữ tách khỏi calibration.

| # | LIBERO task | BF16 | W4A4 | Chênh lệch |
|---:|---|---:|---:|---:|
| 0 | open the middle drawer of the cabinet | 20/20 (100%) | 19/20 (95%) | -5.0 pp |
| 1 | put the bowl on the stove | 19/20 (95%) | 19/20 (95%) | +0.0 pp |
| 2 | put the wine bottle on top of the cabinet | 15/20 (75%) | 18/20 (90%) | +15.0 pp |
| 3 | open the top drawer and put the bowl inside | 17/20 (85%) | 10/20 (50%) | -35.0 pp |
| 4 | put the bowl on top of the cabinet | 20/20 (100%) | 20/20 (100%) | +0.0 pp |
| 5 | push the plate to the front of the stove | 16/20 (80%) | 18/20 (90%) | +10.0 pp |
| 6 | put the cream cheese in the bowl | 20/20 (100%) | 16/20 (80%) | -20.0 pp |
| 7 | turn on the stove | 20/20 (100%) | 20/20 (100%) | +0.0 pp |
| 8 | put the bowl on the plate | 20/20 (100%) | 20/20 (100%) | +0.0 pp |
| 9 | put the wine bottle on the rack | 18/20 (90%) | 18/20 (90%) | +0.0 pp |

**Kết quả suite:** BF16 đạt **185/200 (92.5%)**; W4A4 đạt **178/200 (89.0%)**; chênh lệch **-3.5 pp** và retention **96.22%**.

**Các case yếu:** Task 3 (open the top drawer and put the bowl inside) giảm mạnh 17/20 xuống 10/20; task 6 (put the cream cheese in the bowl) giảm 20/20 xuống 16/20. Task 2 và 5 lại tăng lần lượt 3 và 2 success, nhưng không bù được regression tập trung ở task 3.

**Đánh giá mức chấp nhận:** Kết quả **khả thi nhưng chỉ chấp nhận có điều kiện**. Mức tổng 89,0% vẫn cao, song suy giảm tập trung mạnh ở task 3 và task 6. W4A4 chưa nên được xem là drop-in replacement cho BF16 trên suite này trước khi recalibration hoặc mixed-precision xử lý hai điểm yếu.

### 2.4. LIBERO LONG-HORIZON

Suite `long` gồm 10 task dưới đây. Mỗi task được rollout 20 episode cho BF16 và 20 episode cho W4A4; vì vậy một suite có 400 episode thô và 200 cặp episode để so sánh. Hai mode dùng cùng seed theo task và các initial state đánh giá được giữ tách khỏi calibration.

| # | LIBERO task | BF16 | W4A4 | Chênh lệch |
|---:|---|---:|---:|---:|
| 0 | LIVING ROOM SCENE2 put both the alphabet soup and the tomato sauce in the basket | 18/20 (90%) | 18/20 (90%) | +0.0 pp |
| 1 | LIVING ROOM SCENE2 put both the cream cheese box and the butter in the basket | 20/20 (100%) | 19/20 (95%) | -5.0 pp |
| 2 | KITCHEN SCENE3 turn on the stove and put the moka pot on it | 20/20 (100%) | 20/20 (100%) | +0.0 pp |
| 3 | KITCHEN SCENE4 put the black bowl in the bottom drawer of the cabinet and close it | 19/20 (95%) | 16/20 (80%) | -15.0 pp |
| 4 | LIVING ROOM SCENE5 put the white mug on the left plate and put the yellow and white mug on the right plate | 19/20 (95%) | 16/20 (80%) | -15.0 pp |
| 5 | STUDY SCENE1 pick up the book and place it in the back compartment of the caddy | 19/20 (95%) | 20/20 (100%) | +5.0 pp |
| 6 | LIVING ROOM SCENE6 put the white mug on the plate and put the chocolate pudding to the right of the plate | 19/20 (95%) | 17/20 (85%) | -10.0 pp |
| 7 | LIVING ROOM SCENE1 put both the alphabet soup and the cream cheese box in the basket | 19/20 (95%) | 19/20 (95%) | +0.0 pp |
| 8 | KITCHEN SCENE8 put both moka pots on the stove | 15/20 (75%) | 13/20 (65%) | -10.0 pp |
| 9 | KITCHEN SCENE6 put the yellow and white mug in the microwave and close it | 19/20 (95%) | 20/20 (100%) | +5.0 pp |

**Kết quả suite:** BF16 đạt **187/200 (93.5%)**; W4A4 đạt **178/200 (89.0%)**; chênh lệch **-4.5 pp** và retention **95.19%**.

**Các case yếu:** Task 3 và 4 cùng giảm 3 success; task 6 và 8 cùng giảm 2 success. Task 8 vốn đã khó với BF16 (15/20) và tiếp tục giảm còn 13/20. Hai task 5 và 9 tăng 1 success, nên suy giảm không xảy ra đồng nhất trên toàn suite.

**Đánh giá mức chấp nhận:** Kết quả **khả thi nhưng cần cải thiện trước triển khai nghiêm ngặt**. W4A4 giữ 95,19% số success của BF16, nhưng các nhiệm vụ nhiều bước giảm rõ hơn. Điều này phù hợp với giả thuyết rằng sai số lượng tử tích lũy qua chuỗi thao tác dài làm policy nhạy hơn.

## 3. Kết quả tổng thể

### 3.1. Success rate trên bốn suite

| Suite | BF16 | W4A4 | Chênh lệch | Retention |
|---|---:|---:|---:|---:|
| object | 197/200 (98.5%) | 194/200 (97.0%) | -1.5 pp | 98.48% |
| spatial | 197/200 (98.5%) | 195/200 (97.5%) | -1.0 pp | 98.98% |
| goal | 185/200 (92.5%) | 178/200 (89.0%) | -3.5 pp | 96.22% |
| long | 187/200 (93.5%) | 178/200 (89.0%) | -4.5 pp | 95.19% |
| **Tổng** | **766/800 (95.750%)** | **745/800 (93.125%)** | **-2.625 pp** | **97.26%** |

Tổng thể, BF16 đạt **766/800 = 95.750%**, còn W4A4 đạt **745/800 = 93.125%**. Mức suy giảm tuyệt đối là **-2.625 điểm phần trăm**, tương đương W4A4 giữ lại **97.26%** số success của BF16.

Trong 800 cặp episode có **46** trường hợp chỉ BF16 thành công, **25** trường hợp chỉ W4A4 thành công và **9** trường hợp cả hai cùng thất bại. McNemar exact trên toàn bộ cặp cho `p = 0.0170`; paired bootstrap theo episode trước đó cho khoảng 95% `[-4,625; -0,625] pp`, còn task-cluster bootstrap cho `[-5,375; -0,250] pp`. Vì vậy suy giảm tổng thể nhỏ nhưng có tính hệ thống trong protocol đã thử, không nên mô tả W4A4 là hoàn toàn tương đương BF16.

Ở góc độ feasibility, kết quả vẫn tích cực: tất cả suite đạt ít nhất 89,0%, retention theo suite nằm trong khoảng 95,19–98,98%, và một nửa số task không thay đổi success rate. Object/Spatial gần trần và ít nhạy hơn; Goal/Long giảm mạnh hơn, phù hợp với nhận định rằng nhiệm vụ có quan hệ mục tiêu phức tạp hoặc chuỗi hành động dài nhạy hơn với sai số lượng tử tích lũy. Điểm yếu cần ưu tiên nhất là Goal task 3, tiếp theo là Goal task 6 và Long task 3/4/6/8.

### 3.2. Tối ưu VRAM

| Suite | BF16 peak allocated | W4A4 peak allocated | Giảm tuyệt đối | Giảm tương đối |
|---|---:|---:|---:|---:|
| object | 5.971 GiB | 4.459 GiB | 1.512 GiB | 25.33% |
| spatial | 5.975 GiB | 4.460 GiB | 1.515 GiB | 25.36% |
| goal | 5.969 GiB | 4.458 GiB | 1.511 GiB | 25.32% |
| long | 5.972 GiB | 4.459 GiB | 1.513 GiB | 25.34% |

Mức giảm rất nhất quán giữa bốn suite: trung bình **1.513 GiB**, tương đương **25.34% peak allocated VRAM**. Peak reserved giảm xấp xỉ 23,36%. Đây là lợi ích thực đo của implementation fake-quant hiện tại trên L4 và là kết quả hệ thống rõ ràng nhất của work.

### 3.3. Dung lượng logic và ước lượng native

Phạm vi W4A4 chứa **1,736,441,856 trọng số** trong **304 Linear**. Phần trọng số mục tiêu nếu lưu BF16 chiếm khoảng **3.234 GiB**; raw INT4 tương ứng khoảng **0.809 GiB**, tức giảm 75%. Khi cộng scale và auxiliary tensors, logical W4A4 records chiếm khoảng **0.914 GiB**, tương ứng giảm **71.75%** hay nén **3.54×** so với BF16 của cùng 304 Linear.

Nếu có packed/native backend để thay trực tiếp phần BF16 mục tiêu bằng logical W4A4 records, hybrid model được ước lượng khoảng **30.933 GiB**, tức giảm khoảng **6.98%** so với toàn checkpoint BF16. Số lượng tham số toán học không đổi; phần giảm đến từ số bit biểu diễn trọng số.

### 3.4. Hạn chế của báo cáo

- Runtime hiện tại là fake-quant, chưa có packed INT4 và fused/native W4A4 kernel; do đó báo cáo chưa thể đánh giá   native latency, throughput, tensor-core utilization, energy efficiency hoặc concurrency thực tế.
- Các chỉ số hiệu năng native phải được triển khai và benchmark trên accelerator mục tiêu thực, vì kernel phụ thuộc   kiến trúc GPU; kết quả không thể suy ra chỉ từ fake-quant.
- Rollout mới thực hiện trong LIBERO simulation, chưa đánh giá robot thật, sensor noise, control-loop deadline hoặc sim-to-real.
- Protocol có 20 episode/task nhưng chưa có nhiều đợt lặp độc lập với nhiều evaluation seed base; các task có 20 mẫu   vẫn có khoảng bất định đáng kể.
- Phạm vi hiện tại chỉ gồm checkpoint LIBERO `libero_10`, bốn suite và 304 Linear mục tiêu; chưa khảo sát các   embodiment, checkpoint hoặc mixed-precision policy khác.
- Ước lượng dung lượng native là mô hình logic, chưa phải artifact native đã export và đo trực tiếp.

## 4. Kết luận work

Work đã xây dựng được một pipeline W4A4 hoàn chỉnh ở mức **algorithmic/fake-quant reference** cho GR00T N1.7: xác định đúng 304 Linear mục tiêu, thu thập calibration theo suite, dựng và audit pack, tích hợp inference server, rollout BF16/W4A4 ghép cặp có resume, kiểm tra 20 episode/task và tổng hợp kết quả task/suite với checksum cùng provenance.

Mục tiêu ban đầu về kiểm chứng tính khả thi đã hoàn thành. Trên 800 cặp episode, W4A4 đạt 93,125% so với 95,750% của BF16, giữ 97,26% số success và giảm trung bình 25,34% peak allocated VRAM. Phần 304 Linear có dung lượng logic giảm 71,75%; một native hybrid model được ước lượng giảm khoảng 6,98% cho toàn checkpoint.

Hạn chế còn lại tập trung ở hai hướng. Về chất lượng, Goal/Long và một số task cụ thể vẫn có regression rõ, nên cần recalibration, sensitivity analysis hoặc mixed-precision fallback. Về hệ thống, cần triển khai packed INT4 và native W4A4 kernel trên GPU mục tiêu trước khi kết luận về latency, throughput và lợi ích deployment thực tế.

Ý nghĩa chính của work là tạo bằng chứng thực nghiệm rằng W4A4 **khả thi về mặt chất lượng và có lợi về VRAM** trên GR00T N1.7, đồng thời cung cấp golden reference, task regression set và protocol tái lập để phát triển native backend mà không phải bắt đầu từ một giả thuyết chưa được kiểm chứng.

---

## Phụ lục: nguồn dữ liệu

- `gr00t-n17-holoq-phase2-object-results.zip` — suite `object`, schema 2, deep pack audit: True, record verified: 304/304.
- `gr00t-n17-holoq-phase2-spatial-results.zip` — suite `spatial`, schema 2, deep pack audit: True, record verified: 304/304.
- `gr00t-n17-holoq-phase2-goal-results.zip` — suite `goal`, schema 2, deep pack audit: True, record verified: 304/304.
- `gr00t-n17-holoq-phase2-long-results.zip` — suite `long`, schema 2, deep pack audit: True, record verified: 304/304.

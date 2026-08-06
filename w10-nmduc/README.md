# Thí nghiệm CKA pruning trên GR00T-N1.7 LIBERO

## Điều hướng báo cáo

| Suite | Báo cáo chi tiết | Nội dung chính |
|---|---|---|
| LIBERO Object | [Mở báo cáo Object](object/README.md) | Baseline, P25, P37.5, P50 và P75; trade-off tốt nhất hiện tại: P50 |
| LIBERO Spatial | [Mở báo cáo Spatial](spatial/README.md) | Baseline, P25, P37.5 và P50; trade-off tốt nhất hiện tại: P25 |
| LIBERO Goal | [Mở báo cáo Goal](goal/README.md) | Baseline, P25, P37.5 và P50; vùng nên nghiên cứu tiếp: P18.75–P25 |
| LIBERO Long / LIBERO-10 | [Mở báo cáo Long](long/README.md) | Baseline, P25, P37.5 và P50; vùng nên nghiên cứu tiếp: P12.5–P18.75 |

Mỗi báo cáo suite liên kết trực tiếp tới các biểu đồ, bảng CSV và `comparison.json` nằm trong chính thư mục đó.

## Mục tiêu

Thí nghiệm khảo sát ảnh hưởng của **CKA-guided layer pruning** lên GR00T-N1.7: mô hình có thể giảm số lớp, tham số, VRAM và thời gian suy luận đến mức nào trong khi vẫn duy trì khả năng thực hiện nhiệm vụ. Baseline không pruning được dùng làm mốc; các cấu hình còn lại chỉ thay đổi ngân sách pruning để quan sát trade-off giữa hiệu năng và độ chính xác.

## Cách thực thi

Pipeline chung cho mỗi LIBERO suite:

1. Lấy hidden states/activations của mô hình gốc trên tập calibration.
2. Tính **adjacent linear CKA** để đo mức tương đồng giữa các lớp liền kề.
3. Chọn và loại lớp theo CKA trong language backbone và Action-DiT; giữ nguyên 4 lớp VL self-attention (VLSA).
4. Recovery fine-tuning mô hình đã prune bằng LoRA.
5. Đánh giá offline và closed-loop: số tham số, latency, VRAM, MSE/MAE, CKA heatmap và success rate trên 10 task × 20 episode = 200 episode/suite.

Các mức `P25`, `P37`, `P50`, `P75` là tỷ lệ lớp bị loại trong **hai block được pruning**, không phải tỷ lệ giảm tham số của toàn mô hình:

| Phiên bản | Language | Action-DiT | VLSA | Giảm tham số toàn model (xấp xỉ) |
|---|---:|---:|---:|---:|
| Baseline | 16/16 lớp | 32/32 lớp | 4/4 lớp | 0% |
| P25 | 12/16 lớp | 24/32 lớp | 4/4 lớp | 14.9% |
| P37.5 (`P37`) | 10/16 lớp | 20/32 lớp | 4/4 lớp | 22.3–22.4% |
| P50 | 8/16 lớp | 16/32 lớp | 4/4 lớp | 29.8–29.9% |
| P75 | 4/16 lớp | 8/32 lớp | 4/4 lớp | 44.9% |

## Cấu hình rút gọn do giới hạn phần cứng

Thay vì recovery toàn phần quy mô lớn, thí nghiệm dùng cấu hình phù hợp GPU đơn 16–24 GB:

| Thành phần | Cấu hình thí nghiệm |
|---|---|
| Recovery | 3.000 optimizer steps thay cho lịch 20.000 bước quy mô đầy đủ |
| Batch | Global batch 1, gradient accumulation 8, effective batch 8 thay cho global batch 640 |
| Phần được huấn luyện | LoRA trên các lớp `Linear` của những Action-DiT block còn lại; timestep encoder và output projections |
| Phần bị đóng băng | Qwen3-VL language/vision backbone, VLLN và trọng số gốc của Action-DiT |
| LoRA | Rank 16, alpha 32, dropout 0.05; merge adapter khi xuất model cuối |
| Learning rate | `1e-4` |
| State dropout | `0.2` |
| Seed | `42` |
| GPU | Một GPU cho recovery/inference; activation checkpointing được bật |

Cấu hình này ưu tiên hoàn thành thí nghiệm nhất quán trong giới hạn Kaggle/Colab/Modal. Vì nhỏ hơn đáng kể so với recovery đầy đủ, kết quả thể hiện **xu hướng và ảnh hưởng của CKA pruning**, không nên được xem là giới hạn chất lượng tối đa của mô hình sau pruning.

## Các phiên bản đã kiểm thử

| LIBERO suite | Baseline | P25 | P37.5 | P50 | P75 |
|---|:---:|:---:|:---:|:---:|:---:|
| Object | ✓ | ✓ | ✓ | ✓ | ✓ |
| Spatial | ✓ | ✓ | ✓ | ✓ | — |
| Goal | ✓ | ✓ | ✓ | ✓ | — |
| Long / LIBERO-10 | ✓ | ✓ | ✓ | ✓ | — |

Object được mở rộng tới P75 để thăm dò pruning cực đoan. Ba suite còn lại dừng ở P50 vì nhiệm vụ nhạy hơn với việc mất lớp và chi phí rollout lớn hơn.

## Tiêu chí đọc kết quả

- **Hiệu quả tối ưu:** parameter reduction, latency mean/P95, peak VRAM và policy RPC latency.
- **Độ gần ground truth:** MSE và MAE của action prediction.
- **Chất lượng tác vụ:** success rate closed-loop; chênh lệch chỉ 1–2 episode trên 200 chưa đủ để kết luận một biến thể chính xác hơn rõ rệt.
- **Trade-off:** mức pruning phù hợp phải đồng thời giảm tài nguyên và giữ success rate, không chỉ đạt MSE/MAE thấp hoặc latency nhanh.

Kết quả chi tiết, bảng CSV và biểu đồ của từng suite nằm trong các thư mục [`object`](object), [`spatial`](spatial), [`goal`](goal) và [`long`](long).

## Lưu ý về provenance

Các thư mục này là bản tổng hợp từ compact reference, không chứa model weights, video rollout gốc hoặc toàn bộ heatmap nguồn. Goal/P25 (195/200) và Long/baseline (196/200) được lấy từ các lần đo trước nhưng raw logs/videos hiện không còn được lưu; các giá trị aggregate vẫn được dùng trong so sánh, song không thể truy vết lại từng episode từ archive hiện tại.

# Hướng dẫn train mô hình nhận diện 12 stage

Tài liệu này giải thích rõ lỗi `Assertion 't >= 0 && t < n_classes' failed`, trả lời câu hỏi về `self.w = nn.Linear(attn_dim, 1)`, và các bước cần làm để train đúng cho bài toán 12 stage.

## 1) Kết luận nhanh

- **Không** đổi `self.w = nn.Linear(attn_dim, 1)` thành `nn.Linear(attn_dim, 12)`.
- Lỗi assert xuất phát từ **mismatch giữa nhãn và `num_classes`** trong `CrossEntropyLoss`.
- Với dữ liệu hiện tại (`labels.csv` có nhãn từ **0 đến 11**) thì cần `num_classes = 12`.

## 2) Vì sao `self.w` phải là 1?

Trong `AttentionMIL`:

- `self.w` tạo **attention score cho từng patch** trong bag.
- Attention score là **1 scalar / patch**, sau đó softmax qua các patch.
- Do đó `self.w` đúng phải là:

- `nn.Linear(attn_dim, 1)`

Nếu đổi thành `nn.Linear(attn_dim, 12)`:

- attention sẽ thành vector 12 chiều/patch (sai ngữ nghĩa attention hiện tại),
- shape downstream không còn tương thích,
- không giải quyết tận gốc lỗi nhãn out-of-range.

## 3) Nguồn gốc lỗi assert

`CrossEntropyLoss` yêu cầu label thuộc đoạn:

$$0 \le y < C$$

trong đó $C = \text{num\_classes}$.

Nếu `num_classes=2` nhưng label có giá trị 8, 11,... thì sẽ gây assert:

`Assertion 't >= 0 && t < n_classes' failed`

## 4) Những gì đã làm trong code

### A. Cập nhật `configs/default.yaml`

- `training.num_classes` đã đổi từ `2` -> `12`.

### B. Cập nhật `src/train.py`

Đã thêm validation trước khi train:

1. Đọc toàn bộ label hợp lệ từ `labels.csv`.
2. Tính `label_min`, `label_max`, `inferred_num_classes = label_max + 1`.
3. Báo lỗi rõ ràng nếu `label_max >= num_classes`.
4. In thống kê label để debug nhanh.

Mục tiêu: tránh lỗi CUDA assert mơ hồ, thay bằng lỗi dễ hiểu ngay đầu pipeline.

## 5) Checklist để nhận diện đúng 12 stage

1. **Label mapping thống nhất**
   - Stage phải được map thành số nguyên liên tục.
   - Nếu stage là 0..11 => `num_classes=12`.
   - Nếu stage là 1..12 => cần remap về 0..11 trước khi train.

2. **Đồng bộ dữ liệu đặc trưng và nhãn**
   - Mỗi `slide_id` trong `labels.csv` cần có file `.h5` tương ứng trong `data/h5_features`.
   - Tránh mismatch do tên file khác chuẩn.

3. **Phân phối lớp (class imbalance)**
   - Kiểm tra số lượng mỗi stage.
   - Nếu lệch mạnh, cân nhắc weighted loss / sampler.

4. **Metrics phù hợp multi-class**
   - Dùng `macro-F1`, `weighted-F1`, confusion matrix.
   - Không chỉ nhìn accuracy tổng.

5. **Đánh giá theo split chuẩn**
   - Chia train/val/test theo slide-level.
   - Tránh leakage giữa split.

## 6) Cần làm tiếp (khuyến nghị)

- Thêm script kiểm tra nhãn + `.h5` trước train (data sanity check).
- Bổ sung `class_weight` cho `CrossEntropyLoss` nếu dataset mất cân bằng.
- Nâng `evaluate.py` để in `classification_report` và confusion matrix cho đủ 12 stage.

## 7) Lệnh chạy khuyến nghị

Chạy bằng đúng Python trong `.venv`:

```bash
cd /home/khoa/datamining_project/wsi_processing
source ../.venv/bin/activate
python -m src.train --config configs/default.yaml
```

Nếu bạn vẫn dùng `python3`, hãy kiểm tra trước nó có trỏ vào `.venv` hay không.

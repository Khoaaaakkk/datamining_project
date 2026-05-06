# WSI Processing – File & Flow Explanation

Tài liệu này giải thích chi tiết **mục đích**, **input**, **output** của từng file trong `wsi_processing`, và cách các flow ghép lại thành pipeline hoàn chỉnh.

---

## 1) Big picture

Pipeline của project gồm 3 luồng chính:

1. **Preprocessing + Feature Extraction**
   - `raw_wsi` ➜ tissue mask ➜ lọc patch ➜ feature vectors ➜ `.h5`
2. **MIL Training**
   - `.h5` + `labels.csv` ➜ train Attention MIL ➜ checkpoint
3. **Evaluation**
   - checkpoint + `.h5` + labels ➜ Accuracy/F1

Ngoài ra có flow phụ cho **notebook exploration** để test nhanh từng thành phần.

---

## 2) Folder-by-folder

## `configs/`

### `configs/default.yaml`
- **Mục đích**: cấu hình tập trung cho toàn bộ pipeline.
- **Input**: người dùng chỉnh path + hyperparameters.
- **Output**: các script đọc file này để chạy nhất quán.
- **Nhóm tham số chính**:
  - `data`: đường dẫn dữ liệu/nhãn.
  - `preprocessing`: patch size, ngưỡng tissue/quality, stain norm.
  - `feature_extraction`: batch size, device, pretrained.
  - `training`: tham số MIL.

---

## `src/preprocessing/`

### `src/preprocessing/segment.py`
- **Mục đích**: tách vùng mô bằng HSV saturation + Otsu trên thumbnail.
- **Input**:
  - `openslide.OpenSlide`
  - `level` segmentation.
- **Output**:
  - `mask` nhị phân (`uint8`, 0/255).
  - `downsample` để map tọa độ level-0 sang level mask.
- **Vai trò trong flow**: bước cổng vào, giảm mạnh số patch cần xử lý.

### `src/preprocessing/quality.py`
- **Mục đích**: loại patch kém chất lượng (quá trắng, quá mờ).
- **Input**: patch RGB dạng numpy array.
- **Output**: boolean `is_valid_patch`.
- **Metrics chính**:
  - white ratio,
  - Laplacian variance (blur),
  - độ lệch chuẩn ảnh.

### `src/preprocessing/stain_norm.py`
- **Mục đích**: chuẩn hóa màu (Macenko) để giảm domain shift giữa slide.
- **Input**:
  - patch RGB,
  - (tùy chọn) ảnh reference H&E.
- **Output**: patch RGB sau normalize.
- **Lưu ý**: có fallback an toàn, nếu lỗi thì trả patch gốc.

### `src/preprocessing/extract_feature.py`
- **Mục đích**: script orchestration end-to-end cho preprocessing + extract feature.
- **Input**:
  - config YAML,
  - 1 slide (`--slide`) hoặc toàn bộ `raw_wsi_dir`.
- **Output**:
  - `data/h5_features/<slide_id>.h5` chứa:
    - `features`: `[N, 2048]`
    - `coords`: `[N, 2]`
  - `data/masks/<slide_id>.png`
  - report JSON ở `experiments/logs/`.
- **Flow nội bộ**:
  1) segmentation,
  2) sinh candidate coords,
  3) quality filtering,
  4) (optional) stain norm,
  5) batch inference ResNet50,
  6) ghi HDF5 + metadata.

---

## `src/data_loader/`

### `src/data_loader/wsi_dataset.py`
- **Mục đích**: lazy dataset đọc patch trực tiếp từ file WSI cho DataLoader.
- **Input**: đường dẫn slide + danh sách tọa độ + patch size.
- **Output**: `(patch_tensor, coord_tensor)`.
- **Điểm quan trọng**: mở slide lazy trong worker để tương thích multiprocessing.

### `src/data_loader/h5_dataset.py`
- **Mục đích**: load bag-level dữ liệu từ `.h5` để train/eval MIL.
- **Input**:
  - thư mục `.h5`,
  - `labels.csv` (`slide_id,label`).
- **Output**: dict gồm `slide_id`, `features`, `coords`, `label`.
- **Vai trò**: cầu nối từ preprocessing sang training/evaluation.

---

## `src/models/`

### `src/models/feature_extractor.py`
- **Mục đích**: khởi tạo backbone trích đặc trưng (ResNet50 bỏ lớp FC).
- **Input**: `pretrained` flag, `device`.
- **Output**: model xuất vector feature chiều 2048.

### `src/models/mil_classifier.py`
- **Mục đích**: mô hình Attention MIL cho dự đoán slide-level.
- **Input**: bag features `[N, D]` hoặc `[B, N, D]`.
- **Output**: logits và attention weights.
- **Ý nghĩa**: attention học patch quan trọng thay vì average toàn bộ patch.

---

## `src/`

### `src/train.py`
- **Mục đích**: huấn luyện Attention MIL.
- **Input**:
  - features `.h5`,
  - labels CSV,
  - cấu hình training.
- **Output**:
  - checkpoint tốt nhất ở `experiments/checkpoints/best_mil.pth`.

### `src/evaluate.py`
- **Mục đích**: đánh giá checkpoint MIL trên dataset label.
- **Input**: checkpoint + `.h5` + labels.
- **Output**: Accuracy, F1-weighted.

---

## `src/utils/`

### `src/utils/file_utils.py`
- **Mục đích**: helper xử lý path/thư mục và tạo `slide_id`.
- **Input**: path string/Path.
- **Output**: thư mục được tạo, danh sách file, stem chuẩn.

### `src/utils/viz_utils.py`
- **Mục đích**: hỗ trợ visualize mask overlay và grid patch.
- **Input**: RGB image/mask/list patch.
- **Output**: ảnh blend hoặc figure trực quan.

---

## `notebooks/`

### `notebooks/01_explore_slide.ipynb`
- **Mục đích**: thử nhanh pipeline trên 1 slide, validate logic xử lý patch.
- **Input**: một file `.svs`.
- **Output**: `features.npy`, `coords.npy` (demo/prototype).

### `notebooks/02_test_dataloader.ipynb`
- **Mục đích**: kiểm tra `WSIPatchDataset` và DataLoader behavior.
- **Input**: slide path + coords list.
- **Output**: xác nhận patch loading hoạt động đúng.

---

## `tests/`

### `tests/test_file_utils.py`
- Test tạo thư mục/parent và chuẩn hóa stem file.

### `tests/test_quality.py`
- Test `is_valid_patch` với patch giả lập (valid vs white patch).

### `tests/test_mil_classifier.py`
- Smoke test forward pass của `AttentionMIL`.

### `tests/test_h5_dataset.py`
- Test đọc `.h5` + labels CSV bằng dữ liệu temporary.

---

## 3) End-to-end flow (step-by-step)

1. **Đưa slide vào** `data/raw_wsi/`.
2. Chạy `extract_feature.py`:
   - sinh `mask` và `h5` cho từng slide.
3. Chuẩn bị `data/labels/labels.csv` với cột `slide_id,label`.
4. Chạy `train.py` để huấn luyện MIL.
5. Chạy `evaluate.py` để tính metric.

---

## 4) Input/Output contract ngắn gọn

- **Input dữ liệu gốc**: `.svs/.tif/.ndpi`
- **Input nhãn**: CSV (`slide_id,label`)
- **Output preprocessing**: `.h5` (`features`, `coords`) + mask `.png`
- **Output training**: `.pth`
- **Output evaluation**: metrics text (Accuracy/F1)

---

## 5) Ghi chú thực thi

- Với dữ liệu lớn, hiệu năng phụ thuộc `batch_size`, `num_workers`, VRAM.
- Bật stain normalization chỉ khi cần domain harmonization, vì tăng thời gian xử lý.
- Khi train MIL, các slide không có nhãn hợp lệ (`label < 0`) sẽ bị bỏ qua.

# WSI Processing Pipeline

Pipeline này xử lý ảnh Whole Slide Image (WSI) theo đúng flow trong `tree.txt` và notebook:

1. Tissue Segmentation
2. Patch Quality Control
3. (Tùy chọn) Stain Normalization
4. Feature Extraction (ResNet50)
5. Lưu `.h5` để train MIL

## Cấu trúc flow chính

- Input: file WSI (`.svs/.tif/.ndpi`) trong `data/raw_wsi/`
- Trung gian:
  - tissue mask (`.png`) trong `data/masks/`
  - báo cáo JSON trong `experiments/logs/`
- Output chính: file đặc trưng `.h5` trong `data/h5_features/`
  - dataset `features`: shape `[N, 2048]`
  - dataset `coords`: shape `[N, 2]` tọa độ patch level-0

## Mục đích / input / output từng file

### `src/preprocessing/segment.py`
- **Mục đích**: tách vùng mô khỏi nền bằng HSV + Otsu trên ảnh thumbnail.
- **Input**:
  - `openslide.OpenSlide`
  - `level` segmentation
- **Output**:
  - `mask` nhị phân (0/255)
  - `downsample` để ánh xạ tọa độ level-0 sang level mask

### `src/preprocessing/quality.py`
- **Mục đích**: lọc patch rỗng/trắng/mờ.
- **Input**: patch RGB numpy array.
- **Output**: `bool` patch hợp lệ.

### `src/preprocessing/stain_norm.py`
- **Mục đích**: chuẩn hóa màu theo Macenko (nếu bật).
- **Input**:
  - ảnh tham chiếu H&E (tùy chọn)
  - patch RGB
- **Output**: patch RGB sau chuẩn hóa (fallback về patch gốc nếu lỗi).

### `src/data_loader/wsi_dataset.py`
- **Mục đích**: lazy-load patch trực tiếp từ WSI theo danh sách tọa độ.
- **Input**: đường dẫn slide, danh sách `(x,y)`, patch size, transform.
- **Output**: `(patch, coord_tensor)` cho DataLoader.

### `src/models/feature_extractor.py`
- **Mục đích**: khởi tạo backbone ResNet50 bỏ lớp FC cuối.
- **Input**: flag pretrained, device.
- **Output**: model xuất feature dimension 2048.

### `src/preprocessing/extract_feature.py`
- **Mục đích**: orchestration end-to-end cho preprocessing + feature extraction.
- **Input**:
  - `configs/default.yaml`
  - một slide (`--slide`) hoặc toàn bộ thư mục `raw_wsi`
- **Output**:
  - file `.h5` cho mỗi slide ở `data/h5_features`
  - mask `.png` ở `data/masks`
  - report `.json` ở `experiments/logs`

### `src/data_loader/h5_dataset.py`
- **Mục đích**: load bag-level features từ `.h5` cho MIL.
- **Input**: thư mục `.h5`, file nhãn CSV (`slide_id,label`).
- **Output**: dict chứa `features`, `coords`, `label`, `slide_id`.

### `src/models/mil_classifier.py`
- **Mục đích**: Attention MIL classifier cho phân loại slide-level.
- **Input**: bag features `[N, D]`.
- **Output**: logits + attention weights.

### `src/train.py`
- **Mục đích**: train Attention MIL từ file `.h5`.
- **Input**:
  - features trong `data/h5_features`
  - labels CSV trong `data/labels`
- **Output**:
  - checkpoint tốt nhất `experiments/checkpoints/best_mil.pth`

### `src/evaluate.py`
- **Mục đích**: evaluate checkpoint MIL.
- **Input**: checkpoint + dataset `.h5` + labels.
- **Output**: Accuracy, F1-weighted.

### `src/utils/file_utils.py`
- **Mục đích**: helper tạo thư mục, liệt kê file, tạo slide_id.
- **Input/Output**: path utilities.

### `src/utils/viz_utils.py`
- **Mục đích**: helper visualize overlay mask và grid patch.
- **Input**: RGB image / mask / list patch.
- **Output**: ảnh overlay hoặc figure trực quan.

### `configs/default.yaml`
- **Mục đích**: cấu hình tập trung cho data path, preprocessing, extraction, training.
- **Input**: được đọc bởi `extract_feature.py`, `train.py`, `evaluate.py`.
- **Output**: tham số runtime.

## Flow tổng thể trong `wsi_processing`

- **Flow A – Data Preparation**:
  - `raw_wsi` -> `extract_feature.py` -> `masks` + `h5_features`
- **Flow B – Training**:
  - `h5_features` + `labels` -> `train.py` -> `checkpoints`
- **Flow C – Evaluation**:
  - `h5_features` + `labels` + checkpoint -> `evaluate.py` -> metrics
- **Flow D – Notebook Exploration**:
  - `notebooks/01_explore_slide.ipynb`: thử OpenSlide + pipeline từng bước
  - `notebooks/02_test_dataloader.ipynb`: test `WSIPatchDataset`

## Cách chạy nhanh

```bash
python3 -m src.preprocessing.extract_feature --config configs/default.yaml
python3 -m src.train --config configs/default.yaml
python3 -m src.evaluate --config configs/default.yaml --checkpoint experiments/checkpoints/best_mil.pth
```

## Ghi chú

- Nếu dùng stain normalization cần cài đủ `staintools` và dependency liên quan.
- Với dữ liệu lớn, tăng `batch_size` và `num_workers` theo RAM/VRAM máy.

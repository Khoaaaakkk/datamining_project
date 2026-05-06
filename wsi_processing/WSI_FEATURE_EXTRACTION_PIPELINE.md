# WSI Feature Extraction Pipeline (Download `.svs` -> Extract `.h5` -> Cleanup)

Tài liệu này mô tả pipeline mới theo đúng yêu cầu:

1. Tải file ảnh WSI `.svs` theo **n submitter_id**.
2. Xử lý ảnh sang file đặc trưng `.h5`.
3. Xóa file `.svs` đã xử lý để giải phóng dung lượng.

---

## 1) Trình tự làm việc

Pipeline chính nằm ở `src/wsi_feature_extraction.py`:

1. Đọc config từ `configs/default.yaml`.
2. Resolve danh sách `submitter_id` cần chạy (whitelist hoặc CSV) qua `SubmitterBatchController`.
3. Chia submitter theo batch (`batch_size`) và lặp từng vòng.
4. Với mỗi batch:
   - gọi `GDCSVSDownloader` để query GDC + tạo manifest + download `.svs`.
   - chạy `WSIFeaturePipeline` để sinh `.h5` (và mask).
   - xóa `.svs` nếu `delete_processed_svs=true`.
5. Ghi report JSON vào `gdc_download.report_path`.

---

## 2) Mối quan hệ giữa các file

### `src/preprocessing/gdc_download_and_process.py`
- **Vai trò mới**: chỉ làm nhiệm vụ **download `.svs`** từ GDC theo submitter IDs được truyền vào.
- Không xử lý ảnh, không trích xuất feature.
- API chính:
  - `download_for_submitter_ids(submitter_ids, batch_index)` -> trả về list đường dẫn `.svs` đã tải.

### `src/preprocessing/submitter_batch_controller.py`
- Module điều phối submitter IDs và số lượng mỗi vòng lặp.
- Resolve IDs theo ưu tiên:
  1) `gdc_download.submitter_id_whitelist`
  2) `gdc_download.clinical_csv` + `submitter_id_column`
- Cắt theo `max_patients` và chia batch theo `batch_size`.

### `src/preprocessing/extract_feature.py`
- Module xử lý WSI đã có sẵn:
  - tissue segmentation,
  - quality filtering,
  - feature extraction,
  - lưu `.h5`.
- Được gọi từ `src/wsi_feature_extraction.py`.

### `src/wsi_feature_extraction.py`
- **Entry point chính của pipeline mới**.
- Điều phối toàn bộ luồng: download -> process -> cleanup.

### `src/train.py`
- Được đổi vai trò thành wrapper tương thích.
- Khi chạy, file này sẽ gọi sang `src/wsi_feature_extraction.py`.

### `configs/default.yaml`
- Chứa toàn bộ thông số đường dẫn và hành vi pipeline.
- Các key quan trọng:
  - `gdc_download.submitter_id_whitelist`
  - `gdc_download.batch_size`
  - `gdc_download.max_patients`
  - `gdc_download.delete_processed_svs`
  - `data.raw_wsi_dir`, `data.h5_features_dir`.

---

## 3) Mục đích từng file (ngắn gọn)

- `gdc_download_and_process.py`: tải `.svs` theo submitter.
- `submitter_batch_controller.py`: chọn + chia submitter thành từng vòng lặp.
- `wsi_feature_extraction.py`: chạy pipeline end-to-end.
- `extract_feature.py`: xử lý `.svs` thành `.h5`.
- `default.yaml`: cấu hình runtime.

---

## 4) Cấu hình điển hình

Trong `configs/default.yaml`:

- Dùng whitelist submitter cụ thể:
  - `submitter_id_whitelist: ['TCGA-XX-XXXX', 'TCGA-YY-YYYY']`
- Giới hạn tải mỗi vòng:
  - `batch_size: 2`
- Giới hạn tổng số submitter:
  - `max_patients: 4`
- Xóa `.svs` sau xử lý:
  - `delete_processed_svs: true`

---

## 5) Cách vận hành pipeline bằng terminal

Từ thư mục `wsi_processing`:

```bash
cd /home/khoa/datamining_project/wsi_processing
source ../.venv/bin/activate
python -m src.wsi_feature_extraction --config configs/default.yaml
```

Bạn cũng có thể chạy qua wrapper:

```bash
python -m src.train --config configs/default.yaml
```

> Khuyến nghị: dùng trực tiếp `src.wsi_feature_extraction` để rõ nghĩa hơn.

---

## 6) Điều kiện để chạy thành công

1. `gdc-client` có tại đường dẫn cấu hình `gdc_download.client_path`.
2. Quyền thực thi cho file `gdc-client`.
3. Có mạng để query/tải từ GDC API.
4. Các dependency cần thiết đã được cài trong môi trường Python.

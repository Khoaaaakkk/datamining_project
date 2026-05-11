# WSI Preprocessing v2

Pipeline trích xuất đặc trưng từ WSI theo hướng Multimodal Multi-Instance Evidence Fusion (MMEF).

## Quick start

- Cập nhật cấu hình tại `configs/default.yaml` (đường dẫn dữ liệu, patch size, model).
- Chạy pipeline:
  - `python src/wsi_feature_extraction.py --config configs/default.yaml`
  - Hoặc chạy một slide cụ thể: `python src/wsi_feature_extraction.py --slide path/to/slide.svs`

- Chạy pipeline theo 1 batch (download → xử lý → xóa `.svs` theo manifest):
  - `python src/data_loader/batch_pipeline.py --manifest data/reference/manifests/batch_0004.txt --config configs/default.yaml`
  - Giữ file `.svs` sau xử lý: thêm `--keep`

## Outputs

- `data/h5_features/*.h5`: patch embeddings + tọa độ.
- `data/masks/*.png`: mask mô/tissue.
- `experiments/logs/preprocessing_report.json`: report tổng hợp.

Chi tiết dataflow và liên kết code: xem `PIPELINE_EXPLAINED.md`.

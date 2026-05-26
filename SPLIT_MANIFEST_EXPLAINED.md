# split_manifest.py – Cách hoạt động

File: `wsi_processing/src/data_loader/split_manifest.py`

## Mục tiêu

Tạo các file `batch_XXXX.txt` (manifest) từ danh sách `submitter_id` trong `data/reference/Final_Matched_Clinical.csv` để dùng với `gdc-client`.

## Luồng xử lý

1. **Đọc CSV**

- Đọc cột `submitter_id` từ `Final_Matched_Clinical.csv`.
- Loại bỏ giá trị rỗng và chuẩn hóa thành chuỗi.

1. **Chia batch**

- Chia danh sách `submitter_id` theo `--batch-size` (mặc định 5).
- Batch được đánh số từ `--start-index` (mặc định 0001).

1. **Gọi GDC API**

- Mỗi batch `submitter_id` được truy vấn qua endpoint: `https://api.gdc.cancer.gov/files`.
- Bộ lọc gồm:
  - `cases.submitter_id` ∈ danh sách batch
  - `data_category` = `Biospecimen`
  - `data_type` = `Slide Image`
  - `data_format` = `SVS`

- Lấy các trường: `file_id`, `file_name`, `md5sum`, `file_size`, `state`.

1. **Ghi manifest**

- Mỗi batch tạo file `batch_XXXX.txt` trong `data/reference/manifests`.
- Định dạng TSV chuẩn của `gdc-client`:
  - Header: `id\tfilename\tmd5\tsize\tstate`
  - Mỗi dòng: một WSI file từ GDC.

## Cách chạy

```powershell
C:/Users/Administrator/Documents/Code/datamining_project/.venv/Scripts/python.exe `
  wsi_processing/src/data_loader/split_manifest.py `
  --csv data/reference/Final_Matched_Clinical.csv `
  --out-dir data/reference/manifests `
  --batch-size 5 `
  --max-batches 5
```

## Tuỳ chọn CLI

- `--batch-size`: số `submitter_id` mỗi batch.
- `--start-index`: số thứ tự batch bắt đầu.
- `--max-batches`: giới hạn số batch tạo ra.
- `--max-files`: giới hạn số file trong mỗi manifest (debug).

## Lưu ý

- File tạo ra dùng trực tiếp với:

  ```powershell
  tools/gdc-client.exe download -m data/reference/manifests/batch_0001.txt -d data/raw_wsi
  ```

- Nếu không tìm thấy file từ GDC, manifest vẫn được tạo nhưng có thể rỗng.

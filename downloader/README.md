# GDC Downloader

Script tải dữ liệu từ GDC bằng `tools/gdc-client.exe` và file manifest.

## Mặc định

- Manifest: `data/reference/manifests/batch_0001.txt`
- Output: `data/raw_wsi`
- gdc-client: `tools/gdc-client.exe`

## Cách chạy (PowerShell)

```powershell
.\.venv\Scripts\python.exe downloader\download_gdc_batch.py
```

## Tuỳ chọn

```powershell
.\.venv\Scripts\python.exe downloader\download_gdc_batch.py --manifest data\reference\manifests\batch_0001.txt --output-dir data\raw_wsi
```

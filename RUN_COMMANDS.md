# WSI Preprocessing v2 – Run Commands

Các lệnh dưới đây dùng PowerShell và Python trong `.venv` của repo.

## 1) Tạo batch manifest từ CSV

Tách `submitter_id` trong `data/reference/Final_Matched_Clinical.csv` thành các file `batch_0001.txt`, `batch_0002.txt`, ... (mỗi batch 5 ID).

```powershell
C:/Users/Administrator/Documents/Code/datamining_project/.venv/Scripts/python.exe `
  wsi_processing/src/data_loader/split_manifest.py `
  --csv data/reference/Final_Matched_Clinical.csv `
  --out-dir data/reference/manifests `
  --batch-size 5
```

Ghi chú: cần `pandas` trong `.venv`.

## 2) Download dữ liệu WSI từ GDC (manifest)

Tải tất cả manifest batch trong `data/reference/manifests` về `data/raw_wsi`.

```powershell
C:/Users/Administrator/Documents/Code/datamining_project/.venv/Scripts/python.exe `
  wsi_processing/src/data_loader/wsi_downloader.py `
  --manifests-dir data/reference/manifests `
  --out-dir data/raw_wsi `
  --gdc-client tools/gdc-client.exe
```

Chạy thử (không download thật):

```powershell
C:/Users/Administrator/Documents/Code/datamining_project/.venv/Scripts/python.exe `
  wsi_processing/src/data_loader/wsi_downloader.py `
  --manifests-dir data/reference/manifests `
  --out-dir data/raw_wsi `
  --gdc-client tools/gdc-client.exe `
  --dry-run
```

## 3) Batch pipeline (download → xử lý → xóa .svs)

```powershell
C:/Users/Administrator/Documents/Code/datamining_project/.venv/Scripts/python.exe `
  wsi_processing/src/data_loader/batch_pipeline.py `
  --manifest data/reference/manifests/batch_0004.txt `
  --config wsi_processing/configs/default.yaml
```

### 3.1) Chạy nhiều batch liên tiếp

```powershell
C:/Users/Administrator/Documents/Code/datamining_project/.venv/Scripts/python.exe `
  wsi_processing/src/data_loader/run_batches.py `
  --manifests-dir data/reference/manifests `
  --start 1 `
  --end 5 `
  --config wsi_preprocessing_v2/configs/default.yaml
```

Giữ file `.svs` sau xử lý:

```powershell
C:/Users/Administrator/Documents/Code/datamining_project/.venv/Scripts/python.exe `
  wsi_processing/src/data_loader/batch_pipeline.py `
  --manifest data/reference/manifests/batch_0004.txt `
  --keep
```

## 4) Chạy pipeline trích xuất feature (v2)

Ví dụ chạy 1 slide với `wsi_processing` và config mặc định.

```powershell
# (khuyến nghị) set OpenSlide DLL dir trên Windows
$env:OPENSLIDE_DLL_DIR = "C:\Users\Administrator\Documents\Code\datamining_project\tools\openslide\bin\openslide-bin-4.0.0.13-windows-x64\bin"

C:/Users/Administrator/Documents/Code/datamining_project/.venv/Scripts/python.exe `
  wsi_processing/src/wsi_feature_extraction.py `
  --config wsi_processing/configs/default.yaml `
  --slide "C:\Users\Administrator\Documents\Code\datamining_project\data\raw_wsi\<your_slide>.svs"
```

## 5) Chạy pipeline trích xuất feature (batch)

Chạy mọi file `.svs` trong `data/raw_wsi`:

```powershell
$env:OPENSLIDE_DLL_DIR = "C:\Users\Administrator\Documents\Code\datamining_project\tools\openslide\bin\openslide-bin-4.0.0.13-windows-x64\bin"

Get-ChildItem -Path '.\data\raw_wsi' -Filter '*.svs' -Recurse | ForEach-Object {
  $slidePath = $_.FullName
  C:/Users/Administrator/Documents/Code/datamining_project/.venv/Scripts/python.exe `
    wsi_processing/src/wsi_feature_extraction.py `
    --config wsi_processing/configs/default.yaml `
    --slide $slidePath
}
```

## 6) Kiểm tra môi trường nhanh

```powershell
C:/Users/Administrator/Documents/Code/datamining_project/.venv/Scripts/python.exe -c "import sys; print(sys.executable)"
C:/Users/Administrator/Documents/Code/datamining_project/.venv/Scripts/python.exe -c "import openslide; print('openslide OK')"
```

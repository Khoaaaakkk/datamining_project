# WSI Preprocessing v2 — Pipeline Explanation

## Tổng quan dataflow (theo hình MMEF)

1. **WSI (Whole Slide Image)** → đọc ở level gốc và level downsample.
2. **Tissue Segmentation** → tạo mask mô ở level thấp (HSV + Otsu).
3. **Patch Sampling** → chọn các toạ độ patch ở level 0 dựa trên mask.
4. **Patch Quality Filter** → loại patch trắng/nhiễu/blur.
5. **Stain Normalization (optional)** → chuẩn hoá màu theo Macenko.
6. **Swin-L Embedding** → trích xuất embedding cho từng patch.
7. **Lưu HDF5** → lưu `features` + `coords` để bước MIL/Survival downstream.

## Mối liên kết giữa các file

- `src/wsi_feature_extraction.py` là entrypoint. Đọc config, duyệt danh sách slide, gọi pipeline.
- `src/preprocessing/embedding_pipeline.py` giữ logic chính (segment → sample → filter → embed → save).
- `src/preprocessing/tissue_segmentation.py` tạo tissue mask ở level thấp.
- `src/preprocessing/patch_sampler.py` chuyển mask + downsample thành danh sách tọa độ patch.
- `src/preprocessing/patch_quality.py` lọc patch bằng tỷ lệ trắng + blur + std.
- `src/preprocessing/stain_normalization.py` Macenko normalization (tuỳ chọn).
- `src/data_loader/wsi_patch_dataset.py` đọc patch theo coords (lazy open slide).
- `src/models/feature_extractor.py` tạo Swin-L backbone từ timm.
- `src/utils/file_utils.py` utilities quản lý file/dir.

## Chi tiết pipeline theo module

### 1) Entry script

`wsi_feature_extraction.py`:

- Đọc `configs/default.yaml`.
- Xác định thư mục dữ liệu (`raw_wsi_dir`, `h5_features_dir`, `masks_dir`).
- Cho phép chạy 1 slide hoặc toàn bộ thư mục.
- Gọi `WSIEmbeddingPipeline.process_slide()`.

### 2) Tissue segmentation

`tissue_segmentation.segment_tissue_hsv_otsu()`:

- Tạo thumbnail ở level `seg_level`.
- Chuyển HSV và dùng Otsu threshold trên kênh saturation.
- Morphology open/close làm mượt mask.

### 3) Patch sampling

`patch_sampler.generate_patch_coords()`:

- Duyệt theo lưới `patch_size` và `step_size` ở level 0.
- Map tọa độ về mask level bằng `downsample`.
- Chỉ giữ patch có tỷ lệ mô >= `tissue_threshold`.

### 4) Patch quality filter

`patch_quality.is_valid_patch()`:

- Lọc patch quá trắng (`max_white_ratio`).
- Lọc patch blur qua Laplacian variance (`min_laplacian_var`).
- Lọc patch ít texture (`min_std`).

### 5) Embedding extraction (Swin-L)

`feature_extractor.build_feature_extractor()`:

- Dùng timm model `swin_large_patch4_window7_224`.
- Set `num_classes=0` để lấy embedding.

`embedding_pipeline.extract_embeddings_batch()`:

- Tạo `DataLoader` từ `WSIPatchDataset`.
- (Optional) stain normalization.
- Chuẩn hoá theo ImageNet và chạy model.
- Trả về `features` (N x D) và `coords` (N x 2).

### 6) Output

`embedding_pipeline.save_h5()`:

- Lưu HDF5 với datasets: `features`, `coords`.
- Ghi metadata: `patch_size`, `step_size`, `backbone`, `n_patches`, `feature_dim`.

## Cấu trúc thư mục (v2)

```text
wsi_preprocessing_v2/
├── configs/default.yaml
├── src/
│   ├── preprocessing/
│   │   ├── tissue_segmentation.py
│   │   ├── patch_sampler.py
│   │   ├── patch_quality.py
│   │   ├── stain_normalization.py
│   │   └── embedding_pipeline.py
│   ├── data_loader/wsi_patch_dataset.py
│   ├── models/feature_extractor.py
│   ├── utils/file_utils.py
│   └── wsi_feature_extraction.py
├── tests/test_patch_sampler.py
├── requirements.txt
└── README.md
```

## Notes mở rộng

- Có thể thay `patch_size`, `step_size` theo paper để đồng bộ pipeline.
- Có thể lưu thêm attention scores/thumbnail tuỳ mục đích downstream.
- Nếu dùng GPU, bật `feature_extraction.amp = true` để tăng tốc.

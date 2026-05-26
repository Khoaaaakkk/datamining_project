# Co-Attention Heatmap

Visualize genomic-guided co-attention weights on a WSI thumbnail.

## Inputs

- WSI file (`.svs`, `.tif`)
- H5 file with `coords` dataset (N x 2, level-0 coordinates)
- Numpy file with `attention_weights` (length N)

## Usage

```powershell
C:/Users/Administrator/Documents/Code/datamining_project/.venv/Scripts/python.exe \
  co_attention_heatmap/generate_heatmap.py \
  --wsi data/raw_wsi/<slide>.svs \
  --h5 data/h5_features/<slide>.h5 \
  --attention data/outputs/co_attention/<weights>.npy \
  --level 2 \
  --output outputs/co_attention_heatmap.png
```

## Notes

- `--patch-size` should match the patch size used during feature extraction (default: 256).
- `--alpha` controls overlay transparency (default: 0.5).

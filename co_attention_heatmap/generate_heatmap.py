from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wsi_preprocess.data_loader.batch_pipeline import run_gdc_download


def load_coords(h5_path: Path, dataset_key: str = "coords") -> np.ndarray:
    with h5py.File(h5_path, "r") as handle:
        if dataset_key not in handle:
            raise KeyError(f"Dataset '{dataset_key}' not found in {h5_path}")
        coords = handle[dataset_key][()]
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be shape (N, 2)")
    return coords.astype(np.float32)

def load_attention(attn_path: Path, expected_len: Optional[int] = None) -> np.ndarray:
    if attn_path.suffix.lower() == ".pt":
        payload = torch.load(attn_path, map_location="cpu")
        if isinstance(payload, dict):
            if "attention" not in payload:
                raise KeyError("co_attention payload missing 'attention' key")
            weights = payload["attention"]
        else:
            weights = payload
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        weights = np.asarray(weights)
    else:
        weights = np.load(attn_path)
        
    weights = weights.squeeze()
    
    # Xử lý để luôn trả về mảng 2D có dạng (K, N) với K là số lượng gene (ví dụ K = 6)
    if weights.ndim == 1:
        weights = np.expand_dims(weights, axis=0)
    elif weights.ndim == 3:
        # Nếu shape là (Batch, K, N), lấy batch đầu tiên
        weights = weights[0]
    elif weights.ndim > 3 or weights.ndim == 0:
        raise ValueError("attention_weights format is not supported")

    # Kiểm tra kích thước số lượng patch N (ở chiều thứ 1)
    if expected_len is not None:
        if weights.shape[1] > expected_len:
            weights = weights[:, :expected_len]
        elif weights.shape[1] < expected_len:
            print(f"Shape of weights: {weights.shape}")
            raise ValueError(
                "coords length must match attention_weights length"
            )
    return weights.astype(np.float32)

def normalize_weights(weights: np.ndarray) -> np.ndarray:
    min_val = float(weights.min())
    max_val = float(weights.max())
    if max_val - min_val < 1e-8:
        return np.zeros_like(weights)
    return (weights - min_val) / (max_val - min_val)

def get_thumbnail(slide: openslide.OpenSlide, level: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    width, height = slide.level_dimensions[level]
    thumbnail = slide.read_region((0, 0), level, (width, height)).convert("RGB")
    return np.array(thumbnail), (width, height)

def scale_coords(
    coords: np.ndarray,
    slide: openslide.OpenSlide,
    level: int,
) -> np.ndarray:
    downsample = slide.level_downsamples[level]
    return coords / float(downsample)

def build_heatmap_overlay(
    coords: np.ndarray,
    weights: np.ndarray,
    canvas_size: Tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    width, height = canvas_size
    overlay = np.zeros((height, width), dtype=np.uint8)

    for (x, y), weight in zip(coords, weights):
        x0 = int(round(x))
        y0 = int(round(y))
        x1 = min(x0 + patch_size, width)
        y1 = min(y0 + patch_size, height)
        value = int(weight * 255)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color=value, thickness=-1)

    colored = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
    return colored


def find_file_by_slideid(search_dir: Path, slide_id: str, patterns: Tuple[str, ...]) -> Path:
    match = find_optional_file_by_slideid(search_dir, slide_id, patterns)
    if match is None:
        raise FileNotFoundError(
            f"No files found for slide_id '{slide_id}' in {search_dir}"
        )
    return match


def find_optional_file_by_slideid(
    search_dir: Path,
    slide_id: str,
    patterns: Tuple[str, ...],
) -> Optional[Path]:
    if not search_dir.exists():
        return None
    matches = []
    for pattern in patterns:
        matches.extend(sorted(search_dir.rglob(pattern.format(slide_id=slide_id))))
    return matches[0] if matches else None


def download_wsi_if_needed(
    manifest: Optional[Path],
    gdc_client: Path,
    output_dir: Path,
    slide_id: str,
    patterns: Tuple[str, ...],
) -> None:
    if manifest is None:
        raise ValueError("--manifest is required to download missing WSI files")
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    if find_optional_file_by_slideid(output_dir, slide_id, patterns) is not None:
        return

    run_gdc_download(gdc_client, manifest, output_dir, dry_run=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize co-attention weights as a heatmap overlay on WSI thumbnails."
    )
    parser.add_argument(
        "--slide-id",
        type=str,
        default=None,
        help="Slide ID used to resolve WSI/H5/attention paths.",
    )
    parser.add_argument(
        "--wsi",
        type=Path,
        default=None,
        help="Path to WSI (.svs/.tif).",
    )
    parser.add_argument(
        "--h5",
        type=Path,
        default=None,
        help="Path to H5 file with coords.",
    )
    parser.add_argument(
        "--attention",
        type=Path,
        default=None,
        help="Path to .npy file containing attention weights",
    )
    parser.add_argument(
        "--wsi-dir",
        type=Path,
        default=Path("data/raw_wsi"),
        help="Base directory for WSI files when using --slide-id.",
    )
    parser.add_argument(
        "--h5-dir",
        type=Path,
        default=Path("data/h5_features"),
        help="Base directory for H5 files when using --slide-id.",
    )
    parser.add_argument(
        "--attention-dir",
        type=Path,
        default=Path("data/outputs/co_attention"),
        help="Base directory for attention files when using --slide-id.",
    )
    parser.add_argument(
        "--attention-ext",
        type=str,
        default=".npy",
        help="Extension for attention files when using --slide-id.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional GDC manifest to download WSI if missing.",
    )
    parser.add_argument(
        "--gdc-client",
        type=Path,
        default=Path("tools/gdc-client.exe"),
        help="Path to gdc-client executable for download.",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=2,
        help="WSI level to render (downsampled thumbnail)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Patch size at level 0 (used for rectangle size)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha blending value for heatmap overlay",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/co_attention_heatmap.png"),
        help="Output heatmap image path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.slide_id:
        slide_id = args.slide_id[:24]
        wsi_patterns = ("{slide_id}*.svs", "{slide_id}*.tif", "{slide_id}*.tiff")
        download_wsi_if_needed(
            args.manifest,
            args.gdc_client,
            args.wsi_dir,
            slide_id,
            wsi_patterns,
        )
        wsi_path = find_file_by_slideid(
            args.wsi_dir,
            slide_id,
            patterns=wsi_patterns,
        )
        h5_path = find_file_by_slideid(
            args.h5_dir,
            slide_id,
            patterns=("{slide_id}*.h5",),
        )
        attention_patterns = [
            f"{slide_id}*{args.attention_ext}",
            f"{slide_id}*_co_attention.pt",
        ]
        attention_path = None
        for pattern in attention_patterns:
            attention_path = find_optional_file_by_slideid(
                args.attention_dir,
                slide_id,
                patterns=(pattern,),
            )
            if attention_path is not None:
                break
        if attention_path is None:
            raise FileNotFoundError(
                f"No attention files found for slide_id '{slide_id}' in {args.attention_dir}"
            )
    else:
        if args.wsi is None or args.h5 is None or args.attention is None:
            raise ValueError("--wsi, --h5, --attention are required without --slide-id")
        wsi_path = args.wsi
        h5_path = args.h5
        attention_path = args.attention


    gene_names = [
    'CellCycle',
    'DNADamage',
    'EMT',
    'Hormone',
    'Immune',
    'Other'
    ]
    coords = load_coords(h5_path)
    print(f"Loaded coords shape: {coords.shape}")
    
    # weights_all lúc này có dạng (6, N)
    weights_all = load_attention(attention_path, expected_len=coords.shape[0])
    print(f"Loaded attention weights shape: {weights_all.shape}")

    slide = openslide.OpenSlide(str(wsi_path))
    thumbnail, canvas_size = get_thumbnail(slide, args.level)

    scaled_coords = scale_coords(coords, slide, args.level)
    scaled_patch_size = max(1, int(round(args.patch_size / slide.level_downsamples[args.level])))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    num_genes = weights_all.shape[0]
    
    # Vòng lặp tách qua 6 kênh và vẽ thành 6 file heatmap khác nhau
    for k in range(num_genes):
        # Lấy trọng số của gen thứ k có dạng 1D (N,)
        weights = weights_all[k]
        weights = normalize_weights(weights)

        heatmap = build_heatmap_overlay(
            scaled_coords,
            weights,
            canvas_size,
            scaled_patch_size,
        )
        blended = cv2.addWeighted(thumbnail, 1 - args.alpha, heatmap, args.alpha, 0)
        
        # Đổi tên file đầu ra, ví dụ: co_attention_heatmap.png -> co_attention_heatmap_gene_0.png
        out_name = f"{args.output.stem}_{slide_id}_gene_{gene_names[k]}{args.output.suffix}"
        out_path = args.output.with_name(out_name)
        
        cv2.imwrite(str(out_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        print(f"Saved heatmap for gene {k+1}/{num_genes} at: {out_path}")


if __name__ == "__main__":
    main()
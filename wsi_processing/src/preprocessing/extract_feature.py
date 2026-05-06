from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import openslide
import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_loader.wsi_dataset import WSIPatchDataset
from src.models.feature_extractor import build_feature_extractor, get_device
from src.preprocessing.quality import is_valid_patch
from src.preprocessing.segment import mask_has_tissue, segment_tissue_hsv_otsu
from src.preprocessing.stain_norm import MacenkoNormalizer
from src.utils.file_utils import ensure_dir, stem_without_double_suffix


def read_yaml(path: str | Path) -> Dict:
	with open(path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


class WSIFeaturePipeline:
	def __init__(self, config: Dict):
		self.cfg = config
		pp = config.get("preprocessing", {})
		fe = config.get("feature_extraction", {})

		self.patch_size = int(pp.get("patch_size", 256))
		self.step_size = int(pp.get("step_size", self.patch_size))
		self.seg_level = int(pp.get("seg_level", 2))
		self.tissue_threshold = float(pp.get("tissue_threshold", 0.2))
		self.max_white_ratio = float(pp.get("max_white_ratio", 0.7))
		self.min_laplacian_var = float(pp.get("min_laplacian_var", 50.0))

		self.use_stain_norm = bool(pp.get("use_stain_normalization", False))
		self.reference_image = pp.get("reference_image", "")

		self.batch_size = int(fe.get("batch_size", 64))
		self.num_workers = int(fe.get("num_workers", 2))
		self.device = get_device(fe.get("device"))
		self.model_name = str(fe.get("model_name", "swin_large_patch4_window7_224"))

		self.transform = transforms.Compose(
			[
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			]
		)

		self.model, self.feature_dim = build_feature_extractor(
			model_name=self.model_name,
			pretrained=bool(fe.get("pretrained", True)),
			image_size=self.patch_size,
		)
		self.model = self.model.to(self.device)
		self.model.eval()
		self.normalizer = MacenkoNormalizer(self.reference_image if self.use_stain_norm else None)

	def collect_candidate_coords(self, slide: openslide.OpenSlide, tissue_mask: np.ndarray, downsample: float) -> List[Tuple[int, int]]:
		w, h = slide.dimensions
		coords: List[Tuple[int, int]] = []

		mask_patch = max(1, int(self.patch_size / downsample))
		for y in range(0, h - self.patch_size + 1, self.step_size):
			for x in range(0, w - self.patch_size + 1, self.step_size):
				mx = int(x / downsample)
				my = int(y / downsample)
				if mask_has_tissue(tissue_mask, mx, my, mask_patch, threshold=self.tissue_threshold):
					coords.append((x, y))
		return coords

	def filter_coords_by_quality(self, slide: openslide.OpenSlide, coords: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
		valid: List[Tuple[int, int]] = []
		for x, y in tqdm(coords, desc="Quality filtering", leave=False):
			patch = slide.read_region((x, y), 0, (self.patch_size, self.patch_size)).convert("RGB")
			patch_np = np.asarray(patch)
			if is_valid_patch(
				patch_np,
				max_white_ratio=self.max_white_ratio,
				min_laplacian_var=self.min_laplacian_var,
			):
				valid.append((x, y))
		return valid

	def extract_features_batch(self, svs_path: str, coords: Sequence[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
		if len(coords) == 0:
			return np.zeros((0, self.feature_dim), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)

		dataset = WSIPatchDataset(svs_path, coords, self.patch_size, transform=None)

		def collate_fn(batch):
			patches, c = zip(*batch)
			proc = []
			for pil_patch in patches:
				rgb = np.asarray(pil_patch)
				if self.use_stain_norm:
					rgb = self.normalizer.transform(rgb)
					pil_patch = Image.fromarray(rgb.astype(np.uint8))
				proc.append(self.transform(pil_patch))
			return torch.stack(proc, dim=0), torch.stack(c, dim=0)

		loader = DataLoader(
			dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.num_workers,
			pin_memory=torch.cuda.is_available(),
			collate_fn=collate_fn,
		)

		features_list: List[np.ndarray] = []
		coords_list: List[np.ndarray] = []

		amp_enabled = self.device.type == "cuda" and torch.cuda.is_available()
		with torch.no_grad():
			for patches, c in tqdm(loader, desc="Feature extraction"):
				patches = patches.to(self.device, non_blocking=True)
				with torch.autocast(device_type="cuda", enabled=amp_enabled):
					feat = self.model(patches)
				features_list.append(feat.detach().cpu().numpy().astype(np.float32))
				coords_list.append(c.detach().cpu().numpy().astype(np.int32))

		return np.vstack(features_list), np.vstack(coords_list)

	def process_slide(self, svs_path: str, output_h5_path: str, output_mask_path: str | None = None) -> Dict:
		slide = openslide.OpenSlide(svs_path)
		try:
			tissue_mask, downsample = segment_tissue_hsv_otsu(slide, level=self.seg_level)
			candidate_coords = self.collect_candidate_coords(slide, tissue_mask, downsample)
			valid_coords = self.filter_coords_by_quality(slide, candidate_coords)
			features, coords = self.extract_features_batch(svs_path, valid_coords)

			ensure_dir(Path(output_h5_path).parent)
			with h5py.File(output_h5_path, "w") as f:
				f.create_dataset("features", data=features, compression="gzip")
				f.create_dataset("coords", data=coords, compression="gzip")
				f.attrs["slide_path"] = str(svs_path)
				f.attrs["backbone"] = self.model_name
				f.attrs["patch_size"] = self.patch_size
				f.attrs["step_size"] = self.step_size
				f.attrs["n_patches"] = int(coords.shape[0])
				f.attrs["feature_dim"] = int(features.shape[1]) if features.size else self.feature_dim

			if output_mask_path:
				ensure_dir(Path(output_mask_path).parent)
				Image.fromarray(tissue_mask).save(output_mask_path)

			return {
				"slide_path": svs_path,
				"candidate_patches": len(candidate_coords),
				"valid_patches": len(valid_coords),
				"saved_h5": output_h5_path,
				"saved_mask": output_mask_path,
			}
		finally:
			slide.close()


def parse_args():
	parser = argparse.ArgumentParser(description="WSI preprocessing and feature extraction pipeline")
	parser.add_argument("--config", type=str, default="configs/default.yaml")
	parser.add_argument("--slide", type=str, default="", help="Single .svs path. If empty, process all files in raw_wsi dir")
	parser.add_argument("--output-dir", type=str, default="", help="Override output h5 directory")
	parser.add_argument("--save-report", type=str, default="", help="Path to save json processing report")
	return parser.parse_args()


def main():
	args = parse_args()
	cfg = read_yaml(args.config)

	data_cfg = cfg.get("data", {})
	raw_dir = Path(data_cfg.get("raw_wsi_dir", "data/raw_wsi"))
	h5_dir = Path(args.output_dir or data_cfg.get("h5_features_dir", "data/h5_features"))
	masks_dir = Path(data_cfg.get("masks_dir", "data/masks"))

	ensure_dir(h5_dir)
	ensure_dir(masks_dir)

	if args.slide:
		slides = [Path(args.slide)]
	else:
		slides = sorted([p for p in raw_dir.glob("*") if p.suffix.lower() in {".svs", ".tif", ".tiff", ".ndpi"}])

	if len(slides) == 0:
		raise FileNotFoundError(f"No slide files found in {raw_dir}")

	pipeline = WSIFeaturePipeline(cfg)
	report = []
	for slide_path in slides:
		slide_id = stem_without_double_suffix(slide_path)
		out_h5 = h5_dir / f"{slide_id}.h5"
		out_mask = masks_dir / f"{slide_id}.png"
		result = pipeline.process_slide(str(slide_path), str(out_h5), str(out_mask))
		report.append(result)
		print(f"[DONE] {slide_id}: {result['valid_patches']} patches")

	report_path = args.save_report or cfg.get("report_path", "experiments/logs/preprocessing_report.json")
	ensure_dir(Path(report_path).parent)
	with open(report_path, "w", encoding="utf-8") as f:
		json.dump(report, f, indent=2, ensure_ascii=False)

	print(f"Saved report -> {report_path}")


if __name__ == "__main__":
	main()

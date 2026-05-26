from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from wsi_preprocess.data_loader.wsi_patch_dataset import WSIPatchDataset
from wsi_preprocess.models.feature_extractor import build_feature_extractor, get_device
from wsi_preprocess.preprocessing.patch_quality import is_valid_patch
from wsi_preprocess.preprocessing.patch_sampler import generate_patch_coords
from wsi_preprocess.preprocessing.stain_normalization import MacenkoNormalizer
from wsi_preprocess.preprocessing.tissue_segmentation import segment_tissue_hsv_otsu
from wsi_preprocess.utils.file_utils import ensure_dir


def _configure_openslide_dll() -> None:
	"""Best-effort OpenSlide DLL discovery for Windows."""
	try:
		from ctypes import WinDLL  # noqa: F401
	except Exception:
		return

	import os
	from pathlib import Path

	search_dirs = []
	for env_key in ("OPENSLIDE_DLL_DIR", "OPENSLIDE_PATH", "OPENSLIDE_HOME"):
		value = os.environ.get(env_key)
		if value:
			search_dirs.append(Path(value))

	repo_root = Path(__file__).resolve().parents[3]
	search_dirs.extend(
		[
			repo_root / "tools" / "openslide" / "bin",
			repo_root / "tools" / "openslide",
		]
	)

	for directory in search_dirs:
		if directory.exists() and directory.is_dir():
			try:
				os.add_dll_directory(str(directory))
			except Exception:
				continue


_configure_openslide_dll()

try:
	import openslide
except ModuleNotFoundError as exc:
	raise ModuleNotFoundError(
		"Couldn't locate OpenSlide DLL. Install OpenSlide binaries or set OPENSLIDE_DLL_DIR to the DLL folder. "
		"On Windows, you can place binaries in tools/openslide/bin and re-run."
	) from exc


class WSIEmbeddingPipeline:
	"""WSI -> patch embeddings pipeline used by MMEF-style MIL."""

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
		if __import__("os").name == "nt":
			self.num_workers = 0
		self.device = get_device(fe.get("device"))
		self.model_name = str(fe.get("model_name", "swin_large_patch4_window7_224"))
		self.amp = bool(fe.get("amp", True))

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

	def collect_candidate_coords(
		self,
		slide: openslide.OpenSlide,
		tissue_mask: np.ndarray,
		downsample: float,
	) -> List[Tuple[int, int]]:
		return generate_patch_coords(
			slide_dims=slide.dimensions,
			patch_size=self.patch_size,
			step_size=self.step_size,
			mask=tissue_mask,
			downsample=downsample,
			tissue_threshold=self.tissue_threshold,
		)

	def filter_coords_by_quality(
		self, slide: openslide.OpenSlide, coords: Sequence[Tuple[int, int]]
	) -> List[Tuple[int, int]]:
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

	def extract_embeddings_batch(
		self, svs_path: str, coords: Sequence[Tuple[int, int]]
	) -> Tuple[np.ndarray, np.ndarray]:
		if len(coords) == 0:
			return np.zeros((0, self.feature_dim), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)

		dataset = WSIPatchDataset(svs_path, list(coords), self.patch_size)

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

		amp_enabled = self.device.type == "cuda" and torch.cuda.is_available() and self.amp
		with torch.no_grad():
			for patches, c in tqdm(loader, desc="Embedding extraction"):
				patches = patches.to(self.device, non_blocking=True)
				with torch.autocast(device_type="cuda", enabled=amp_enabled):
					feat = self.model(patches)
				features_list.append(feat.detach().cpu().numpy().astype(np.float32))
				coords_list.append(c.detach().cpu().numpy().astype(np.int32))

		dataset.close()
		return np.vstack(features_list), np.vstack(coords_list)

	def save_h5(
		self,
		output_h5_path: str,
		features: np.ndarray,
		coords: np.ndarray,
		slide_path: str,
	) -> None:
		ensure_dir(Path(output_h5_path).parent)
		with h5py.File(output_h5_path, "w") as f:
			f.create_dataset("features", data=features, compression="gzip")
			f.create_dataset("coords", data=coords, compression="gzip")
			f.attrs["slide_path"] = str(slide_path)
			f.attrs["backbone"] = self.model_name
			f.attrs["patch_size"] = self.patch_size
			f.attrs["step_size"] = self.step_size
			f.attrs["n_patches"] = int(coords.shape[0])
			f.attrs["feature_dim"] = int(features.shape[1]) if features.size else self.feature_dim

	def process_slide(self, svs_path: str, output_h5_path: str, output_mask_path: str | None = None) -> Dict:
		slide = openslide.OpenSlide(svs_path)
		try:
			tissue_mask, downsample = segment_tissue_hsv_otsu(slide, level=self.seg_level)
			candidate_coords = self.collect_candidate_coords(slide, tissue_mask, downsample)
			valid_coords = self.filter_coords_by_quality(slide, candidate_coords)
			features, coords = self.extract_embeddings_batch(svs_path, valid_coords)

			self.save_h5(output_h5_path, features, coords, svs_path)

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


def save_report(report_path: str, reports: List[Dict]) -> None:
	ensure_dir(Path(report_path).parent)
	Path(report_path).write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")

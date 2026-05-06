from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import openslide


def segment_tissue_hsv_otsu(
	slide: openslide.OpenSlide,
	level: int = 2,
	morph_kernel: int = 5,
	min_tissue_ratio: float = 0.01,
) -> Tuple[np.ndarray, float]:
	"""
	Segment tissue on low-resolution thumbnail via HSV saturation + Otsu.

	Returns:
		mask: uint8 binary mask (0/255) at selected level.
		downsample: level downsample factor relative to level 0.
	"""
	level = min(level, slide.level_count - 1)
	w, h = slide.level_dimensions[level]
	downsample = float(slide.level_downsamples[level])

	thumb = slide.read_region((0, 0), level, (w, h)).convert("RGB")
	thumb_np = np.asarray(thumb)

	hsv = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2HSV)
	sat = hsv[:, :, 1]
	_, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

	tissue_ratio = float((mask > 0).mean())
	if tissue_ratio < min_tissue_ratio:
		mask = np.zeros_like(mask, dtype=np.uint8)

	return mask.astype(np.uint8), downsample


def mask_has_tissue(mask: np.ndarray, x: int, y: int, patch_size: int, threshold: float = 0.2) -> bool:
	"""Check tissue coverage inside a mask region."""
	h, w = mask.shape
	x2 = min(x + patch_size, w)
	y2 = min(y + patch_size, h)
	if x >= w or y >= h or x2 <= x or y2 <= y:
		return False
	region = mask[y:y2, x:x2]
	return float((region > 0).mean()) >= threshold

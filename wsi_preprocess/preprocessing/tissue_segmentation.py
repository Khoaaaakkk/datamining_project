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
	"""Segment tissue on a low-resolution thumbnail using HSV saturation + Otsu."""
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

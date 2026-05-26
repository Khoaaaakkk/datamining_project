from __future__ import annotations

from typing import List, Tuple

import numpy as np


def mask_has_tissue(mask: np.ndarray, x: int, y: int, patch_size: int, threshold: float = 0.2) -> bool:
	"""Check tissue coverage inside a mask region."""
	h, w = mask.shape
	x2 = min(x + patch_size, w)
	y2 = min(y + patch_size, h)
	if x >= w or y >= h or x2 <= x or y2 <= y:
		return False
	region = mask[y:y2, x:x2]
	return float((region > 0).mean()) >= threshold


def generate_patch_coords(
	slide_dims: Tuple[int, int],
	patch_size: int,
	step_size: int,
	mask: np.ndarray | None,
	downsample: float,
	tissue_threshold: float,
) -> List[Tuple[int, int]]:
	"""Generate patch coordinates on level-0 using a low-res tissue mask."""
	w, h = slide_dims
	coords: List[Tuple[int, int]] = []

	if mask is None or mask.size == 0:
		return coords

	mask_patch = max(1, int(patch_size / max(downsample, 1.0)))
	for y in range(0, h - patch_size + 1, step_size):
		for x in range(0, w - patch_size + 1, step_size):
			mx = int(x / downsample)
			my = int(y / downsample)
			if mask_has_tissue(mask, mx, my, mask_patch, threshold=tissue_threshold):
				coords.append((x, y))
	return coords

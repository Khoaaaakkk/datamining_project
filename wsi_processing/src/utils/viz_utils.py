from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def overlay_mask(
	rgb_image: np.ndarray,
	mask: np.ndarray,
	alpha: float = 0.35,
	color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
	"""Overlay binary mask on RGB image and return blended image."""
	if mask.ndim != 2:
		raise ValueError("mask must be a 2D array")

	if rgb_image.shape[:2] != mask.shape[:2]:
		mask = cv2.resize(mask, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)

	out = rgb_image.copy()
	color_arr = np.array(color, dtype=np.uint8)
	out[mask > 0] = ((1 - alpha) * out[mask > 0] + alpha * color_arr).astype(np.uint8)
	return out


def plot_patches_grid(
	patches: Sequence[np.ndarray],
	coords: Iterable[Tuple[int, int]] | None = None,
	cols: int = 6,
	figsize: Tuple[int, int] = (14, 10),
) -> None:
	"""Quick visualization of extracted patches."""
	if len(patches) == 0:
		raise ValueError("patches is empty")

	rows = (len(patches) + cols - 1) // cols
	plt.figure(figsize=figsize)
	coords = list(coords) if coords is not None else [None] * len(patches)

	for i, patch in enumerate(patches):
		ax = plt.subplot(rows, cols, i + 1)
		ax.imshow(patch)
		ax.axis("off")
		if i < len(coords) and coords[i] is not None:
			x, y = coords[i]
			ax.set_title(f"({x},{y})", fontsize=8)

	plt.tight_layout()
	plt.show()

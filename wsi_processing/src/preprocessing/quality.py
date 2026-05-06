from __future__ import annotations

import cv2
import numpy as np


def compute_white_ratio(patch_rgb: np.ndarray, white_threshold: int = 210) -> float:
	gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
	return float((gray > white_threshold).mean())


def compute_blur_score_laplacian(patch_rgb: np.ndarray) -> float:
	gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
	return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_valid_patch(
	patch_rgb: np.ndarray,
	max_white_ratio: float = 0.7,
	min_laplacian_var: float = 50.0,
	min_std: float = 5.0,
) -> bool:
	"""Filter empty/white and blurry patches."""
	white_ratio = compute_white_ratio(patch_rgb)
	if white_ratio > max_white_ratio:
		return False

	blur_score = compute_blur_score_laplacian(patch_rgb)
	if blur_score < min_laplacian_var:
		return False

	if float(patch_rgb.std()) < min_std:
		return False

	return True

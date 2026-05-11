from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
	import staintools
except Exception:  # pragma: no cover - optional dependency runtime fallback
	staintools = None


class MacenkoNormalizer:
	"""Thin wrapper around staintools with graceful degradation."""

	def __init__(self, reference_image_path: Optional[str] = None):
		self.enabled = staintools is not None
		self._normalizer = None
		if not self.enabled:
			return

		self._normalizer = staintools.StainNormalizer(method="macenko")
		if reference_image_path:
			ref_path = Path(reference_image_path)
			if ref_path.exists():
				target = np.array(Image.open(ref_path).convert("RGB"))
				self._normalizer.fit(target)

	def fit(self, reference_rgb: np.ndarray) -> None:
		if not self.enabled or self._normalizer is None:
			return
		self._normalizer.fit(reference_rgb)

	def transform(self, patch_rgb: np.ndarray) -> np.ndarray:
		if not self.enabled or self._normalizer is None:
			return patch_rgb
		try:
			return self._normalizer.transform(patch_rgb)
		except Exception:
			return patch_rgb

from __future__ import annotations

from typing import List, Tuple

import torch
from PIL import Image


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
from torch.utils.data import Dataset


class WSIPatchDataset(Dataset):
	"""Lazy-load WSI patches from a slide at given coords."""

	def __init__(self, slide_path: str, coords: List[Tuple[int, int]], patch_size: int):
		self.slide_path = slide_path
		self.coords = coords
		self.patch_size = patch_size
		self._slide: openslide.OpenSlide | None = None

	def _get_slide(self) -> openslide.OpenSlide:
		if self._slide is None:
			self._slide = openslide.OpenSlide(self.slide_path)
		return self._slide

	def __len__(self) -> int:
		return len(self.coords)

	def __getitem__(self, idx: int):
		x, y = self.coords[idx]
		slide = self._get_slide()
		patch = slide.read_region((x, y), 0, (self.patch_size, self.patch_size)).convert("RGB")
		return patch, torch.tensor([x, y], dtype=torch.int32)

	def close(self) -> None:
		slide = getattr(self, "_slide", None)
		if slide is not None:
			slide.close()
			self._slide = None

	def __del__(self) -> None:
		self.close()

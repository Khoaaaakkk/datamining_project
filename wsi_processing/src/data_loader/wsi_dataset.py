from __future__ import annotations

from typing import List, Sequence, Tuple

import openslide
import torch
from torch.utils.data import Dataset


class WSIPatchDataset(Dataset):
	"""
	Lazy patch reader for OpenSlide usable with multi-process DataLoader.
	"""

	def __init__(
		self,
		svs_path: str,
		coords_list: Sequence[Tuple[int, int]],
		patch_size: int,
		transform=None,
	):
		self.svs_path = svs_path
		self.coords_list: List[Tuple[int, int]] = [tuple(map(int, c)) for c in coords_list]
		self.patch_size = int(patch_size)
		self.transform = transform
		self.slide = None

	def __len__(self) -> int:
		return len(self.coords_list)

	def __getitem__(self, idx: int):
		if self.slide is None:
			self.slide = openslide.OpenSlide(self.svs_path)

		x, y = self.coords_list[idx]
		patch = self.slide.read_region((x, y), 0, (self.patch_size, self.patch_size)).convert("RGB")
		if self.transform is not None:
			patch = self.transform(patch)
		return patch, torch.tensor([x, y], dtype=torch.int32)

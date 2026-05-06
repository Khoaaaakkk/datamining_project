from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset


class H5BagDataset(Dataset):
	"""
	MIL bag dataset:
	- each sample = one slide .h5 file containing datasets: features [N,D], coords [N,2]
	- labels loaded from CSV with columns: slide_id,label
	"""

	def __init__(
		self,
		h5_dir: str,
		labels_csv: Optional[str] = None,
		min_instances: int = 1,
	):
		self.h5_dir = Path(h5_dir)
		self.files: List[Path] = sorted(self.h5_dir.glob("*.h5"))
		self.min_instances = int(min_instances)

		label_map: Dict[str, int] = {}
		if labels_csv and Path(labels_csv).exists():
			df = pd.read_csv(labels_csv)
			if {"slide_id", "label"}.issubset(df.columns):
				label_map = {str(r.slide_id): int(r.label) for r in df.itertuples(index=False)}
		self.label_map = label_map

	def __len__(self) -> int:
		return len(self.files)

	def __getitem__(self, idx: int):
		fp = self.files[idx]
		with h5py.File(fp, "r") as f:
			features = f["features"][:]
			coords = f["coords"][:]

		if features.shape[0] < self.min_instances:
			raise ValueError(f"Bag {fp.name} has too few instances: {features.shape[0]}")

		slide_id = fp.stem
		label = self.label_map.get(slide_id, -1)

		return {
			"slide_id": slide_id,
			"features": torch.tensor(features, dtype=torch.float32),
			"coords": torch.tensor(coords, dtype=torch.int32),
			"label": torch.tensor(label, dtype=torch.long),
		}

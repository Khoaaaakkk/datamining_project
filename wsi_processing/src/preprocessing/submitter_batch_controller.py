from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List

import pandas as pd


class SubmitterBatchController:
	"""Resolve submitter IDs and iterate them in batches."""

	def __init__(self, config: Dict):
		self.cfg = config
		self.gdc_cfg = config.get("gdc_download", {})

		self.csv_file = Path(self.gdc_cfg.get("clinical_csv", "data/reference/Final_Matched_Clinical.csv"))
		self.submitter_column = str(self.gdc_cfg.get("submitter_id_column", "submitter_id"))
		self.batch_size = int(self.gdc_cfg.get("batch_size", 5))
		self.max_patients = self.gdc_cfg.get("max_patients", None)

		raw_whitelist = self.gdc_cfg.get("submitter_id_whitelist", None)
		if raw_whitelist is None:
			self.submitter_id_whitelist: List[str] = []
		elif isinstance(raw_whitelist, list):
			self.submitter_id_whitelist = [str(x).strip() for x in raw_whitelist if str(x).strip()]
		else:
			raise TypeError("gdc_download.submitter_id_whitelist must be a list of strings")

	def resolve_submitter_ids(self) -> List[str]:
		"""
		Priority:
		1) gdc_download.submitter_id_whitelist (if non-empty)
		2) clinical_csv + submitter_id_column
		"""
		if len(self.submitter_id_whitelist) > 0:
			submitter_ids = list(dict.fromkeys(self.submitter_id_whitelist))
		else:
			if not self.csv_file.exists():
				raise FileNotFoundError(
					f"Clinical CSV not found: {self.csv_file}. "
					"Provide gdc_download.submitter_id_whitelist or fix gdc_download.clinical_csv"
				)
			df = pd.read_csv(self.csv_file)
			if self.submitter_column not in df.columns:
				raise KeyError(f"Column '{self.submitter_column}' not found in {self.csv_file}")
			submitter_ids = [str(v) for v in df[self.submitter_column].dropna().unique().tolist()]

		if self.max_patients is not None:
			submitter_ids = submitter_ids[: int(self.max_patients)]
		return submitter_ids

	def iter_batches(self, submitter_ids: List[str]) -> Iterator[List[str]]:
		if self.batch_size <= 0:
			raise ValueError("gdc_download.batch_size must be > 0")
		for i in range(0, len(submitter_ids), self.batch_size):
			yield submitter_ids[i : i + self.batch_size]

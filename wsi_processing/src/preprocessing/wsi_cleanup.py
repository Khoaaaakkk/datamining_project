from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, List


class WSICleanupManager:
	"""Handle post-processing cleanup for downloaded WSI files."""

	def __init__(self, enabled: bool = True):
		self.enabled = bool(enabled)

	def cleanup_slide(self, slide_path: str | Path) -> bool:
		"""Delete one WSI file if cleanup is enabled. Return True if deleted."""
		if not self.enabled:
			return False
		p = Path(slide_path)
		if not p.exists() or not p.is_file():
			return False
		try:
			p.unlink()
			return True
		except Exception as e:
			print(f"[WARN] Could not delete processed slide {p}: {e}")
			return False

	def cleanup_many(self, slide_paths: Iterable[str | Path]) -> List[Path]:
		"""Delete many WSI files, return deleted paths."""
		deleted: List[Path] = []
		for sp in slide_paths:
			p = Path(sp)
			if self.cleanup_slide(p):
				deleted.append(p)
		return deleted

	def cleanup_raw_wsi_subfolders(self, raw_wsi_dir: str | Path) -> List[Path]:
		"""Delete all direct subfolders inside raw_wsi_dir, return deleted paths."""
		deleted: List[Path] = []
		if not self.enabled:
			return deleted

		root = Path(raw_wsi_dir)
		if not root.exists() or not root.is_dir():
			return deleted

		for child in root.iterdir():
			if not child.is_dir():
				continue
			try:
				shutil.rmtree(child)
				deleted.append(child)
			except Exception as e:
				print(f"[WARN] Could not delete raw_wsi folder {child}: {e}")

		return deleted

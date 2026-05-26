from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def ensure_dir(path: str | Path) -> Path:
	"""Create directory if missing and return resolved Path."""
	p = Path(path)
	p.mkdir(parents=True, exist_ok=True)
	return p


def ensure_parent(path: str | Path) -> Path:
	"""Create parent directory of a file path and return parent Path."""
	p = Path(path)
	p.parent.mkdir(parents=True, exist_ok=True)
	return p.parent


def list_files(directory: str | Path, suffixes: Iterable[str]) -> List[Path]:
	"""Return sorted files in directory matching any suffix."""
	d = Path(directory)
	suffix_set = {s.lower() for s in suffixes}
	if not d.exists():
		return []
	return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in suffix_set])


def stem_without_double_suffix(path: str | Path) -> str:
	"""Handle names like *.svs and *.tar.gz safely for ID creation."""
	p = Path(path)
	name = p.name
	for suffix in [".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn"]:
		if name.lower().endswith(suffix):
			return name[: -len(suffix)]
	return p.stem

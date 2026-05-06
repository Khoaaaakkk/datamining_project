from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import requests

from src.utils.file_utils import ensure_dir

SUPPORTED_EXTENSIONS = {".svs"}


class GDCSVSDownloader:
	def __init__(self, config: Dict):
		self.cfg = config
		self.data_cfg = config.get("data", {})
		self.gdc_cfg = config.get("gdc_download", {})
		self.project_root = Path(__file__).resolve().parents[2]

		self.raw_wsi_dir = self._resolve_path(self.data_cfg.get("raw_wsi_dir", "data/raw_wsi"))
		self.api_url = str(self.gdc_cfg.get("api_url", "https://api.gdc.cancer.gov/files"))
		self.data_format = str(self.gdc_cfg.get("data_format", "SVS"))
		self.max_files_per_patient = int(self.gdc_cfg.get("max_files_per_patient", 1))
		self.request_timeout_sec = int(self.gdc_cfg.get("request_timeout_sec", 60))

		self.client_dir = self._resolve_path(self.gdc_cfg.get("client_dir", "src/tools/gdc_client"))
		self.client_path = self._resolve_client_path(self.gdc_cfg.get("client_path", str(self.client_dir / "gdc-client.exe")))
		self.manifest_dir = self._resolve_path(self.gdc_cfg.get("manifest_dir", "data/reference/manifests"))

		extensions = self.gdc_cfg.get("downloaded_extensions", list(SUPPORTED_EXTENSIONS))
		self.downloaded_extensions = {str(ext).lower() for ext in extensions}

		ensure_dir(self.raw_wsi_dir)
		ensure_dir(self.client_dir)
		ensure_dir(self.manifest_dir)

	def _resolve_path(self, path_value: str | Path) -> Path:
		p = Path(path_value)
		if p.is_absolute():
			return p
		return (self.project_root / p).resolve()

	def _resolve_client_path(self, configured_path: str | Path) -> Path:
		configured = self._resolve_path(configured_path)
		if configured.exists() and not (os.name != "nt" and configured.suffix.lower() == ".exe"):
			return configured

		candidates = [
			self.client_dir / "gdc-client",
			self.client_dir / "gdc-client.exe",
			(self.project_root / "src/tools/gdc_client/gdc-client").resolve(),
			(self.project_root / "src/tools/gdc_client/gdc-client.exe").resolve(),
			(self.project_root / "tools/gdc_client/gdc-client").resolve(),
			(self.project_root / "tools/gdc_client/gdc-client.exe").resolve(),
		]
		for c in candidates:
			if c.exists():
				return c

		path_candidates = [shutil.which("gdc-client"), shutil.which("gdc-client.exe")]
		for p in path_candidates:
			if p:
				return Path(p).resolve()

		# If configured path exists (even incompatible), return it for explicit validation error.
		if configured.exists():
			return configured

		# Return configured location for clear error message in _run_gdc_download
		return configured

	def _validate_client_binary(self) -> None:
		"""Validate executable compatibility before invoking gdc-client."""
		if os.name != "nt" and self.client_path.suffix.lower() == ".exe":
			raise RuntimeError(
				"Detected Windows gdc-client executable on a non-Windows system: "
				f"{self.client_path}.\n"
				"This causes download failures (e.g. WinError path/state errors).\n"
				"Please use Linux gdc-client binary named 'gdc-client' and update config:\n"
				"  gdc_download.client_path: src/tools/gdc_client/gdc-client"
			)

	def _query_svs_files(self, submitter_id: str) -> List[Dict]:
		filters = {
			"op": "and",
			"content": [
				{"op": "in", "content": {"field": "cases.submitter_id", "value": [submitter_id]}},
				{"op": "in", "content": {"field": "files.data_format", "value": [self.data_format]}},
			],
		}
		params = {
			"filters": json.dumps(filters),
			"fields": "file_id,file_name,md5sum,file_size,state",
			"size": str(max(1, self.max_files_per_patient)),
		}
		resp = requests.get(self.api_url, params=params, timeout=self.request_timeout_sec)
		resp.raise_for_status()
		data = resp.json()
		hits = data.get("data", {}).get("hits", [])
		return hits[: self.max_files_per_patient]

	def _build_manifest_entries(self, submitter_ids: List[str]) -> List[Dict]:
		entries: List[Dict] = []
		seen = set()
		for sid in submitter_ids:
			try:
				hits = self._query_svs_files(sid)
			except Exception as e:
				print(f"[WARN] Query failed for {sid}: {e}")
				continue
			for h in hits:
				fid = h.get("file_id", "")
				if not fid or fid in seen:
					continue
				seen.add(fid)
				entries.append(
					{
						"id": fid,
						"filename": h.get("file_name", ""),
						"md5": h.get("md5sum", ""),
						"size": str(h.get("file_size", "")),
						"state": h.get("state", ""),
					}
				)
		return entries

	def _write_manifest(self, entries: List[Dict], manifest_path: Path) -> None:
		lines = ["id\tfilename\tmd5\tsize\tstate"]
		for e in entries:
			lines.append(
				f"{e['id']}\t{e['filename']}\t{e['md5']}\t{e['size']}\t{e['state']}"
			)
		manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

	def _run_gdc_download(self, manifest_path: Path) -> None:
		if not self.client_path.exists():
			raise FileNotFoundError(
				f"gdc-client not found at {self.client_path}. "
				f"Please place executable inside {self.client_dir} and make it executable."
			)
		self._validate_client_binary()
		cmd = [str(self.client_path), "download", "-m", str(manifest_path), "-d", str(self.raw_wsi_dir)]
		try:
			subprocess.run(cmd, check=True)
		except subprocess.CalledProcessError as e:
			raise RuntimeError(
				"gdc-client download failed. "
				f"command={cmd}, returncode={e.returncode}. "
				"If you're on Linux, ensure you are using Linux gdc-client binary (not .exe)."
			) from e

	def _collect_downloaded_slide_paths(self, entries: List[Dict]) -> List[Path]:
		result: List[Path] = []
		for e in entries:
			fid = e["id"]
			file_name = str(e.get("filename", ""))
			if file_name:
				existing_root_file = self.raw_wsi_dir / file_name
				if existing_root_file.exists() and existing_root_file.suffix.lower() in self.downloaded_extensions:
					result.append(existing_root_file)

			download_folder = self.raw_wsi_dir / fid
			if not download_folder.exists():
				continue

			for f in download_folder.rglob("*"):
				if not f.is_file() or f.suffix.lower() not in self.downloaded_extensions:
					continue
				dst = self.raw_wsi_dir / f.name
				if f.resolve() != dst.resolve():
					if dst.exists():
						# If same file already exists, keep existing and remove duplicate
						if f.stat().st_size == dst.stat().st_size:
							f.unlink(missing_ok=True)
						else:
							stem = dst.stem
							suffix = dst.suffix
							dst = self.raw_wsi_dir / f"{stem}_{fid[:8]}{suffix}"
					shutil.move(str(f), str(dst))
				result.append(dst)

			if download_folder.exists():
				for _ in range(2):
					try:
						if any(download_folder.iterdir()):
							break
						download_folder.rmdir()
						break
					except Exception:
						break

		# de-duplicate while preserving order
		unique = []
		seen = set()
		for p in result:
			sp = str(p.resolve())
			if sp not in seen:
				seen.add(sp)
				unique.append(p)
		return unique

	def download_for_submitter_ids(self, submitter_ids: List[str], batch_index: int = 1) -> List[Path]:
		"""Download .svs files for provided submitter IDs and return local downloaded slide paths."""
		if len(submitter_ids) == 0:
			return []

		entries = self._build_manifest_entries(submitter_ids)
		if len(entries) == 0:
			print(f"[WARN] Batch {batch_index}: no valid GDC entries")
			return []

		manifest_path = self.manifest_dir / f"batch_{batch_index:04d}.txt"
		self._write_manifest(entries, manifest_path)
		print(f"[INFO] Batch {batch_index}: manifest -> {manifest_path}")

		self._run_gdc_download(manifest_path)
		downloaded_slides = self._collect_downloaded_slide_paths(entries)
		print(f"[INFO] Batch {batch_index}: downloaded slides={len(downloaded_slides)}")
		return downloaded_slides

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ManifestRecord:
	file_id: str
	filename: str


def read_manifest(manifest_path: Path) -> List[ManifestRecord]:
	if not manifest_path.exists():
		raise FileNotFoundError(f"Manifest not found: {manifest_path}")

	records: List[ManifestRecord] = []
	with manifest_path.open("r", encoding="utf-8") as f:
		reader = csv.DictReader(f, delimiter="\t")
		for row in reader:
			file_id = str(row.get("id", "")).strip()
			filename = str(row.get("filename", "")).strip()
			if file_id and filename:
				records.append(ManifestRecord(file_id=file_id, filename=filename))

	if not records:
		raise ValueError(f"No records found in manifest: {manifest_path}")

	return records


def build_gdc_command(gdc_client: Path, manifest: Path, out_dir: Path) -> List[str]:
	return [str(gdc_client), "download", "-m", str(manifest), "-d", str(out_dir)]


def run_gdc_download(gdc_client: Path, manifest: Path, out_dir: Path, dry_run: bool) -> None:
	if not gdc_client.exists():
		raise FileNotFoundError(f"gdc-client not found: {gdc_client}")

	out_dir.mkdir(parents=True, exist_ok=True)
	command = build_gdc_command(gdc_client, manifest, out_dir)
	print(f"[INFO] Running: {' '.join(command)}")
	if dry_run:
		return

	subprocess.run(command, check=True)


def find_slide_path(raw_dir: Path, record: ManifestRecord) -> Optional[Path]:
	candidate = raw_dir / record.file_id / record.filename
	if candidate.exists():
		return candidate
	return None


def run_processing(
	processor_script: Path,
	config_path: Path,
	slide_path: Path,
	output_dir: Optional[Path],
	skip_existing: bool,
	dry_run: bool,
) -> None:
	command = [
		sys.executable,
		str(processor_script),
		"--config",
		str(config_path),
		"--slide",
		str(slide_path),
	]
	if output_dir:
		command.extend(["--output-dir", str(output_dir)])
	if skip_existing:
		command.append("--skip-existing")

	print(f"[INFO] Processing: {' '.join(command)}")
	if dry_run:
		return

	subprocess.run(command, check=True)


def delete_slide(slide_path: Path, dry_run: bool) -> None:
	if dry_run:
		print(f"[DRY-RUN] Delete {slide_path}")
		return
	try:
		slide_path.unlink()
		print(f"[INFO] Deleted {slide_path}")
	except FileNotFoundError:
		print(f"[WARN] Slide not found for deletion: {slide_path}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Download one GDC manifest batch, process it, then delete slides based on manifest."
	)
	parser.add_argument(
		"--manifest",
		required=True,
		help="Path to batch manifest (e.g., data/reference/manifests/batch_0004.txt)",
	)
	parser.add_argument(
		"--gdc-client",
		default="tools/gdc-client.exe",
		help="Path to gdc-client.exe",
	)
	parser.add_argument(
		"--raw-dir",
		default="data/raw_wsi",
		help="Download output directory for raw WSI files.",
	)
	parser.add_argument(
		"--processor",
		default="wsi_preprocessing_v2/src/wsi_feature_extraction.py",
		help="WSI processing entrypoint script.",
	)
	parser.add_argument(
		"--config",
		default="wsi_preprocessing_v2/configs/default.yaml",
		help="Config used by the processing pipeline.",
	)
	parser.add_argument(
		"--output-dir",
		default="",
		help="Optional override for output h5 directory.",
	)
	parser.add_argument(
		"--skip-existing",
		action="store_true",
		help="Skip slides that already have outputs.",
	)
	parser.add_argument(
		"--keep",
		action="store_true",
		help="Do not delete slides after processing.",
	)
	parser.add_argument(
		"--max-slides",
		type=int,
		default=None,
		help="Optional limit for debugging.",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Print commands without executing.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	manifest_path = Path(args.manifest)
	gdc_client = Path(args.gdc_client)
	raw_dir = Path(args.raw_dir)
	processor_script = Path(args.processor)
	config_path = Path(args.config)
	output_dir = Path(args.output_dir) if args.output_dir else None

	records = read_manifest(manifest_path)
	run_gdc_download(gdc_client, manifest_path, raw_dir, args.dry_run)

	processed = 0
	for record in records:
		slide_path = find_slide_path(raw_dir, record)
		if slide_path is None:
			print(f"[WARN] Missing slide for {record.file_id} ({record.filename})")
			continue

		run_processing(
			processor_script,
			config_path,
			slide_path,
			output_dir,
			args.skip_existing,
			args.dry_run,
		)

		if not args.keep:
			delete_slide(slide_path, args.dry_run)

		processed += 1
		if args.max_slides is not None and processed >= args.max_slides:
			break

	print(f"[DONE] Processed {processed} slide(s) from {manifest_path.name}")


if __name__ == "__main__":
	main()

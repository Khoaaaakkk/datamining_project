from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List


def find_manifests(manifests_dir: Path, pattern: str) -> List[Path]:
	if not manifests_dir.exists():
		raise FileNotFoundError(f"Manifests directory not found: {manifests_dir}")
	manifests = sorted(manifests_dir.glob(pattern))
	if not manifests:
		raise FileNotFoundError(f"No manifests found in {manifests_dir} with pattern '{pattern}'")
	return manifests


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Execute multiple batch manifests sequentially.")
	parser.add_argument(
		"--manifests-dir",
		default="data/reference/manifests",
		help="Folder containing batch manifests.",
	)
	parser.add_argument(
		"--pattern",
		default="batch_*.txt",
		help="Glob pattern for manifest files.",
	)
	parser.add_argument(
		"--start",
		type=int,
		default=1,
		help="Start index (1-based) in the sorted manifest list.",
	)
	parser.add_argument(
		"--end",
		type=int,
		default=None,
		help="End index (inclusive) in the sorted manifest list.",
	)
	parser.add_argument(
		"--batch-pipeline",
		default="wsi_preprocess/data_loader/batch_pipeline.py",
		help="Path to the single-batch pipeline script.",
	)
	parser.add_argument(
		"--config",
		default="configs/default.yaml",
		help="Config file passed to the batch pipeline.",
	)
	parser.add_argument(
		"--raw-dir",
		default="data/raw_wsi",
		help="Download output directory for raw WSI files.",
	)
	parser.add_argument(
		"--gdc-client",
		default="tools/gdc-client.exe",
		help="Path to gdc-client.exe.",
	)
	parser.add_argument(
		"--processor",
		default="wsi_preprocess/wsi_feature_extraction.py",
		help="WSI processing entrypoint script.",
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
		"--dry-run",
		action="store_true",
		help="Print commands without executing.",
	)
	return parser.parse_args()


def resolve_legacy_path(path: Path, legacy_root: str, new_root: str) -> Path:
	if path.exists():
		return path
	path_str = str(path)
	if legacy_root in path_str:
		candidate = Path(path_str.replace(legacy_root, new_root))
		if candidate.exists():
			print(f"[WARN] Path not found, using {candidate} instead of {path}")
			return candidate
	return path


def main() -> None:
	args = parse_args()
	manifests_dir = Path(args.manifests_dir)
	batch_pipeline = Path(args.batch_pipeline)
	batch_pipeline = resolve_legacy_path(
		batch_pipeline,
		legacy_root="wsi_preprocessing_v2",
		new_root="wsi_preprocess",
	)
	config_path = resolve_legacy_path(
		Path(args.config),
		legacy_root="wsi_preprocessing_v2",
		new_root="configs",
	)
	processor_path = resolve_legacy_path(
		Path(args.processor),
		legacy_root="wsi_preprocessing_v2",
		new_root="wsi_preprocess",
	)

	manifests = find_manifests(manifests_dir, args.pattern)
	start_idx = max(args.start - 1, 0)
	end_idx = args.end if args.end is not None else len(manifests)
	selected = manifests[start_idx:end_idx]
	if not selected:
		raise ValueError("No manifests selected. Check --start/--end or pattern.")

	for manifest in selected:
		command = [
			sys.executable,
			str(batch_pipeline),
			"--manifest",
			str(manifest),
			"--raw-dir",
			args.raw_dir,
			"--gdc-client",
			args.gdc_client,
			"--processor",
			str(processor_path),
			"--config",
			str(config_path),
		]
		if args.output_dir:
			command.extend(["--output-dir", args.output_dir])
		if args.skip_existing:
			command.append("--skip-existing")
		if args.keep:
			command.append("--keep")
		if args.dry_run:
			command.append("--dry-run")

		print(f"[INFO] Run batch: {manifest.name}")
		print(f"[INFO] Command: {' '.join(command)}")
		if args.dry_run:
			continue

		result = __import__("subprocess").run(command, check=False)
		if result.returncode != 0:
			raise RuntimeError(f"Batch failed ({manifest.name}) with code {result.returncode}")

	print(f"[DONE] Finished {len(selected)} batch(es)")


if __name__ == "__main__":
	main()

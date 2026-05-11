from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List


def build_command(gdc_client: Path, manifest: Path, output_dir: Path) -> List[str]:
	return [
		str(gdc_client),
		"download",
		"-m",
		str(manifest),
		"-d",
		str(output_dir),
	]


def parse_args():
	parser = argparse.ArgumentParser(description="Download GDC data using gdc-client and a manifest file")
	parser.add_argument(
		"--manifest",
		type=str,
		default="data/reference/manifests/batch_0001.txt",
		help="Path to GDC manifest file",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="data/raw_wsi",
		help="Directory to save downloaded files",
	)
	parser.add_argument(
		"--gdc-client",
		type=str,
		default="tools/gdc-client.exe",
		help="Path to gdc-client executable",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	repo_root = Path(__file__).resolve().parents[1]
	manifest = (repo_root / args.manifest).resolve()
	output_dir = (repo_root / args.output_dir).resolve()
	gdc_client = (repo_root / args.gdc_client).resolve()

	if not gdc_client.exists():
		raise FileNotFoundError(f"gdc-client not found: {gdc_client}")
	if not manifest.exists():
		raise FileNotFoundError(f"Manifest not found: {manifest}")

	output_dir.mkdir(parents=True, exist_ok=True)
	command = build_command(gdc_client, manifest, output_dir)

	print("[INFO] Running:", " ".join(command))
	process = subprocess.run(command, check=False)
	if process.returncode != 0:
		raise RuntimeError(f"gdc-client failed with exit code {process.returncode}")

	print(f"[DONE] Downloaded files to {output_dir}")


if __name__ == "__main__":
	main()

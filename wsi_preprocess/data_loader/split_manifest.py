from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd
import requests

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"


@dataclass
class FileRecord:
	file_id: str
	file_name: str
	md5sum: str
	file_size: int
	state: str


def chunk_list(items: Sequence[str], batch_size: int) -> Iterable[List[str]]:
	for idx in range(0, len(items), batch_size):
		yield list(items[idx : idx + batch_size])


def build_filters(submitter_ids: List[str]) -> Dict:
	return {
		"op": "and",
		"content": [
			{
				"op": "in",
				"content": {"field": "cases.submitter_id", "value": submitter_ids},
			},
			{
				"op": "in",
				"content": {"field": "data_category", "value": ["Biospecimen"]},
			},
			{
				"op": "in",
				"content": {"field": "data_type", "value": ["Slide Image"]},
			},
			{
				"op": "in",
				"content": {"field": "data_format", "value": ["SVS"]},
			},
		],
	}


def fetch_files(submitter_ids: List[str], size: int = 2000) -> List[FileRecord]:
	filters = build_filters(submitter_ids)
	params = {
		"filters": json.dumps(filters),
		"fields": "file_id,file_name,md5sum,file_size,state",
		"format": "JSON",
		"size": str(size),
	}
	response = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=60)
	response.raise_for_status()
	payload = response.json()
	files: List[FileRecord] = []
	for hit in payload.get("data", {}).get("hits", []):
		files.append(
			FileRecord(
				file_id=str(hit.get("file_id", "")),
				file_name=str(hit.get("file_name", "")),
				md5sum=str(hit.get("md5sum", "")),
				file_size=int(hit.get("file_size", 0) or 0),
				state=str(hit.get("state", "")),
			)
		)
	return [
		f
		for f in files
		if f.file_id and f.file_name and "-DX" in f.file_name.upper()
	]


def write_manifest(path: Path, records: List[FileRecord]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f, delimiter="\t")
		writer.writerow(["id", "filename", "md5", "size", "state"])
		for record in records:
			writer.writerow(
				[record.file_id, record.file_name, record.md5sum, record.file_size, record.state]
			)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Create GDC manifest batch files from submitter_id list."
	)
	parser.add_argument(
		"--csv",
		default="data/reference/Final_Matched_Clinical.csv",
		help="CSV containing submitter_id column.",
	)
	parser.add_argument(
		"--out-dir",
		default="data/reference/manifests",
		help="Output directory for batch_XXXX.txt files.",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=5,
		help="Number of submitter IDs per batch.",
	)
	parser.add_argument(
		"--start-index",
		type=int,
		default=1,
		help="Starting batch index for filenames.",
	)
	parser.add_argument(
		"--max-batches",
		type=int,
		default=None,
		help="Optional limit on number of batches written.",
	)
	parser.add_argument(
		"--max-files",
		type=int,
		default=None,
		help="Optional limit on files per manifest.",
	)
	parser.add_argument(
		"--cohort",
		default="",
		help="Optional Study_Cohort value to filter (e.g., BLCA).",
	)
	parser.add_argument(
		"--max-submitters",
		type=int,
		default=None,
		help="Optional limit on number of submitter_id used.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	csv_path = Path(args.csv)
	out_dir = Path(args.out_dir)

	if not csv_path.exists():
		raise FileNotFoundError(f"CSV not found: {csv_path}")

	df = pd.read_csv(csv_path)
	if "submitter_id" not in df.columns:
		raise ValueError("CSV must contain a submitter_id column")
	if args.cohort:
		if "Study_Cohort" not in df.columns:
			raise ValueError("CSV must contain a Study_Cohort column to filter by cohort")
		df = df[df["Study_Cohort"].astype(str) == args.cohort]

	submitter_ids = [str(v).strip() for v in df["submitter_id"].dropna().tolist()]
	if args.max_submitters is not None:
		submitter_ids = submitter_ids[: args.max_submitters]
	if not submitter_ids:
		raise ValueError("No submitter IDs found in CSV")

	batch_index = int(args.start_index)
	written = 0
	for batch_ids in chunk_list(submitter_ids, args.batch_size):
		if args.max_batches is not None and written >= args.max_batches:
			break

		records = fetch_files(batch_ids)
		if args.max_files is not None:
			records = records[: args.max_files]

		manifest_path = out_dir / f"batch_{batch_index:04d}.txt"
		write_manifest(manifest_path, records)
		print(f"[DONE] {manifest_path} -> {len(records)} file(s)")

		batch_index += 1
		written += 1

	print(f"[DONE] Wrote {written} batch manifest(s) to {out_dir}")


if __name__ == "__main__":
	main()

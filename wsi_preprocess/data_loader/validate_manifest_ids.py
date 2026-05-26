from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd

SUBMITTER_PATTERN = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)


@dataclass
class ValidationResult:
	manifest: str
	files_checked: int
	unique_submitter_ids: int
	unknown_submitter_ids: List[str]
	missing_in_manifest: List[str]


def parse_submitter_id(filename: str) -> Optional[str]:
	match = SUBMITTER_PATTERN.match(filename)
	if match:
		return match.group(1)
	parts = filename.split("-")
	if len(parts) >= 3 and parts[0].upper() == "TCGA":
		return "-".join(parts[:3])
	return None


def read_manifest_submitter_ids(manifest_path: Path) -> Set[str]:
	ids: Set[str] = set()
	with manifest_path.open("r", encoding="utf-8") as f:
		reader = csv.DictReader(f, delimiter="\t")
		for row in reader:
			filename = str(row.get("filename", "")).strip()
			submitter = parse_submitter_id(filename)
			if submitter:
				ids.add(submitter)
	return ids


def load_csv_submitter_ids(csv_path: Path) -> Set[str]:
	df = pd.read_csv(csv_path)
	if "submitter_id" not in df.columns:
		raise ValueError("CSV must contain a submitter_id column")
	return {str(v).strip() for v in df["submitter_id"].dropna().tolist() if str(v).strip()}


def find_manifests(manifests_dir: Path, pattern: str) -> List[Path]:
	if not manifests_dir.exists():
		raise FileNotFoundError(f"Manifests directory not found: {manifests_dir}")
	manifests = sorted(manifests_dir.glob(pattern))
	if not manifests:
		raise FileNotFoundError(f"No manifests found in {manifests_dir} with pattern '{pattern}'")
	return manifests


def validate_manifest(manifest_path: Path, csv_ids: Set[str]) -> ValidationResult:
	manifest_ids = read_manifest_submitter_ids(manifest_path)
	unknown = sorted([v for v in manifest_ids if v not in csv_ids])
	missing = sorted([v for v in csv_ids if v not in manifest_ids])
	return ValidationResult(
		manifest=str(manifest_path),
		files_checked=len(manifest_ids),
		unique_submitter_ids=len(manifest_ids),
		unknown_submitter_ids=unknown,
		missing_in_manifest=missing,
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Validate submitter_id in manifests against Final_Matched_Clinical.csv"
	)
	parser.add_argument(
		"--csv",
		default="data/reference/Final_Matched_Clinical.csv",
		help="CSV with submitter_id column.",
	)
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
		"--report",
		default="",
		help="Optional JSON output file to store validation results.",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Optional limit on number of manifests checked.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	csv_path = Path(args.csv)
	manifests_dir = Path(args.manifests_dir)
	manifests = find_manifests(manifests_dir, args.pattern)
	if args.limit is not None:
		manifests = manifests[: args.limit]

	csv_ids = load_csv_submitter_ids(csv_path)
	results: List[Dict] = []
	for manifest in manifests:
		result = validate_manifest(manifest, csv_ids)
		results.append(result.__dict__)
		print(
			f"[CHECK] {Path(result.manifest).name}: {result.unique_submitter_ids} IDs, "
			f"unknown={len(result.unknown_submitter_ids)}, missing={len(result.missing_in_manifest)}"
		)

	if args.report:
		report_path = Path(args.report)
		report_path.parent.mkdir(parents=True, exist_ok=True)
		report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
		print(f"[DONE] Wrote report -> {report_path}")


if __name__ == "__main__":
	main()

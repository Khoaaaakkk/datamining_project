from __future__ import annotations

import argparse
import csv
import sys
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
	from .fusion_main import MultimodalFusion
except ImportError:  # pragma: no cover - fallback for script execution
	from fusion_feature.fusion_main import MultimodalFusion


GenomicData = Union[Dict[str, Any], torch.Tensor]


def _load_genomic_features(pkl_path: Path) -> GenomicData:
	with pkl_path.open("rb") as handle:
		data = pickle.load(handle)
	if isinstance(data, dict):
		return data
	if isinstance(data, torch.Tensor):
		return data
	raise ValueError(
		"Expected a dict mapping submitter_id to features or a torch.Tensor"
	)


def _load_submitter_ids(csv_path: Path) -> List[str]:
	if not csv_path.exists():
		raise FileNotFoundError(f"Submitter ID CSV not found: {csv_path}")
	with csv_path.open("r", newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle)
		if "submitter_id" not in (reader.fieldnames or []):
			raise ValueError("Submitter ID CSV must contain 'submitter_id' column")
		return [str(row["submitter_id"]).strip() for row in reader]


def _load_submitter_ids_from_manifest(manifest_path: Path) -> Dict[str, str]:
	if not manifest_path.exists():
		raise FileNotFoundError(f"Manifest not found: {manifest_path}")
	submitter_ids: Dict[str, str] = {}
	with manifest_path.open("r", encoding="utf-8") as handle:
		for line_index, line in enumerate(handle):
			if line_index == 0:
				continue
			parts = line.strip().split("\t")
			if len(parts) < 2:
				continue
			filename = parts[1]
			if len(filename) < 12:
				continue
			submitter_id = filename[:12]
			if submitter_id not in submitter_ids:
				submitter_ids[submitter_id] = filename
	if not submitter_ids:
		raise ValueError("No submitter IDs found in manifest")
	return submitter_ids


def _resolve_submitter_id(
	features: GenomicData,
	submitter_id: str,
	submitter_ids: Optional[List[str]],
) -> Tuple[str, Any]:
	if isinstance(features, dict):
		if submitter_id in features:
			return submitter_id, features[submitter_id]
		raise KeyError(
			f"Submitter id '{submitter_id}' not found in genomic features"
		)

	if submitter_ids is None:
		raise ValueError(
			"Submitter ID CSV is required when genomic features are a tensor"
		)
	if submitter_id not in submitter_ids:
		raise KeyError(
			f"Submitter id '{submitter_id}' not found in submitter ID CSV"
		)
	index = submitter_ids.index(submitter_id)
	return submitter_id, features[index]


def _resolve_co_attention_path(
	co_attention_dir: Path,
	submitter_id: str,
	filename: str,
) -> Path:
	name_prefix = filename[12:24]
	candidate = co_attention_dir / f"{submitter_id}{name_prefix}_co_attention.pt"
	if candidate.exists():
		return candidate
	fallbacks = list(co_attention_dir.glob(f"{submitter_id}*_co_attention.pt"))
	if fallbacks:
		return fallbacks[0]
	raise FileNotFoundError(
		f"No co-attention output found for submitter id '{submitter_id}'"
	)


def _to_batch(tensor: torch.Tensor) -> torch.Tensor:
	if tensor.dim() == 2:
		return tensor.unsqueeze(0)
	return tensor


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run multimodal fusion for submitters in a batch manifest."
	)
	parser.add_argument(
		"--manifest",
		type=Path,
		required=True,
		help="Path to batch_000x.txt manifest file.",
	)
	parser.add_argument(
		"--genomic-pkl",
		type=Path,
		default=Path("data/csv/combined_genomic_features.pkl"),
		help="Path to combined_genomic_features.pkl",
	)
	parser.add_argument(
		"--submitter-ids-csv",
		type=Path,
		default=Path("data/reference/Final_Matched_Clinical.csv"),
		help="CSV containing submitter_id column for tensor-based genomic data.",
	)
	parser.add_argument(
		"--co-attention-dir",
		type=Path,
		default=Path("data/outputs/co_attention"),
		help="Directory containing co-attention outputs.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("data/outputs/fusion"),
		help="Directory to save fusion feature outputs.",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="cuda" if torch.cuda.is_available() else "cpu",
		help="Device to run on (cpu or cuda).",
	)
	parser.add_argument(
		"--gpu-index",
		type=int,
		default=0,
		help="GPU index to use when device is cuda.",
	)
	parser.add_argument(
		"--max-submitters",
		type=int,
		default=None,
		help="Optional limit on number of submitters to process.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	manifest_map = _load_submitter_ids_from_manifest(args.manifest)
	submitter_ids = list(manifest_map.keys())
	if args.max_submitters is not None:
		submitter_ids = submitter_ids[: args.max_submitters]

	genomic_data = _load_genomic_features(args.genomic_pkl)
	all_submitter_ids = None
	if not isinstance(genomic_data, dict):
		all_submitter_ids = _load_submitter_ids(args.submitter_ids_csv)
		if len(all_submitter_ids) != genomic_data.shape[0]:
			raise ValueError(
				"Submitter ID count does not match genomic tensor length"
			)

	device = torch.device(args.device)
	if device.type == "cuda":
		torch.cuda.set_device(args.gpu_index)
		torch.backends.cudnn.benchmark = True

	model = MultimodalFusion().to(device)
	model.eval()

	args.output_dir.mkdir(parents=True, exist_ok=True)
	for submitter_id in submitter_ids:
		filename = manifest_map[submitter_id]
		co_attention_path = _resolve_co_attention_path(
			args.co_attention_dir,
			submitter_id,
			filename,
		)
		payload = torch.load(co_attention_path, map_location="cpu")
		guided_wsi_embeds = _to_batch(payload["output"])
		_, genomic_features = _resolve_submitter_id(
			genomic_data,
			submitter_id,
			all_submitter_ids,
		)
		q_gene_features = _to_batch(
			genomic_features
			if isinstance(genomic_features, torch.Tensor)
			else torch.tensor(genomic_features, dtype=torch.float32)
		)

		guided_wsi_embeds = guided_wsi_embeds.to(device)
		q_gene_features = q_gene_features.to(device)

		with torch.no_grad():
			fusion_feature = model(guided_wsi_embeds, q_gene_features)

		output_name = f"{Path(filename).stem}_fusion_feature.pt"
		output_path = args.output_dir / output_name
		torch.save(fusion_feature.cpu(), output_path)
		print(f"Submitter ID: {submitter_id}")
		print(f"Fusion shape: {tuple(fusion_feature.shape)}")
		print(f"Saved to: {output_path}")


if __name__ == "__main__":
	main()

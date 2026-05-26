from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

try:
	from .train_fc_projection import LinearAutoEncoder
except ImportError:  # pragma: no cover - fallback for script execution
	from co_attention.train_fc_projection import LinearAutoEncoder

try:
	from .co_attention import GenomicGuidedCoAttention
except ImportError:  # pragma: no cover - fallback for script execution
	from co_attention import GenomicGuidedCoAttention


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


def _load_submitter_ids_from_manifest(manifest_path: Path) -> Dict[str, Dict[str, str]]:
	if not manifest_path.exists():
		raise FileNotFoundError(f"Manifest not found: {manifest_path}")
	submitter_ids: Dict[str, Dict[str, str]] = {}
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
				submitter_ids[submitter_id] = {
					"filename": filename,
					"slide_id": filename[:24] if len(filename) >= 24 else filename,
				}
	if not submitter_ids:
		raise ValueError("No submitter IDs found in manifest")
	return submitter_ids


def _resolve_submitter_id(
	features: GenomicData,
	submitter_id: Optional[str],
	submitter_ids: Optional[List[str]],
) -> Tuple[str, Any]:
	if isinstance(features, dict):
		if submitter_id:
			if submitter_id in features:
				return submitter_id, features[submitter_id]
			raise KeyError(
				f"Submitter id '{submitter_id}' not found in genomic features"
			)
		first_id = next(iter(features.keys()))
		return first_id, features[first_id]

	if submitter_ids is None:
		raise ValueError(
			"Submitter ID CSV is required when genomic features are a tensor"
		)
	if submitter_id is None:
		submitter_id = submitter_ids[0]
	if submitter_id not in submitter_ids:
		raise KeyError(
			f"Submitter id '{submitter_id}' not found in submitter ID CSV"
		)
	index = submitter_ids.index(submitter_id)
	return submitter_id, features[index]


def _find_h5_file(h5_dir: Path, submitter_id: str, slide_id: Optional[str]) -> Path:
	patterns = []
	if slide_id:
		patterns.append(f"{slide_id}*.h5")
	patterns.append(f"{submitter_id}*.h5")
	matches: List[Path] = []
	for pattern in patterns:
		matches.extend(sorted(h5_dir.glob(pattern)))
	dx_matches = [path for path in matches if "-DX" in path.name.upper()]
	if dx_matches:
		return dx_matches[0]
	if not matches:
		raise FileNotFoundError(
			f"No .h5 files found for submitter id '{submitter_id}' in {h5_dir}"
		)
	return matches[0]


def _find_first_dataset(group: h5py.Group) -> Tuple[str, h5py.Dataset]:
	for key in group.keys():
		obj = group[key]
		if isinstance(obj, h5py.Dataset):
			return key, obj
		if isinstance(obj, h5py.Group):
			child_key, dataset = _find_first_dataset(obj)
			return f"{key}/{child_key}", dataset
	raise ValueError("No dataset found in H5 file")


def _load_patch_features(
	h5_path: Path,
	dataset_name: Optional[str],
	max_patches: Optional[int],
) -> Tuple[torch.Tensor, str]:
	with h5py.File(h5_path, "r") as handle:
		if dataset_name:
			if dataset_name not in handle:
				raise KeyError(
					f"Dataset '{dataset_name}' not found in {h5_path.name}"
				)
			dataset = handle[dataset_name]
			resolved_name = dataset_name
		else:
			if "features" in handle:
				resolved_name = "features"
				dataset = handle[resolved_name]
			else:
				resolved_name, dataset = _find_first_dataset(handle)
		data = dataset[()]
	if max_patches is not None:
		data = data[:max_patches]
	if data.ndim != 2:
		raise ValueError(
			f"Expected patch features to be 2D (N, D), got {data.shape}"
		)
	return torch.tensor(data, dtype=torch.float32), resolved_name


def _to_tensor(data: Any) -> torch.Tensor:
	if isinstance(data, torch.Tensor):
		return data.float()
	return torch.tensor(data, dtype=torch.float32)


def _build_projection(
	input_dim: int,
	output_dim: int,
	weights_path: Optional[Path],
	device: torch.device,
) -> nn.Module:
	if weights_path is None:
		layer = nn.Linear(input_dim, output_dim).to(device)
		layer.eval()
		return layer

	state = torch.load(weights_path, map_location=device)
	if any(key.startswith("encoder.") for key in state.keys()):
		model = LinearAutoEncoder(input_dim, output_dim).to(device)
		model.load_state_dict(state)
		model.eval()
		return model.encoder
	layer = nn.Linear(input_dim, output_dim).to(device)
	layer.load_state_dict(state)
	layer.eval()
	return layer


def run_co_attention(
	genomic_features: Any,
	patch_features: torch.Tensor,
	fc_weights: Optional[Path],
	device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
	q = _to_tensor(genomic_features).to(device)
	# Project patch features to the same dimension as genomic features (N, 1536) -> (N, 512)
	patch_features = patch_features.to(device)
	if patch_features.shape[-1] != q.shape[-1]:
		proj = _build_projection(
			patch_features.shape[-1],
			q.shape[-1],
			fc_weights,
			device,
		)
		with torch.no_grad():
			patch_features = proj(patch_features)
	k = patch_features
	v = patch_features

	model = GenomicGuidedCoAttention(dim=q.shape[-1]).to(device)
	model.eval()
	with torch.no_grad():
		output, attn = model(q, k, v)
	return output.cpu(), attn.cpu()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run genomic-guided co-attention for a submitter id.")
	parser.add_argument(
		"--genomic-pkl",
		type=Path,
		default=Path("data/csv/combined_genomic_features.pkl"),
		help="Path to combined_genomic_features.pkl",
	)
	parser.add_argument(
		"--h5-dir",
		type=Path,
		default=Path("data/h5_features"),
		help="Directory containing H5 feature files",
	)
	parser.add_argument(
		"--submitter-id",
		type=str,
		default=None,
		help="Submitter ID (12 chars) to run. Defaults to first entry in pickle.",
	)
	parser.add_argument(
		"--manifest",
		type=Path,
		default=None,
		help="Optional batch_000x.txt manifest to run co-attention for all submitters.",
	)
	parser.add_argument(
		"--submitter-ids-csv",
		type=Path,
		default=Path("data/reference/Final_Matched_Clinical.csv"),
		help=(
			"CSV containing submitter_id column for mapping tensor-based genomic data."
		),
	)
	parser.add_argument(
		"--dataset",
		type=str,
		default=None,
		help="Optional dataset name inside the H5 file. Defaults to first dataset.",
	)
	parser.add_argument(
		"--max-patches",
		type=int,
		default=200000,
		help="Max number of patch embeddings to load.",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="cuda" if torch.cuda.is_available() else "cpu",
		help="Device to run on (cpu or cuda).",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("data/outputs/co_attention"),
		help="Directory to save outputs (.pt).",
	)
	parser.add_argument(
		"--fc-weights",
		type=Path,
		# default=Path("data/outputs/fc_projection/fc_weights.pt"),
		default=Path("data/outputs/fc_projection/fc_autoencoder.pt"),
		help=(
			"Optional path to Linear layer weights (state_dict) to project patch features."
		),
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	genomic_data = _load_genomic_features(args.genomic_pkl)
	all_submitter_ids = None
	if not isinstance(genomic_data, dict):
		all_submitter_ids = _load_submitter_ids(args.submitter_ids_csv)
		if len(all_submitter_ids) != genomic_data.shape[0]:
			raise ValueError(
				"Submitter ID count does not match genomic tensor length"
			)

	if args.manifest:
		manifest_map = _load_submitter_ids_from_manifest(args.manifest)
		submitter_ids = list(manifest_map.keys())
	else:
		manifest_map = {}
		submitter_ids = [
			_resolve_submitter_id(
				genomic_data,
				args.submitter_id,
				all_submitter_ids,
			)[0]
		]

	device = torch.device(args.device)
	args.output_dir.mkdir(parents=True, exist_ok=True)
	for submitter_id in submitter_ids:
		_, genomic_features = _resolve_submitter_id(
			genomic_data,
			submitter_id,
			all_submitter_ids,
		)
		manifest_entry = manifest_map.get(submitter_id)
		slide_id = manifest_entry["slide_id"] if manifest_entry else None
		h5_path = _find_h5_file(args.h5_dir, submitter_id, slide_id)
		patch_features, dataset_name = _load_patch_features(
			h5_path, args.dataset, args.max_patches
		)
		output, attn = run_co_attention(
			genomic_features,
			patch_features,
			args.fc_weights,
			device,
		)
		attn_flat = attn
		if attn_flat.dim() == 3:
			attn_flat = attn_flat.mean(dim=(0, 1))
		elif attn_flat.dim() == 2:
			attn_flat = attn_flat.mean(dim=0)
		elif attn_flat.dim() != 1:
			raise ValueError("attention weights must be 1D/2D/3D")

		if submitter_id in manifest_map:
			slide_id = manifest_map[submitter_id]["slide_id"]
			output_name = f"{slide_id}_co_attention.pt"
		else:
			output_name = f"{submitter_id}_co_attention.pt"
		output_path = args.output_dir / output_name
		payload = {
			"submitter_id": submitter_id,
			"h5_file": h5_path.name,
			"dataset": dataset_name,
			"fc_weights": str(args.fc_weights) if args.fc_weights else None,
			"output": output,
			"attention": attn,
			"attention_flat": attn_flat.cpu(),
		}
		torch.save(payload, output_path)

		print(f"Submitter ID: {submitter_id}")
		print(f"H5 file: {h5_path.name} (dataset: {dataset_name})")
		print(f"Output shape: {tuple(output.shape)}")
		print(f"Attention shape: {tuple(attn.shape)}")
		print(f"Saved to: {output_path}")


if __name__ == "__main__":
	main()

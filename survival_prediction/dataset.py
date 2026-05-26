from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import h5py
import torch
from torch.utils.data import Dataset


@dataclass
class PatientSample:
    submitter_id: str
    # h5_features: torch.Tensor
    # co_attention: torch.Tensor
    fusion: torch.Tensor


class SurvivalDataset(Dataset):
    """Dataset that loads WSI, co-attention, and fusion features per patient.

    Expected files (skip if any missing):
    - h5_dir/{submitter_id}.h5 (or prefix match)
    - co_attention_dir/{submitter_id}.pt
    - fusion_dir/{submitter_id}.pt
    """

    def __init__(
        self,
        manifest_path: Path,
        # h5_dir: Path,
        # co_attention_dir: Path,
        fusion_dir: Path,
        dataset_key: str = "features",
    ) -> None:
        self.manifest_path = manifest_path
        # self.h5_dir = self._resolve_h5_dir(h5_dir)
        # self.co_attention_dir = co_attention_dir
        self.fusion_dir = fusion_dir
        self.dataset_key = dataset_key
        self.submitter_ids = self._load_manifest(manifest_path)
        self.samples: List[PatientSample] = []
        self._load_all_samples()

    @staticmethod
    def _resolve_h5_dir(h5_dir: Path) -> Path:
        if h5_dir.exists():
            return h5_dir
        fallback = h5_dir.parent / "h5_features"
        if fallback.exists():
            print(f"[WARN] {h5_dir} not found. Using {fallback} instead.")
            return fallback
        return h5_dir

    @staticmethod
    def _load_manifest(path: Path) -> List[str]:
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        ids: List[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                candidate = line.strip()
                if not candidate:
                    continue
                if candidate.lower().startswith("id"):
                    continue
                if "\t" in candidate:
                    parts = candidate.split("\t")
                    if len(parts) >= 2:
                        filename = parts[1]
                        if len(filename) >= 12:
                            ids.append(filename[:12])
                            continue
                ids.append(candidate)
        if not ids:
            raise ValueError("Manifest contains no submitter_id values")
        return ids

    def _resolve_h5_path(self, submitter_id: str) -> Optional[Path]:
        matches = sorted(self.h5_dir.glob(f"{submitter_id}*.h5"))
        return matches[0] if matches else None

    def _resolve_pt_path(self, directory: Path, submitter_id: str) -> Optional[Path]:
        matches = sorted(directory.glob(f"{submitter_id}*.pt"))
        return matches[0] if matches else None

    def _load_h5_features(self, path: Path) -> torch.Tensor:
        with h5py.File(path, "r") as handle:
            if self.dataset_key in handle:
                data = handle[self.dataset_key][()]
            else:
                data = next(iter(handle.values()))[()]
        data = torch.tensor(data, dtype=torch.float32)
        if data.ndim == 2:
            return data.mean(dim=0)
        return data.flatten().float()

    @staticmethod
    def _load_pt_tensor(path: Path, key: Optional[str] = None) -> torch.Tensor:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            if key and key in payload:
                tensor = payload[key]
            elif "fusion_feature" in payload:
                tensor = payload["fusion_feature"]
            elif "output" in payload:
                tensor = payload["output"]
            else:
                tensor = next(iter(payload.values()))
        else:
            tensor = payload
        return tensor.float().mean(dim=0) if tensor.ndim > 1 else tensor.float()

    def _load_all_samples(self) -> None:
        for submitter_id in self.submitter_ids:
            # h5_path = self._resolve_h5_path(submitter_id)
            # co_att_path = self._resolve_pt_path(self.co_attention_dir, submitter_id)
            fusion_path = self._resolve_pt_path(self.fusion_dir, submitter_id)
            if not fusion_path:
                print(
                    f"[WARN] Missing files for {submitter_id}. "
                    f"fusion={bool(fusion_path)}"
                )
                continue
            # if not h5_path or not co_att_path or not fusion_path:
            #     print(
            #         f"[WARN] Missing files for {submitter_id}. "
            #         f"h5={bool(h5_path)} co_att={bool(co_att_path)} fusion={bool(fusion_path)}"
            #     )
            #     continue
            # h5_features = self._load_h5_features(h5_path)
            # co_attention = self._load_pt_tensor(co_att_path, key="attention_flat")
            fusion = self._load_pt_tensor(fusion_path)
            self.samples.append(
                PatientSample(
                    submitter_id=submitter_id,
                    # h5_features=h5_features,
                    # co_attention=co_attention,
                    fusion=fusion,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "submitter_id": sample.submitter_id,
            # "h5_features": sample.h5_features,
            # "co_attention": sample.co_attention,
            "fusion": sample.fusion,
        }

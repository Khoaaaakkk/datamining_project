from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.data_loader.h5_dataset import H5BagDataset
from src.models.feature_extractor import get_device
from src.models.mil_classifier import AttentionMIL


def parse_args():
	p = argparse.ArgumentParser(description="Evaluate Attention MIL model")
	p.add_argument("--config", default="configs/default.yaml")
	p.add_argument("--checkpoint", default="experiments/checkpoints/best_mil.pth")
	return p.parse_args()


def main():
	args = parse_args()
	cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
	data_cfg = cfg.get("data", {})
	tr_cfg = cfg.get("training", {})

	dataset = H5BagDataset(
		h5_dir=data_cfg.get("h5_features_dir", "data/h5_features"),
		labels_csv=data_cfg.get("labels_file", "data/labels/labels.csv"),
	)
	loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda b: b)

	device = get_device(tr_cfg.get("device", "cuda"))
	expected_in_dim = int(tr_cfg.get("feature_dim", 1536))
	model = AttentionMIL(
		in_dim=expected_in_dim,
		attn_dim=int(tr_cfg.get("attn_dim", 256)),
		num_classes=int(tr_cfg.get("num_classes", 12)),
	).to(device)

	ckpt = Path(args.checkpoint)
	if not ckpt.exists():
		raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

	model.load_state_dict(torch.load(ckpt, map_location=device))
	model.eval()

	y_true, y_pred = [], []
	with torch.no_grad():
		for sample_list in loader:
			sample = sample_list[0]
			label = int(sample["label"].item())
			if label < 0:
				continue
			feats = sample["features"].to(device)
			if feats.shape[-1] != expected_in_dim:
				raise ValueError(
					"Feature dimension mismatch: "
					f"dataset has {feats.shape[-1]} but training.feature_dim={expected_in_dim}."
				)
			logits, _ = model(feats)
			pred = int(torch.argmax(logits, dim=-1).item())
			y_true.append(label)
			y_pred.append(pred)

	if len(y_true) == 0:
		print("No labeled samples found for evaluation.")
		return

	acc = accuracy_score(y_true, y_pred)
	f1 = f1_score(y_true, y_pred, average="weighted")
	print(f"Accuracy: {acc:.4f}")
	print(f"F1-weighted: {f1:.4f}")


if __name__ == "__main__":
	main()

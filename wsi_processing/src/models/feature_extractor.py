from __future__ import annotations

import torch
import torch.nn as nn

try:
	import timm
except Exception:  # pragma: no cover - optional runtime dependency
	timm = None


def build_feature_extractor(
	model_name: str = "swin_large_patch4_window7_224",
	pretrained: bool = True,
	image_size: int = 224,
) -> tuple[nn.Module, int]:
	"""Create Swin-L backbone from timm for patch embeddings."""
	name = model_name.lower()
	if name != "swin_large_patch4_window7_224":
		raise ValueError(f"Unsupported model_name: {model_name}")

	if timm is None:
		raise ValueError(
			"timm is required for Swin-L. Install timm>=1.0.0 to use 'swin_large_patch4_window7_224'."
		)

	model = timm.create_model(
		"swin_large_patch4_window7_224",
		pretrained=pretrained,
		num_classes=0,
		global_pool="avg",
		img_size=image_size,
	)
	out_dim = int(getattr(model, "num_features", 1536))
	model.eval()
	return model, out_dim


def get_device(device: str | None = None) -> torch.device:
	if device:
		requested = torch.device(device)
		if requested.type == "cuda" and not torch.cuda.is_available():
			print("[WARN] CUDA requested but unavailable, falling back to CPU.")
			return torch.device("cpu")
		return requested
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")

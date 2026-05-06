from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

try:
	import timm
except Exception:  # pragma: no cover - optional runtime dependency
	timm = None


def build_feature_extractor(
	model_name: str = "swin_large_patch4_window7_224",
	pretrained: bool = True,
	image_size: int = 224,
) -> tuple[nn.Module, int]:
	"""
	Build image feature extractor and return (model, output_dim).

	Supported model_name:
	- "swin_large_patch4_window7_224" -> output dim 1536
	- "resnet50" -> output dim 2048
	"""
	name = model_name.lower()

	if name == "swin_large_patch4_window7_224":
		if hasattr(models, "swin_large"):
			weights = models.Swin_L_Weights.IMAGENET1K_V1 if pretrained else None
			model = models.swin_large(weights=weights)
			out_dim = int(model.head.in_features)
			model.head = nn.Identity()
			model.eval()
			return model, out_dim

		if timm is None:
			raise ValueError(
				"swin_large is unavailable in torchvision and timm is not installed. "
				"Install timm>=1.0.0 to use backbone 'swin_large_patch4_window7_224'."
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

	if name == "resnet50":
		weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
		model = models.resnet50(weights=weights)
		model.fc = nn.Identity()
		model.eval()
		return model, 2048

	raise ValueError(f"Unsupported model_name: {model_name}")


def get_device(device: str | None = None) -> torch.device:
	if device:
		requested = torch.device(device)
		if requested.type == "cuda" and not torch.cuda.is_available():
			print("[WARN] CUDA requested but unavailable, falling back to CPU.")
			return torch.device("cpu")
		return requested
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")

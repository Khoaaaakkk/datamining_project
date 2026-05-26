from __future__ import annotations

import torch
from torch import nn

try:
	from .transformers_gap import GenomicBranch, WSIBranch
except ImportError:  # pragma: no cover - fallback for script execution
	from fusion_feature.transformers_gap import GenomicBranch, WSIBranch


class MultimodalFusion(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.wsi_branch = WSIBranch()
		self.genomic_branch = GenomicBranch()

	def forward(
		self,
		guided_wsi_embeds: torch.Tensor,
		q_gene_features: torch.Tensor,
	) -> torch.Tensor:
		wsi_features = self.wsi_branch(guided_wsi_embeds)
		genomic_features = self.genomic_branch(q_gene_features)
		fusion_feature = torch.cat([wsi_features, genomic_features], dim=-1)
		return fusion_feature


if __name__ == "__main__":
    # Mock input tensors
	device = torch.device("cpu")
	model = MultimodalFusion().to(device)
	model.eval()

	batch_size = 2
	guided_wsi_embeds = torch.randn(batch_size, 6, 512, device=device)
	q_gene_features = torch.randn(batch_size, 6, 512, device=device)

	with torch.no_grad():
		fusion_feature = model(guided_wsi_embeds, q_gene_features)

	filename = "patient_001"
	torch.save(fusion_feature, f"{filename}_fusion_feature.pt")
	print(f"Fusion feature shape: {tuple(fusion_feature.shape)}")
	print(f"Saved to: {filename}_fusion_feature.pt")

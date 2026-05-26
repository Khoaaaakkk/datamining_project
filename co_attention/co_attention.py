from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn


class GenomicGuidedCoAttention(nn.Module):
	"""
	Genomic-guided co-attention for WSI patches.

	Inputs:
	- Q: (B, G, D) or (G, D) genomic embeddings
	- K: (B, N, D) or (N, D) patch embeddings (keys)
	- V: (B, N, D) or (N, D) patch embeddings (values)

	Outputs:
	- output: (B, G, D) or (G, D)
	- attn_weights: (B, G, N) or (G, N)
	"""

	def __init__(self, dim: int = 512):
		super().__init__()
		self.dim = int(dim)

	def forward(
		self,
		q: torch.Tensor,
		k: torch.Tensor,
		v: torch.Tensor,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		if q.dim() == 2:
			q = q.unsqueeze(0)
		if k.dim() == 2:
			k = k.unsqueeze(0)
		if v.dim() == 2:
			v = v.unsqueeze(0)

		if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
			raise ValueError("Q, K, V must be 2D or 3D tensors")

		batch_size, g_len, d_q = q.shape
		k_batch, n_len, d_k = k.shape
		v_batch, v_len, d_v = v.shape
		print(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
		if k_batch != batch_size or v_batch != batch_size:
			raise ValueError("Batch size mismatch between Q, K, V")
		if n_len != v_len:
			raise ValueError("K and V must have same sequence length")
		if d_q != d_k or d_k != d_v:
			raise ValueError("Q, K, V must share the same embedding dimension")

		scale = 1.0 / math.sqrt(d_k)
		attn_scores = torch.bmm(q, k.transpose(1, 2)) * scale
		attn_weights = torch.softmax(attn_scores, dim=-1)
		output = torch.bmm(attn_weights, v)

		if output.size(0) == 1:
			return output.squeeze(0), attn_weights.squeeze(0)
		return output, attn_weights

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionMIL(nn.Module):
	"""
	Simple gated-attention MIL for bag-level classification.
	Input: x [N, D] or [B, N, D]
	Output: logits [num_classes] or [B, num_classes], attention weights.
	"""

	def __init__(self, in_dim: int = 2048, attn_dim: int = 256, num_classes: int = 2):
		super().__init__()
		self.v = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Tanh())
		self.u = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Sigmoid())
		self.w = nn.Linear(attn_dim, 1)
		self.classifier = nn.Linear(in_dim, num_classes)

	def _forward_single(self, x: torch.Tensor):
		a = self.w(self.v(x) * self.u(x)).squeeze(-1)  # [N]
		attn = torch.softmax(a, dim=0)
		bag_repr = torch.sum(attn.unsqueeze(-1) * x, dim=0)
		logits = self.classifier(bag_repr)
		return logits, attn

	def forward(self, x: torch.Tensor):
		if x.dim() == 2:
			return self._forward_single(x)
		if x.dim() == 3:
			logits_list = []
			attn_list = []
			for i in range(x.shape[0]):
				logits, attn = self._forward_single(x[i])
				logits_list.append(logits)
				attn_list.append(attn)
			return torch.stack(logits_list, dim=0), attn_list
		raise ValueError("Input x must be 2D [N,D] or 3D [B,N,D]")

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


# class GlobalAttentionPooling(nn.Module):
# 	def __init__(self, dim: int = 512, hidden_dim: int = 256) -> None:
# 		super().__init__()
# 		self.score = nn.Sequential(
# 			nn.Linear(dim, hidden_dim),
# 			nn.Tanh(),
# 			nn.Linear(hidden_dim, 1),
# 		)

# 	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
# 		"""
# 		x: [B, L, D]
# 		returns: context [B, D], weights [B, L]
# 		"""
# 		attn_logits = self.score(x).squeeze(-1)
# 		attn_weights = torch.softmax(attn_logits, dim=1)
# 		context = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
# 		return context, attn_weights

class GlobalAttentionPooling(nn.Module):
    def __init__(self, dim: int = 512, hidden_dim: int = 256) -> None:
        super().__init__()
        # Nhánh Tanh (Tương ứng với ma trận trọng số V trong công thức)
        self.linear_V = nn.Linear(dim, hidden_dim)
        
        # Nhánh Sigmoid (Tương ứng với ma trận trọng số U - đóng vai trò làm Cổng)
        self.linear_U = nn.Linear(dim, hidden_dim)
        
        # Lớp tuyến tính cuối cùng (Tương ứng với ma trận trọng số W) để ra điểm số scalar
        self.linear_W = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, L, D] (với B là Batch, L=6 là Sequence Length, D=512 là Dimension)
        returns: context [B, D], weights [B, L]
        """
        # 1. Tính toán hai nhánh song song theo công thức
        out_V = torch.tanh(self.linear_V(x))       # tanh(V * h)
        out_U = torch.sigmoid(self.linear_U(x))    # sigmoid(U * h)
        
        # 2. Phép nhân Gated Attention (ký hiệu ⊙ - Hadamard product)
        gated_attention = out_V * out_U
        
        # 3. Áp dụng trọng số W để đưa về dạng logits [B, L]
        attn_logits = self.linear_W(gated_attention).squeeze(-1)
        
        # 4. Softmax để chuẩn hóa tổng trọng số bằng 1 dọc theo chiều Sequence (L)
        attn_weights = torch.softmax(attn_logits, dim=1)
        
        # 5. Nhân trọng số ngược lại với input gốc và tính tổng (Weighted Sum)
        context = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        
        return context, attn_weights
# chck lại GAP
class WSIBranch(nn.Module):
	def __init__(
		self,
		d_model: int = 512,
		nhead: int = 8,
		num_layers: int = 2,
		dim_feedforward: int = 2048,
		dropout: float = 0.1,
	) -> None:
		super().__init__()
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=nhead,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			batch_first=True,
		)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.pool = GlobalAttentionPooling(dim=d_model)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		x: [B, 6, 512]
		returns: [B, 512]
		"""
		encoded = self.encoder(x)
		context, _ = self.pool(encoded)
		return context


class GenomicBranch(nn.Module):
	def __init__(
		self,
		d_model: int = 512,
		nhead: int = 8,
		num_layers: int = 2,
		dim_feedforward: int = 2048,
		dropout: float = 0.1,
	) -> None:
		super().__init__()
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=nhead,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			batch_first=True,
		)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.pool = GlobalAttentionPooling(dim=d_model)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		x: [B, 6, 512]
		returns: [B, 512]
		"""
		encoded = self.encoder(x)
		context, _ = self.pool(encoded)
		return context

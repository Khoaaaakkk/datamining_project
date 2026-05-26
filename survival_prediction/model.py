from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

class NLLSurvivalLoss(nn.Module):
    """
    Negative Log-Likelihood (NLL) Loss cho bài toán Discrete Survival Prediction (MCAT).
    """
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, hazards: torch.Tensor, Y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hazards: Tensor shape (batch_size, num_bins). Đầu ra của mô hình (đã qua Sigmoid).
            Y: Tensor shape (batch_size,). Chỉ số khoảng thời gian (bin) thực tế bệnh nhân rơi vào (từ 0 đến num_bins - 1).
            c: Tensor shape (batch_size,). Trạng thái kiểm duyệt (Censoring). 
               Quy ước: c = 1 là Kiểm duyệt (Censored/Alive), c = 0 là Không kiểm duyệt (Uncensored/Dead).
        """
        batch_size = hazards.shape[0]

        # 1. Tính log(h) và log(1 - h) để đảm bảo ổn định số học (tránh log(0))
        log_h = torch.log(hazards + self.eps)
        log_1_minus_h = torch.log(1.0 - hazards + self.eps)

        # 2. Tính log(S_t)
        # Thay vì S_t = cumprod(1 - h), ta dùng log(S_t) = cumsum(log(1 - h))
        log_S = torch.cumsum(log_1_minus_h, dim=1)

        # Padding thêm 1 cột số 0 vào đầu. 
        # Lý do: Khi Y = 0 (bệnh nhân chết ở khoảng đầu tiên), S_{Y-1} = S_{-1} = 1 -> log(1) = 0
        log_S_padded = torch.cat(
            [torch.zeros(batch_size, 1, device=hazards.device), log_S], 
            dim=1
        )

        # Reshape index để dùng cho hàm gather
        idx = Y.view(batch_size, 1)

        # 3. Tính toán Log-Likelihood dựa trên công thức toán học
        # Cho ca tử vong (c = 0): Xác suất = h_Y * S_{Y-1} => log = log(h_Y) + log(S_{Y-1})
        uncensored_log_prob = torch.gather(log_h, 1, idx) + torch.gather(log_S_padded, 1, idx)
        
        # Cho ca kiểm duyệt/còn sống (c = 1): Xác suất = S_Y => log = log(S_Y)
        censored_log_prob = torch.gather(log_S_padded, 1, idx + 1)

        # Đổi dấu thành Negative Log-Likelihood
        uncensored_loss = -uncensored_log_prob.squeeze(1)
        censored_loss = -censored_log_prob.squeeze(1)

        # 4. Gộp Loss dựa vào biến 'c'
        loss = (1.0 - c) * uncensored_loss + c * censored_loss

        # Trả về trung bình loss của cả batch
        return loss.mean()

class SurvivalRiskPredictor(nn.Module):
    """Simple survival prediction head.

    Expects a single vector per patient (concatenated features) and outputs
    discrete hazard logits per time bin.
    """

    def __init__(self, input_dim: int, num_bins: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_bins),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return torch.sigmoid(logits)


def build_feature_vector(
    # h5_features: torch.Tensor,
    # co_attention: torch.Tensor,
    fusion: torch.Tensor,
) -> torch.Tensor:
    """Concatenate per-patient features into a single vector.

    Adjust this to match your feature dimensions (e.g., use pooling strategy).
    """

    # h5_vec = h5_features.flatten()
    # co_att_vec = co_attention.flatten()
    fusion_vec = fusion.flatten()
    return torch.cat([fusion_vec], dim=0)
    # return torch.cat([h5_vec, fusion_vec], dim=0)
    # return torch.cat([h5_vec, co_att_vec, fusion_vec], dim=0)

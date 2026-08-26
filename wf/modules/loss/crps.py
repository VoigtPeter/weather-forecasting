from typing import Literal

import torch
from torch import nn


class CRPSLoss(nn.Module):
    def __init__(self, fair: bool = True, reduction: Literal["mean"] | None = "mean") -> None:
        super().__init__()
        self.fair = fair
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_pred -> (batch, ensemble, vars, lat, lon)
        # y_true -> (batch, vars, lat, lon)
        M = y_pred.shape[1]
        if self.fair:
            spread_coef = 1.0 / (2.0 * M * (M - 1))
        else:
            spread_coef = 1.0 / (2.0 * (M ** 2))
        crps = torch.abs(y_pred - y_true.unsqueeze(dim=1)).mean(dim=1) - \
            (spread_coef * ( torch.sum(torch.abs(y_pred.unsqueeze(dim=1) - y_pred.unsqueeze(dim=2)), dim=(1, 2)) ))
        if self.reduction == "mean":
            return crps.mean()  # -> scalar
        return crps  # -> (batch, vars, lat, lon)

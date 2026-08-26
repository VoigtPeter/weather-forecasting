import torch
from torch import nn


class MSELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_pred - y_true) ** 2)

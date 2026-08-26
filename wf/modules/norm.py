import torch
from torch import nn

from wf.modules.ffn import FFN


class AdaLN(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.ffn = FFN(cond_dim, 2)
        self.to_dim = nn.Linear(cond_dim, 2 * dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:

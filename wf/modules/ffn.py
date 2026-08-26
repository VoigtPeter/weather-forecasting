import torch
from torch import nn


class FFN(nn.Module):
    def __init__(self, dim: int, expansion_factor: int):
        super().__init__()
        assert expansion_factor % 2 == 0 and expansion_factor >= 2
        self.lin_up = nn.Linear(dim, dim * expansion_factor)
        self.lin_down = nn.Linear((dim * expansion_factor) // 2, dim)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin_up(x)
        x1, x2 = x.chunk(2, dim=-1)
        return self.lin_down(self.activation(x1) * x2)

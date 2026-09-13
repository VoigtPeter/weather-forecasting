import torch
from torch import nn


class PeriodicSinusoidalPE(nn.Module):
    def __init__(self, dim: int, period: float = 10000):
        super().__init__()
        self.period = period
        self.dim = dim
        assert dim % 2 == 0
        self.register_buffer("coef", (torch.arange(1.0, self.dim // 2 + 1, 1.0).to(torch.float32).view(1, -1) * 2 * torch.pi) / self.period, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,) -> (B, D)
        x = x.view(-1, 1) * self.coef
        return torch.cat((torch.sin(x), torch.cos(x)), dim=-1)


class LatLonWrap(nn.Module):
    """
    Wrap encoding, as used by MacAodha et al

    (copied from https://github.com/MarcCoru/locationencoder/blob/main/locationencoder/pe/wrap.py)
    """
    def __init__(self) -> None:
        super().__init__()
        self.out_dim = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (*, 2 {lat, lon}) -> (*, 4)
        x = torch.deg2rad(x)
        return torch.cat((torch.cos(x), torch.sin(x)), dim=-1)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    D = 100
    S = 1.0
    pe = PeriodicSinusoidalPE(dim=D, period=365.25)
    fig, ax = plt.subplots()
    ax.imshow(pe(torch.arange(0.0, 365.25, S)).cpu().numpy())
    ax.set_aspect(D / (365 / S))
    plt.show()
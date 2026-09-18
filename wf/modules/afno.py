import torch
from torch import nn
from torch.nn import functional as F

from einops.layers.torch import EinMix


class ComplexBlockLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_blocks: int, bias: bool = True):
        super().__init__()
        assert in_features % num_blocks == 0
        assert out_features % num_blocks == 0
        block_in_features = in_features // num_blocks
        block_out_features = out_features // num_blocks
        self.lin1 = EinMix("... (k i) -> ... (k o)", weight_shape="k o i", bias_shape=None,
                           i=block_in_features, o=block_out_features, k=num_blocks)
        self.lin2 = EinMix("... (k i) -> ... (k o)", weight_shape="k o i", bias_shape=None,
                           i=block_in_features, o=block_out_features, k=num_blocks)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, 2))
            nn.init.kaiming_uniform_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (*, in, 2)
        x_real = x[..., 0]
        x_imag = x[..., 1]
        x = torch.stack((
                self.lin1(x_real) - self.lin2(x_imag), # real
                self.lin1(x_imag) + self.lin2(x_real), # imag
        ), dim=-1)
        if self.bias is not None:
            x = x + self.bias
        return x  # x -> (*, out, 2)


class AFNO2D(nn.Module):
    def __init__(self,
                 nlat: int,
                 nlon: int,
                 dim: int,
                 num_blocks: int = 1,
                 sparsity: float = 0.1,
                 expansion_factor: int = 1,
                 activation: type[nn.Module] = nn.SiLU) -> None:
        super().__init__()
        self.nlat = nlat
        self.nlon = nlon
        self.dim = dim
        self.sparsity = sparsity

        self.mlp = nn.Sequential(
            ComplexBlockLinear(dim, dim * expansion_factor, num_blocks),
            activation(),
            ComplexBlockLinear(dim * expansion_factor, dim, num_blocks),
        )

    def forward(self, x: torch.Tensor, field_size: tuple[int, int] | None = None) -> torch.Tensor:
        flatten_field: bool = True
        if x.dim() == 3:  # x -> (batch, H*W, dim)
            H, W = self.nlat, self.nlon
            x = x.view(-1, H, W, self.dim)
        else:
            _, H, W, _ = x.shape
            flatten_field = False
        # x -> (batch, H, W, dim)
        x = torch.view_as_real(torch.fft.rfft2(x, dim=(1, 2), norm="ortho"))  # -> (batch, H, W//2 + 1, dim, 2)
        x = self.mlp(x)
        x = F.softshrink(x, lambd=self.sparsity)
        x = torch.fft.irfft2(torch.view_as_complex(x), s=(H, W), dim=(1, 2), norm="ortho")  # -> (batch, H, W, dim)
        if flatten_field:
            x = x.view(-1, H*W, self.dim)  # -> (batch, H*W, dim)
        return x

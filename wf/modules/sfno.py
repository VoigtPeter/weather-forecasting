import torch
from torch import nn

import torch_harmonics as th

from einops.layers.torch import Rearrange


from wf.modules.afno import ComplexBlockLinear


class SFNO(nn.Module):
    def __init__(self, nlat: int, nlon: int, dim: int, num_blocks: int = 1, expansion_factor: int = 1, activation: type[nn.Module] = nn.SiLU) -> None:
        super().__init__()
        self.nlat = nlat
        self.nlon = nlon
        self.dim = dim

        truncate_dim = (nlat // 2) + 1
        self.sht = th.RealSHT(
            nlat,
            nlon,
            grid="equiangular",
            norm="ortho",
            lmax=truncate_dim,
            mmax=truncate_dim,
        )
        self.isht = th.InverseRealSHT(
            nlat,
            nlon,
            grid="equiangular",
            norm="ortho",
            lmax=truncate_dim,
            mmax=truncate_dim,
        )

        self.mlp = nn.Sequential(
            ComplexBlockLinear(dim, dim * expansion_factor, num_blocks),
            activation(),
            ComplexBlockLinear(dim * expansion_factor, dim, num_blocks),
        )

        self.before_sht = Rearrange("... lat lon dim -> ... dim lat lon")
        self.after_sht = Rearrange("... dim lat lon -> ... lat lon dim")

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        flatten_field: bool = True
        if x.dim() == 3:  # x -> (batch, H*W, dim)
            x = x.view(-1, self.nlat, self.nlon, self.dim)
        else:
            flatten_field = False
        # x -> (batch, H, W, dim)

        x = self.before_sht(x)  # -> (batch, dim, H, W)
        x = self.after_sht(self.sht(x))  # -> (batch, sh.lmax, sh.mmax, dim)
        x = torch.view_as_real(x)  # -> (batch, sh.lmax, sh.mmax, dim, 2)

        x = self.mlp(x)  # -> (batch, sh.lmax, sh.mmax, dim, 2)

        x = torch.view_as_complex(x)  # -> (batch, sh.lmax, sh.mmax, dim)
        x = self.before_sht(x)  # -> (batch, dim, sh.lmax, sh.mmax)
        x = self.after_sht(self.isht(x))  # -> (batch, H, W, dim)

        if flatten_field:
            x = x.view(-1, self.nlat * self.nlon, self.dim)  # -> (batch, H*W, dim)
        return x


if __name__ == "__main__":
    sfno = SFNO(nlat=32, nlon=64, dim=16, num_blocks=4, expansion_factor=2, activation=nn.GELU)

    x = torch.randn(3, 32, 64, 16)
    y = sfno(x)

    print(x.shape)
    print(y.shape)
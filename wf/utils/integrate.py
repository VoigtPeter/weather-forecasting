import math

import torch
from torch import nn


_1_4PI = 1.0 / (4.0 * math.pi)


class IntegrateSphere(nn.Module):
    def __init__(self, nlat: int, nlon: int) -> None:
        super().__init__()
        #n_lat = math.floor(180.0 / angle)
        #n_lon = math.floor(360.0 / angle)
        angle = 180.0 / nlat
        lat_grid = torch.deg2rad(
            (torch.arange(nlat, dtype=torch.float32) * angle) +
            ((180.0 - ((nlat - 1) * angle)) / 2.0)  # center grid over 180° with equal padding at top and bottom
        )
        self.register_buffer(
            "quadrature_weights",
            ((2.0 * (math.pi ** 2)) / (nlat * nlon)) *
            torch.sin(lat_grid.view(-1, 1).expand(nlat, nlon)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of shape (*, lat, lon)
        :return: Tensor of shape (*,)
        """
        # x -> (*, lat, lon)
        return _1_4PI * (x * self.quadrature_weights).sum(dim=(-1, -2))  # -> (*,)

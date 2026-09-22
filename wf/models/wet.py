from dataclasses import dataclass, asdict
from typing import Literal

import torch
from torch import nn

from einops.layers.torch import EinMix, Rearrange

from wf.modules.ffn import FFN
from wf.modules.pe import PeriodicSinusoidalPE, LatLonWrap
from wf.modules.transformer import TransformerBlock


class TimePE(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.time_day_pe = PeriodicSinusoidalPE(dim=dim, period=24.0)
        self.time_year_pe = PeriodicSinusoidalPE(dim=dim, period=356.25 * 24.0)
        self.out_dim: int = 2 * dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Hours of the year as tensor of shape (B,)
        :return: Time encoding as tensor of shape (B, 2*dim)
        """
        tod_pe = self.time_day_pe(x)  # -> (B, emb_dim)
        toy_pe = self.time_year_pe(x)  # -> (B, emb_dim)
        return torch.cat((tod_pe, toy_pe), dim=-1)  # (B, 2*dim)


class WeTEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            dim: int,
            patch_h: int,
            patch_w: int,
            tokens_h: int,
            tokens_w: int,
            separable_embed: bool = True,
            ffn_factor: int = 2,
            activation: type[nn.Module] | None = nn.GELU,
    ) -> None:
        super().__init__()
        if separable_embed:
            self.to_tokens = EinMix(
                "... C (tokens_h patch_h) (tokens_w patch_w) -> ... (tokens_h tokens_w) (C D)",
                weight_shape="C patch_h patch_w D",
                #bias_shape="tokens_h tokens_w C D",
                C=in_channels, D=dim, patch_h=patch_h, patch_w=patch_w, tokens_h=tokens_h, tokens_w=tokens_w,
            )
        else:
            self.to_tokens = EinMix(
                "... C (tokens_h patch_h) (tokens_w patch_w) -> ... (tokens_h tokens_w) (C D)",
                weight_shape="patch_h patch_w D",
                C=in_channels, D=dim, patch_h=patch_h, patch_w=patch_w,
            )
        self.proj_down = nn.Linear(dim * in_channels, dim)
        #self.activation = activation() if activation is not None else nn.Identity()
        #self.ffn = FFN(dim, ffn_factor, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (..., vars, H, W)
        :return: (..., th*tw, dim)
        """
        x = self.to_tokens(x)
        #x = self.activation(x)
        x = self.proj_down(x)
        #x = self.activation(x)
        #x = self.ffn(x)
        return x

class WeTDecoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            dim: int,
            patch_h: int,
            patch_w: int,
            tokens_h: int,
            tokens_w: int,
            separable_embed: bool = True,
            init_zeros: bool = False,
            ffn_factor: int = 2,
            activation: type[nn.Module] | None = nn.GELU,
    ) -> None:
        super().__init__()
        if separable_embed:
            self.to_field = EinMix(
                "... (tokens_h tokens_w) (C D) -> ... C (tokens_h patch_h) (tokens_w patch_w)",
                weight_shape="D patch_w patch_h C",
                #bias_shape="patch_w patch_h C",
                C=in_channels, D=dim, patch_h=patch_h, patch_w=patch_w, tokens_h=tokens_h, tokens_w=tokens_w,
            )
        else:
            self.to_field = EinMix(
                "... (tokens_h tokens_w) (C D) -> ... C (tokens_h patch_h) (tokens_w patch_w)",
                weight_shape="D patch_h patch_w",
                C=in_channels, D=dim, patch_h=patch_h, patch_w=patch_w, tokens_h=tokens_h, tokens_w=tokens_w,
            )
        self.proj_up = nn.Linear(dim, dim * in_channels)
        if init_zeros:
            nn.init.constant_(self.proj_up.weight, 0.0)
            nn.init.constant_(self.proj_up.bias, 0.0)
        #self.activation = activation() if activation is not None else nn.Identity()
        #self.ffn = FFN(dim, ffn_factor, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (..., th*tw, dim)
        :return: (..., vars, H, W)
        """
        #x = self.ffn(x)
        #x = self.activation(x)
        x = self.proj_up(x)
        #x = self.activation(x)
        return self.to_field(x)


class WeT(nn.Module):
    def __init__(
            self,
            dim: int,
            num_layers: int,
            num_heads: int,
            dim_heads: int,
            ffn_factor: int,
            num_surface_vars: int,
            num_atmosphere_vars: int,
            nlat: int,
            nlon: int,
            patch_h: int,
            patch_w: int,
            separable_embed: bool = True,
            noise_dim: int | None = None,
            dropout: float = 0.0,
            time_embed: int | None = None,
            latlon_embed: bool = False,
            latlon_pe: Literal["wrap"] | None = "wrap",
            surface_geopotential: bool = False,
            land_sea_mask: bool = False,
            aux_spatial_dim: int | None = None,
            spatial_mixer: Literal["mhsa", "afno", "sfno"] = "mhsa",
            graph_conv: bool = False,
            graph_conv_level: int = 2,
            graph_conv_layers: int = 1,
            graph_conv_node_embed: int = 0,
            graph_conv_edge_embed: int = 0,
            global_skip: bool = False,
            activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        assert dim % 2 == 0, "'dim' must be multiple of 2"

        self.num_surface_vars = num_surface_vars
        self.num_atmosphere_vars = num_atmosphere_vars
        self.global_skip = global_skip

        self.H_tokens = nlat // patch_h
        self.W_tokens = nlon // patch_w

        # AUX: global
        self.aux_global_dim: int = 0
        self.time_pe = None
        self.noise_dim = noise_dim
        if time_embed is not None and time_embed > 0:
            self.time_pe = TimePE(dim=time_embed)
            self.aux_global_dim += self.time_pe.out_dim
        if noise_dim is not None and noise_dim > 0:
            self.aux_global_dim += noise_dim

        # AUX: spatial
        self.aux_spatial_num: int = 0
        self.latlon_embed: nn.Module | None = None
        if latlon_embed is not None and latlon_embed > 0:
            latlon_pe_module = None
            latlon_ffn_dim = 2
            if latlon_pe is not None:
                if latlon_pe == "wrap":
                    latlon_pe_module = LatLonWrap()
                    latlon_ffn_dim = latlon_pe_module.out_dim
                else:
                    raise NotImplementedError()
            latlon_net = FFN(latlon_ffn_dim, expansion_factor=ffn_factor)
            self.latlon_embed = nn.Sequential(
                latlon_pe_module,
                latlon_net,
            ) if latlon_pe_module is not None else latlon_net
            self.aux_spatial_num += latlon_ffn_dim
            self.latlon_view = Rearrange("H W c -> 1 c H W")
        self.surface_geopotential = surface_geopotential
        if surface_geopotential:
            self.aux_spatial_num += 1
        self.land_sea_mask = land_sea_mask
        if land_sea_mask:
            self.aux_spatial_num += 1
        self.aux_spatial_encoder = None
        if self.aux_spatial_num > 0:
            assert aux_spatial_dim is not None and aux_spatial_dim > 0
            self.aux_spatial_encoder = WeTEncoder(
                self.aux_spatial_num,
                aux_spatial_dim,
                patch_h,
                patch_w,
                self.H_tokens,
                self.W_tokens,
                separable_embed,
                ffn_factor=ffn_factor,
                activation=activation,
            )

        # ENCODER
        self.surface_encoder = WeTEncoder(
            self.num_surface_vars,
            dim // 2,
            patch_h,
            patch_w,
            self.H_tokens,
            self.W_tokens,
            separable_embed,
            ffn_factor=ffn_factor,
            activation=activation,
        )
        self.atmosphere_encoder = WeTEncoder(
            self.num_atmosphere_vars,
            dim // 2,
            patch_h,
            patch_w,
            self.H_tokens,
            self.W_tokens,
            separable_embed,
            ffn_factor=ffn_factor,
            activation=activation,
        )

        # DECODER
        self.surface_decoder = WeTDecoder(
            self.num_surface_vars,
            dim,
            patch_h,
            patch_w,
            self.H_tokens,
            self.W_tokens,
            separable_embed,
            init_zeros=global_skip,
            ffn_factor=ffn_factor,
            activation=activation,
        )
        self.atmosphere_decoder = WeTDecoder(
            self.num_atmosphere_vars,
            dim,
            patch_h,
            patch_w,
            self.H_tokens,
            self.W_tokens,
            separable_embed,
            init_zeros=global_skip,
            ffn_factor=ffn_factor,
            activation=activation,
        )

        # TRANSFORMER BLOCKS
        self.blocks = nn.ModuleList([TransformerBlock(
            nlat=self.H_tokens,
            nlon=self.W_tokens,
            dim=dim,
            num_heads=num_heads,
            dim_heads=dim_heads,
            expansion_factor=ffn_factor,
            conditioning="adaLN",
            cond_dim=self.aux_global_dim,
            add_conditioning=aux_spatial_dim if self.aux_spatial_num > 0 else None,
            dropout=dropout,
            mixer=spatial_mixer,
            graph_conv=graph_conv,
            graph_conv_level=graph_conv_level,
            graph_conv_layers=graph_conv_layers,
            graph_conv_node_embed=graph_conv_node_embed,
            graph_conv_edge_embed=graph_conv_edge_embed,
            activation=activation,
        ) for _ in range(num_layers)])

    def forward(
            self,
            x: torch.Tensor,                                   # -> (B, vars, H, W)
            time: torch.Tensor | None = None,                  # -> (B,)
            latlon: torch.Tensor | None = None,                # -> (H, W, 2)
            land_sea_mask: torch.Tensor | None = None,         # -> (H, W)
            surface_geopotential: torch.Tensor | None = None,  # -> (H, W)
            noise_cond: torch.Tensor | None = None,            # -> (B, noise_dim)
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        orig_x = x

        # TIME
        aux_global: torch.Tensor | None = None  # -> (B, aux_global_dim)
        if self.time_pe is not None:
            aux_global = self.time_pe(time)
        if self.noise_dim is not None and self.noise_dim > 0:
            assert noise_cond is not None, "argument 'noise_cond' cannot be None when 'noise_dim' > 0"
            if aux_global is None:
                aux_global = noise_cond
            else:
                aux_global = torch.cat((aux_global, noise_cond), dim=-1).unsqueeze(1)  # -> (B, 1, aux_global_dim)

        # LAT/LON
        aux_spatial: torch.Tensor | None = None  # -> (B, aux_spatial_num, H, W)
        if self.latlon_embed is not None:
            aux_spatial = self.latlon_view(self.latlon_embed(latlon))  # -> (1, latlon_dim, H, W)
        if self.land_sea_mask:
            assert land_sea_mask is not None
            land_sea_mask = land_sea_mask.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)
            if aux_spatial is None:
                aux_spatial = land_sea_mask
            else:
                aux_spatial = torch.cat((aux_spatial, land_sea_mask), dim=1)  # cat along var dim
        if self.surface_geopotential:
            assert surface_geopotential is not None
            surface_geopotential = surface_geopotential.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)
            if aux_spatial is None:
                aux_spatial = land_sea_mask
            else:
                aux_spatial = torch.cat((aux_spatial, surface_geopotential), dim=1)  # cat along var dim
        if aux_spatial is not None:
            aux_spatial = self.aux_spatial_encoder(aux_spatial)  # -> (1, th*tw, aux_dim)
            aux_spatial = aux_spatial.expand(batch_size, *aux_spatial.shape[1:])  # -> (B, H*W, aux_dim)

        # ENCODE
        x = torch.cat([
            self.surface_encoder(x[:, :self.num_surface_vars, :, :]),     # -> (B, th*tw, dim//2),
            self.atmosphere_encoder(x[:, self.num_surface_vars:, :, :]),  # -> (B, th*tw, dim//2)
        ], dim=-1)  # -> (B, th*tw, dim)

        # BLOCKS
        for block in self.blocks:
            x = block(x, c=aux_global, add_c=aux_spatial)

        # DECODE
        x = torch.cat([
            self.surface_decoder(x),  # -> (B, vars_surface, th, tw),
            self.atmosphere_decoder(x),  # -> (B, vars_atmos, th, tw)
        ], dim=1)  # cat along var dim

        if self.global_skip:
            return x + orig_x
        return x

    @classmethod
    def from_config(cls, config: "WeTConfig", nlat: int, nlon: int, num_surface_vars: int, num_atmosphere_vars: int) -> "WeT":
        kwargs = asdict(config)
        patch_size = kwargs.pop("patch_size", [4, 4])

        kwargs.update({
            "num_surface_vars": num_surface_vars,
            "num_atmosphere_vars": num_atmosphere_vars,
            "nlat": nlat,
            "nlon": nlon,
            "patch_h": patch_size[0],
            "patch_w": patch_size[1],
        })

        return WeT(**kwargs)

    def compile(self, *args, **kwargs) -> None:
        self.surface_encoder.compile(*args, **kwargs)
        self.atmosphere_encoder.compile(*args, **kwargs)
        self.surface_decoder.compile(*args, **kwargs)
        self.surface_encoder.compile(*args, **kwargs)
        for block in self.blocks:
            block.compile(*args, **kwargs)


@dataclass
class WeTConfig:
    dim: int
    num_layers: int
    num_heads: int
    dim_heads: int
    ffn_factor: int
    patch_size: tuple[int, int]
    separable_embed: bool = True
    #noise: Literal["adaLN"] | None = None
    noise_dim: int | None = None
    dropout: float = 0.0
    time_embed: int | None = None
    latlon_embed: bool = False
    latlon_pe: Literal["wrap"] | None = "wrap"
    surface_geopotential: bool = False
    land_sea_mask: bool = False
    aux_spatial_dim: int | None = None
    spatial_mixer: Literal["mhsa", "afno", "sfno"] = "mhsa"
    graph_conv: bool = False
    graph_conv_level: int = 2
    graph_conv_layers: int = 1
    graph_conv_node_embed: int = 0
    graph_conv_edge_embed: int = 0
    global_skip: bool = False

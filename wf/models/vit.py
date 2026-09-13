from dataclasses import dataclass, asdict
from typing import Literal

import torch
from torch import nn

from einops.layers.torch import EinMix, Rearrange

from wf.modules.ffn import FFN
from wf.modules.pe import PeriodicSinusoidalPE, LatLonWrap
from wf.modules.transformer import TransformerBlock


class ViT(nn.Module):
    def __init__(
            self,
            dim: int,
            num_layers: int,
            num_heads: int,
            dim_heads: int,
            ffn_factor: int,
            num_vars: int,
            field_size: tuple[int, int],
            patch_size: tuple[int, int],
            separable_embed: bool = True,
            noise: Literal["adaLN"] | None = None,
            noise_dim: int | None = None,
            dropout: float = 0.0,
            time_embed: int | None = None,
            latlon_embed: bool = False,
            latlon_pe: Literal["wrap"] | None = "wrap",
            surface_geopotential: bool = False,
            land_sea_mask: bool = False,
            spatial_mixer: Literal["mhsa", "afno", "sfno"] = "mhsa",
            global_skip: bool = False,
    ):
        super().__init__()

        self.global_skip = global_skip
        self.noise = noise
        self.noise_dim = noise_dim
        self.dim = dim
        self.separable_embed = separable_embed
        self.num_vars = num_vars
        self.w, self.h = patch_size
        self.H_tokens, self.W_tokens = (field_size[0] // self.h), (field_size[1] // self.w)

        self.blocks = nn.ModuleList([TransformerBlock(
            nlat=self.H_tokens,
            nlon=self.W_tokens,
            dim=dim,
            num_heads=num_heads,
            dim_heads=dim_heads,
            expansion_factor=ffn_factor,
            conditioning=noise,
            cond_dim=noise_dim,
            dropout=dropout,
            mixer=spatial_mixer,
        ) for _ in range(num_layers)])

        # aux. features
        # - lat/lon embedding
        aux_dim = 0
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
            aux_dim += latlon_ffn_dim
            self.latlon_view = Rearrange("H W c -> 1 c H W")
        self.surface_geopotential = surface_geopotential
        if surface_geopotential:
            aux_dim += 1
        self.land_sea_mask = land_sea_mask
        if land_sea_mask:
            aux_dim += 1

        self.pos_embedding = nn.Parameter(torch.randn(1, (field_size[0] // self.h) * (field_size[1] // self.w) , dim ))

        if self.separable_embed:
            self.to_tokens = EinMix("b v (H h) (W w) -> b H W (v d)", weight_shape="v h w d", v=num_vars + aux_dim, d=dim, w=self.w, h=self.h)
            self.to_field = EinMix("b (H W) (v d) -> b v (H h) (W w)", weight_shape="v d h w", v=num_vars, d=dim, w=self.w, h=self.h, H=(field_size[0] // self.h))
            self.proj_down = nn.Linear(dim * (num_vars + aux_dim), dim)
            self.proj_up = nn.Linear(dim, dim * num_vars)
        else:
            self.to_tokens = nn.Sequential(
                Rearrange("b v (H h) (W w) -> b H W (h w v)", w=self.w, h=self.h),
                EinMix("b H W (h w v) -> b H W d", weight_shape="h w d", v=num_vars, d=dim, w=self.w, h=self.h, H=(field_size[0] // self.h), W=(field_size[1] // self.w)),
            )
            self.to_field = nn.Sequential(
                EinMix("b (H W) d -> b (H W) hwv", weight_shape="d hwv", v=num_vars, d=dim, w=self.w, h=self.h, H=(field_size[0] // self.h), W=(field_size[1] // self.w)),
                Rearrange("b (H W) (h w v) -> b v (H h) (W w)", w=self.w, h=self.h, H=(field_size[0] // self.h)),
            )
            #self.to_field = EinMix("b (H W) d -> b v (H h) (W w)", weight_shape="d h w", v=num_vars, d=config.dim,
            #                       w=self.w, h=self.h, H=(field_size[0] // self.h))
            self.proj_down = nn.Identity()
            self.proj_up = nn.Identity()

        # time embedding
        if time_embed is not None:
            self.time_day_pe = PeriodicSinusoidalPE(dim=time_embed, period=24.0)
            self.time_year_pe = PeriodicSinusoidalPE(dim=time_embed, period=356.25)
            self.time_to_token = nn.Linear(2 * time_embed, dim)


    def forward(
            self,
            x: torch.Tensor,
            time: torch.Tensor | None = None,
            latlon: torch.Tensor | None = None,                # -> (H, W, 2)
            land_sea_mask: torch.Tensor | None = None,         # -> (H, W)
            surface_geopotential: torch.Tensor | None = None,  # -> (H, W)
            noise_cond: torch.Tensor | None = None,            # -> (B, 1, noise_dim)
    ) -> torch.Tensor:
        # x -> (batch, vars, H, W)
        batch_size = x.shape[0]
        x_orig = x

        # auxiliary features
        if self.latlon_embed is not None:
            latlon_embed = self.latlon_view(self.latlon_embed(latlon))  # -> (1, latlon_dim, H, W)
            x = torch.cat((
                x,
                latlon_embed.expand(batch_size, *latlon_embed.shape[1:])  # -> (batch, latlon_dim, H, W)
            ), dim=1) # cat along var dim
        if self.land_sea_mask:
            land_sea_mask = land_sea_mask.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)
            x = torch.cat((
                x,
                land_sea_mask.expand(batch_size, *land_sea_mask.shape[1:])  # -> (batch, 1, H, W)
            ), dim=1)  # cat along var dim
        if self.surface_geopotential:
            surface_geopotential = surface_geopotential.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)
            x = torch.cat((
                x,
                surface_geopotential.expand(batch_size, *surface_geopotential.shape[1:])  # -> (batch, 1, H, W)
            ), dim=1)  # cat along var dim

        #tokens = rearrange(x, "b v (H h) (W w) -> b (H W) (h w v)", w=self.w, h=self.h)
        #tokens = self.encode(tokens)  # -> (batch, tokens, dim)
        tokens = self.to_tokens(x)  # -> (b, H, W, d)
        batch_size, H, W, _ = tokens.shape
        tokens = tokens.reshape(batch_size, H * W, -1)  # -> (b, H*W, d)
        tokens = self.proj_down(tokens)
        tokens = tokens + self.pos_embedding  # learned spatial embedding

        if time is not None:
            tod_pe = self.time_year_pe(torch.fmod(time, time.new_tensor(24)))  # -> (b, emb_dim)
            toy_pe = self.time_year_pe(time)  # -> (b, emb_dim)
            time_pe = self.time_to_token(torch.cat((tod_pe, toy_pe), dim=-1)).unsqueeze(1)  # (b, 2 * emb_dim) -> (b, 1, dim)
            tokens = tokens + time_pe

        if noise_cond is None:
            if self.noise == "adaLN":
                noise_cond = torch.randn((tokens.shape[0], 1, self.noise_dim), device=tokens.device)  # noise vector is the same for all tokens of a batch item

        for block in self.blocks:
            tokens = block(tokens, c=noise_cond, field_size=(H, W))
        tokens = self.proj_up(tokens)

        if self.global_skip:
            return self.to_field(tokens) + x_orig
        return self.to_field(tokens)


        #tokens = self.decode(tokens)
        #return rearrange(tokens, "b (H W) (h w v) -> b v (H h) (W w)", W=(x.shape[-1]//self.w), w=self.w, h=self.h)

    @classmethod
    def from_config(cls, config: ViTConfig, field_size: tuple[int, int], num_vars: int) -> "ViT":
        kwargs = asdict(config)
        kwargs.update({"num_vars": num_vars, "field_size": field_size})
        return ViT(**kwargs)


@dataclass
class ViTConfig:
    dim: int
    num_layers: int
    num_heads: int
    dim_heads: int
    ffn_factor: int
    patch_size: tuple[int, int]
    separable_embed: bool = True
    noise: Literal["adaLN"] | None = None
    noise_dim: int | None = None
    dropout: float = 0.0
    time_embed: int | None = None
    latlon_embed: bool = False
    latlon_pe: Literal["wrap"] | None = "wrap"
    surface_geopotential: bool = False
    land_sea_mask: bool = False
    spatial_mixer: Literal["mhsa", "afno", "sfno"] = "mhsa"
    global_skip: bool = False

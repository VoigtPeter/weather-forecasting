from dataclasses import dataclass, asdict
from typing import Literal

import torch
from torch import nn

from einops.layers.torch import EinMix, Rearrange

from wf.modules.pe import PeriodicSinusoidalPE
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
    ):
        super().__init__()

        self.noise = noise
        self.noise_dim = noise_dim
        self.dim = dim

        self.blocks = nn.ModuleList([TransformerBlock(
            dim=dim,
            num_heads=num_heads,
            dim_heads=dim_heads,
            expansion_factor=ffn_factor,
            conditioning=noise,
            cond_dim=noise_dim,
            dropout=dropout,
        ) for _ in range(num_layers)])

        self.separable_embed = separable_embed
        self.num_vars = num_vars
        self.w, self.h = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, (field_size[0] // self.h) * (field_size[1] // self.w) , dim ))

        if self.separable_embed:
            self.to_tokens = EinMix("b v (H h) (W w) -> b (H W) (v d)", weight_shape="v h w d", v=num_vars, d=dim, w=self.w, h=self.h)
            self.to_field = EinMix("b (H W) (v d) -> b v (H h) (W w)", weight_shape="v d h w", v=num_vars, d=dim, w=self.w, h=self.h, H=(field_size[0] // self.h))
            self.proj_down = nn.Linear(dim * num_vars, dim)
            self.proj_up = nn.Linear(dim, dim * num_vars)
        else:
            self.to_tokens = nn.Sequential(
                Rearrange("b v (H h) (W w) -> b (H W) (h w v)", w=self.w, h=self.h),
                EinMix("b (H W) (h w v) -> b (H W) d", weight_shape="h w d", v=num_vars, d=dim, w=self.w, h=self.h, H=(field_size[0] // self.h), W=(field_size[1] // self.w)),
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


    def forward(self, x: torch.Tensor, time: torch.Tensor | None = None) -> torch.Tensor:
        # x -> (batch, vars, H, W)
        #tokens = rearrange(x, "b v (H h) (W w) -> b (H W) (h w v)", w=self.w, h=self.h)
        #tokens = self.encode(tokens)  # -> (batch, tokens, dim)
        tokens = self.to_tokens(x)
        tokens = self.proj_down(tokens)
        tokens = tokens + self.pos_embedding  # learned spatial embedding

        if time is not None:
            tod_pe = self.time_year_pe(torch.fmod(time, time.new_tensor(24)))  # -> (b, emb_dim)
            toy_pe = self.time_year_pe(time)  # -> (b, emb_dim)
            time_pe = self.time_to_token(torch.cat((tod_pe, toy_pe), dim=-1)).unsqueeze(1)  # (b, 2 * emb_dim) -> (b, 1, dim)
            tokens = tokens + time_pe

        noise_c = None
        if self.noise == "adaLN":
            noise_c = torch.randn((tokens.shape[0], 1, self.noise_dim), device=tokens.device)  # noise vector is the same for all tokens of a batch item
        for block in self.blocks:
            tokens = block(tokens, c=noise_c)
        tokens = self.proj_up(tokens)
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

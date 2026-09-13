import math
from typing import Literal

import torch
from torch import nn

from einops import einsum
from einops.layers.torch import EinMix

from wf.modules.afno import AFNO2D
from wf.modules.ffn import FFN
from wf.modules.sfno import SFNO


class MHSA(nn.Module):
    def __init__(self, dim: int, num_heads: int, dim_heads: int):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)
        self.w_q = nn.Parameter(torch.empty(num_heads, dim, dim_heads))

        self.to_Q = EinMix("b t d -> b n t h", weight_shape="n d h", n=num_heads, h=dim_heads, d=dim)
        self.to_K = EinMix("b t d -> b n t h", weight_shape="n d h", n=num_heads, h=dim_heads, d=dim)
        self.to_V = EinMix("b t d -> b n t h", weight_shape="n d h", n=num_heads, h=dim_heads, d=dim)
        self.to_out = EinMix("b n t h -> b t d", weight_shape="n h d", n=num_heads, h=dim_heads, d=dim)

        self.scale = math.sqrt(dim_heads)

    def forward(self, x: torch.Tensor, q: torch.Tensor | None = None, *args, **kwargs) -> torch.Tensor:
        # x -> (batch, tokens, dim)
        Q = self.to_Q(x) if q is None else self.to_Q(q)  # self or cross attention
        K = self.to_K(x)
        V = self.to_V(x)
        # Q, K, V -> (batch, heads, tokens, dim_heads)

        attention = self.softmax(einsum(Q, K, "b n tQ h, b n tK h -> b n tQ tK") / self.scale)  # -> (batch, heads, tokens_Q, tokens_KV)
        attV = einsum(attention, V, "b n tQ tk, b n tk h -> b n tQ h")  # -> (batch, heads, tokens_Q, dim_heads)

        return self.to_out(attV)  # -> (batch, tokens_Q, dim)


class TransformerBlock(nn.Module):
    def __init__(
            self,
            nlat: int,
            nlon: int,
            dim: int,
            num_heads: int,
            dim_heads: int,
            expansion_factor: int,
            conditioning: Literal["adaLN", "concat"] | None = None,
            cond_dim: int | None = None,
            dropout: float = 0.0,
            mixer: Literal["mhsa", "afno", "sfno"] = "mhsa",
    ):
        super().__init__()

        self.ffn = FFN(dim, expansion_factor)
        if mixer == "mhsa":
            self.mixer = MHSA(dim, num_heads, dim_heads)
        elif mixer == "afno":
            self.mixer = AFNO2D(dim, num_heads, expansion_factor=expansion_factor)
        elif mixer == "sfno":
            self.mixer = SFNO(nlat, nlon, dim, num_blocks=num_heads, expansion_factor=expansion_factor)
        else:
            raise NotImplementedError()
        self.ffn_norm = nn.RMSNorm(dim)
        self.mhsa_norm = nn.RMSNorm(dim)

        self.conditioning = conditioning
        if self.conditioning == "adaLN":
            assert cond_dim is not None, "cond_dim cannot be None"
            lin = nn.Linear(cond_dim, dim * 6, bias=False)
            nn.init.zeros_(lin.weight[(4 * dim):, :])  # init partially with zeros to ensure alpha starts with zeros
            self.adaLN = nn.Sequential(
                FFN(cond_dim, 2),
                nn.SiLU(),
                lin,
            )
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None, field_size: tuple[int, int] | None = None) -> torch.Tensor:
        if self.conditioning == "adaLN":
            gamma_1, gamma_2, beta_1, beta_2, alpha_1, alpha_2 = torch.chunk(self.adaLN(c), 6, dim=-1)

            x_att = self.mhsa_norm(x)  # pre-norm
            x_att = (x_att + beta_1) * gamma_1  # shift, scale
            x_att = self.mixer(x_att, field_size=field_size)  # attention
            x_att = x_att * alpha_1  # scale
            x = x + x_att
            x = self.dropout(x)

            x_ffn = self.ffn_norm(x)  # pre-norm
            x_ffn = (x_ffn + beta_2) * gamma_2  # shift, scale
            x_ffn = self.ffn(x_ffn)  # FFN
            x_ffn = x_ffn * alpha_2  # scale
            x = x + x_ffn
            x = self.dropout(x)
            return x

        # default transformer block without conditioning
        x_att = self.mhsa_norm(x)  # pre-norm
        x_att = self.mhsa(x_att)  # attention
        x = x + x_att

        x_ffn = self.ffn_norm(x)  # pre-norm
        x_ffn = self.ffn(x_ffn)  # FFN
        x = x + x_ffn
        return x

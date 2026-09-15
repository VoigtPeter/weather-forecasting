import math
from typing import Literal

import torch
from torch import nn

from einops import einsum
from einops.layers.torch import EinMix

from wf.modules.gnn import GridGraphConv
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
            add_conditioning: int | None = None,
            dropout: float = 0.0,
            mixer: Literal["mhsa", "afno", "sfno"] = "mhsa",
            graph_conv: bool = False,
            graph_conv_level: int = 2,
            graph_conv_layers: int = 1,
            graph_conv_node_embed: int = 0,
            graph_conv_edge_embed: int = 0,
            activation: type[nn.Module] = nn.GELU,
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

        self.adaLN = None
        self.conditioning = conditioning
        if self.conditioning == "adaLN":
            assert cond_dim is not None, "cond_dim cannot be None"
            lin = nn.Linear(cond_dim, dim * 6, bias=False)
            nn.init.zeros_(lin.weight[(4 * dim):, :])  # init partially with zeros to ensure alpha starts with zeros
            self.adaLN = nn.Sequential(
                FFN(cond_dim, expansion_factor, activation=activation),
                activation(),
                lin,
            )

        self.add_cond_encoder = None
        self.add_conditioning = add_conditioning
        if self.add_conditioning is not None and self.add_conditioning > 0:
            self.add_cond_encoder = nn.Sequential(
                nn.Linear(self.add_conditioning, dim),
                activation(),
                FFN(dim, expansion_factor, activation=activation),
            )

        self.graph_conv = graph_conv
        if self.graph_conv:
            self.conv = GridGraphConv(
                dim,
                nlat,
                nlon,
                num_layers=graph_conv_layers,
                mesh_level=graph_conv_level,
                node_embed=graph_conv_node_embed,
                edge_embed=graph_conv_edge_embed,
                ffn_factor=expansion_factor,
                activation=activation,
            )
        else:
            self.conv = None

        self.dropout = nn.Dropout(p=dropout)


    def forward(
            self, x: torch.Tensor,
            c: torch.Tensor | None = None,
            add_c: torch.Tensor | None = None,
            field_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if self.add_cond_encoder is not None:
            x = x + self.add_cond_encoder(add_c)

        if self.conditioning == "adaLN":
            gamma_1, gamma_2, beta_1, beta_2, alpha_1, alpha_2 = torch.chunk(self.adaLN(c), 6, dim=-1)

            x_att = self.mhsa_norm(x)  # pre-norm
            x_att = (x_att + beta_1) * gamma_1  # shift, scale
            x_att_mixed = self.mixer(x_att, field_size=field_size)  # attention
            if self.graph_conv:
                x_att_conv = self.conv(x_att)
                x_att = x_att_conv + x_att_mixed
            else:
                x_att = x_att_mixed
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
        x_att_mixed = self.mhsa(x_att)  # attention
        if self.graph_conv:
            x_att_conv = self.conv(x_att)
            x_att = x_att_conv + x_att_mixed
        else:
            x_att = x_att_mixed
        x = x + x_att

        x_ffn = self.ffn_norm(x)  # pre-norm
        x_ffn = self.ffn(x_ffn)  # FFN
        x = x + x_ffn
        return x

    def compile(self, *args, **kwargs) -> None:
        if not isinstance(self.mixer, SFNO):
            super().compile(*args, **kwargs)
            return

        self.ffn.compile(*args, **kwargs)
        if self.conv is not None:
            self.conv.compile(*args, **kwargs)
        if self.add_cond_encoder is not None:
            self.add_cond_encoder.compile(*args, **kwargs)
        if self.adaLN is not None:
            self.adaLN.compile(*args, **kwargs)

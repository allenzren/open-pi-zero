import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.half_dim = dim // 2
        self.max_period = max_period

    def forward(self, t: torch.FloatTensor) -> torch.FloatTensor:
        emb = math.log(self.max_period) / (self.half_dim - 1)
        emb = torch.exp(
            torch.arange(self.half_dim, device=t.device, dtype=t.dtype) * -emb
        )
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ActionEncoder(nn.Module):
    """Matching pi0 appendix"""

    def __init__(self, action_dim: int, width: int, time_cond: bool = False):
        super().__init__()
        self.linear_1 = nn.Linear(action_dim, width)
        if time_cond:
            self.linear_2 = nn.Linear(2 * width, width)
        else:
            self.linear_2 = nn.Linear(width, width)
        self.nonlinearity = nn.SiLU()  # swish
        self.linear_3 = nn.Linear(width, width)
        self.time_cond = time_cond

    def forward(
        self,
        action: torch.FloatTensor,
        time_emb: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # [Batch_Size, Seq_Len, Width]
        emb = self.linear_1(action)
        if self.time_cond:
            # repeat time embedding for seq_len
            # [Batch_Size, Seq_Len, Width]
            time_emb_full = time_emb.unsqueeze(1).expand(-1, action.size(1), -1)
            emb = torch.cat([time_emb_full, emb], dim=-1)
        emb = self.nonlinearity(self.linear_2(emb))
        emb = self.linear_3(emb)
        return emb


class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    """

    def __init__(
        self,
        input_dim,
        embed_dim=256,
        scale=10,
    ):
        super(GaussianFourierFeatureTransform, self).__init__()
        self.b = torch.randn(input_dim, embed_dim) * scale
        self.pi = 3.14159265359

    def forward(self, v: torch.FloatTensor) -> torch.FloatTensor:
        x_proj = torch.matmul(2 * self.pi * v, self.b.to(v.device).to(v.dtype))
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], -1)


class AdaptiveRMSNorm(nn.Module):
    def __init__(self, dim: int, dim_cond: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.to_gamma = nn.Sequential(
            nn.Linear(dim_cond, dim),
            nn.Sigmoid(),
        )
        self.to_beta = nn.Linear(dim_cond, dim, bias=False)

    def _norm(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(
        self, x: torch.FloatTensor, cond: torch.FloatTensor
    ) -> torch.FloatTensor:
        output = self._norm(x)
        if cond.ndim == 2:
            cond = rearrange(cond, "b d -> b 1 d")
        gamma = self.to_gamma(cond)
        beta = self.to_beta(cond)
        return output * gamma + beta


class AdaptiveLayerscale(nn.Module):
    def __init__(
        self, dim: int, dim_cond: int, adaln_zero_bias_init_value: float = -2.0
    ):
        super().__init__()
        adaln_zero_gamma_linear = nn.Linear(dim_cond, dim)
        nn.init.zeros_(adaln_zero_gamma_linear.weight)
        nn.init.constant_(adaln_zero_gamma_linear.bias, adaln_zero_bias_init_value)

        self.to_adaln_zero_gamma = adaln_zero_gamma_linear

    def forward(
        self, x: torch.FloatTensor, cond: torch.FloatTensor
    ) -> torch.FloatTensor:
        if cond.ndim == 2:
            cond = rearrange(cond, "b d -> b 1 d")
        gamma = self.to_adaln_zero_gamma(cond)
        return x * gamma.sigmoid()

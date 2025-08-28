"""
Fusion module: implements three fusion strategies:
 - concat (simple early fusion)
 - two_way_gating (two-way gating attention)
 - qkv_cross_attention (small cross-attention block)

All modules accept projected graph embeddings g (N,d') and llm embeddings l (N,d') and optional z (scalar)
and return fused vector f (N, out_dim) which can be classified.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatFusion(nn.Module):
    def __init__(self, d_in: int, hidden: int = 256, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        # d_in is 2 * d' (g and l concat) + optional 1 if z is present; caller ensures shapes
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class TwoWayGatingFusion(nn.Module):
    """
    Two-way gating described in your notes.

    We first compute projections to d', then compute scalar scores e_g and e_l using small MLPs,
    apply softmax over [e_g, e_l] and fuse: a_g * g + a_l * l.
    """

    def __init__(self, d_proj: int, hidden: int = 128, out_dim: int = 128):
        super().__init__()
        # scoring networks
        self.score_g = nn.Sequential(
            nn.Linear(2 * d_proj, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        self.score_l = nn.Sequential(
            nn.Linear(2 * d_proj, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        # final projector
        self.out_proj = nn.Sequential(
            nn.Linear(d_proj, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, g: torch.Tensor, l: torch.Tensor):
        # g, l are (N, d_proj)
        pair_gl = torch.cat([g, l], dim=1)  # (N, 2d)
        pair_lg = torch.cat([l, g], dim=1)
        e_g = self.score_g(pair_gl).squeeze(-1)  # (N,)
        e_l = self.score_l(pair_lg).squeeze(-1)
        a = F.softmax(torch.stack([e_g, e_l], dim=1), dim=1)  # (N,2)
        a_g = a[:, 0].unsqueeze(-1)
        a_l = a[:, 1].unsqueeze(-1)
        hF = a_g * g + a_l * l
        return self.out_proj(hF)


class QKVCrossAttentionFusion(nn.Module):
    """
    Simple single-head cross-attention where g attends to l (or vice-versa).
    We'll implement a symmetric variant: both attend to each other and then combine.
    """

    def __init__(self, d_proj: int, out_dim: int = 128, hidden: int = 128):
        super().__init__()
        self.qg = nn.Linear(d_proj, d_proj)
        self.kl = nn.Linear(d_proj, d_proj)
        self.vl = nn.Linear(d_proj, d_proj)

        self.ql = nn.Linear(d_proj, d_proj)
        self.kg = nn.Linear(d_proj, d_proj)
        self.vg = nn.Linear(d_proj, d_proj)

        self.out = nn.Sequential(
            nn.Linear(2 * d_proj, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim)
        )

        self.scale = d_proj ** 0.5

    def forward(self, g: torch.Tensor, l: torch.Tensor):
        # g attends to l
        qg = self.qg(g)  # (N,d)
        kl = self.kl(l)
        vl = self.vl(l)
        attn_gl = torch.softmax((qg @ kl.T) / self.scale, dim=1)  # (N,N)
        agg_gl = attn_gl @ vl  # (N,d)

        # l attends to g
        ql = self.ql(l)
        kg = self.kg(g)
        vg = self.vg(g)
        attn_lg = torch.softmax((ql @ kg.T) / self.scale, dim=1)
        agg_lg = attn_lg @ vg

        N = g.size(0)
        fused = torch.cat([agg_gl, agg_lg], dim=1)  # (N, 2d)
        return self.out(fused)

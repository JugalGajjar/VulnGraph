"""
Normalize and project embeddings to a common dimension d'.

Functions:
 - l2_normalize_np(embs, eps=1e-12)
 - Projector(nn.Module): learnable linear projection + optional layernorm
 - project_embeddings_np(embs, projector, device) -> torch.Tensor
"""
from typing import Optional
import numpy as np
import torch
import torch.nn as nn


def l2_normalize_np(embs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return embs / norms


class Projector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, use_ln: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.use_ln = use_ln
        if use_ln:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.linear(x)
        if self.use_ln:
            x = self.ln(x)
        return x


def project_embeddings_np(embs_np: np.ndarray, projector: Projector, device: torch.device,
                          batch_size: int = 2048) -> torch.Tensor:
    """
    Projects numpy embeddings via the projector and returns a torch.Tensor on cpu.

    emb_np: (N, D_in)
    returns: torch.FloatTensor (N, D_out) on CPU
    """
    projector = projector.to(device)
    projector.eval()
    embs_t = []
    with torch.no_grad():
        for i in range(0, len(embs_np), batch_size):
            b = embs_np[i:i+batch_size]
            xb = torch.from_numpy(b).float().to(device)
            y = projector(xb)
            embs_t.append(y.cpu())
    return torch.cat(embs_t, dim=0)

"""
gating.py — Modality gating module.

Computes a per-sample softmax trust score over all active modalities:

    g = softmax( MLP([z_1; z_2; ...; z_M]) )  ∈ ℝ^M

The weighted embeddings are then passed to the fusion module.

Notes
-----
- When only ONE modality is active, gating is bypassed (identity).
- g is returned alongside the weighted embeddings so it can be logged.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNetwork(nn.Module):
    """
    Soft modality gating via a shared MLP.

    Parameters
    ----------
    num_modalities : int
        Number of active modalities M.
    embed_dim : int
        Embedding dimension d (all modalities share the same d).
    hidden_dim : int
        Hidden size of the gating MLP.
    dropout : float
    """

    def __init__(
        self,
        num_modalities: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.M = num_modalities
        self.d = embed_dim
        self.bypass = num_modalities == 1

        if not self.bypass:
            in_dim = num_modalities * embed_dim
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_modalities),
            )

    def forward(
        self, embeddings: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Parameters
        ----------
        embeddings : dict   modality_name → [B, d]

        Returns
        -------
        weighted : dict   modality_name → [B, d]   (z_m * g_m)
        gates    : [B, M]  softmax trust scores
        """
        keys = list(embeddings.keys())  # deterministic order
        zs = [embeddings[k] for k in keys]  # list of [B, d]

        if self.bypass:
            gates = torch.ones(zs[0].shape[0], 1, device=zs[0].device)
            weighted = {keys[0]: zs[0]}
            return weighted, gates

        # Concatenate all embeddings
        z_cat = torch.cat(zs, dim=-1)          # [B, M*d]
        logits = self.mlp(z_cat)               # [B, M]
        gates = F.softmax(logits, dim=-1)      # [B, M]

        weighted = {}
        for i, k in enumerate(keys):
            weighted[k] = zs[i] * gates[:, i : i + 1]  # [B, d]

        return weighted, gates

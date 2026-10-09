"""
Generalized Modality Trust Gating for N modalities.

Given a list or arguments of modality embeddings [z_0, z_1, ..., z_{N-1}],
each of shape (B, embed_dim), computes per-sample, per-modality softmax trust
scores and returns scaled gated embeddings.

Key features:
  - Accepts any number of modalities (N >= 1).
  - Supports both list input: gating([z1, z2, ...]) -> (list_of_gated, trust_scores)
    and positional args: gating(z1, z2, ...) -> (g1, g2, ..., trust_scores).
  - Scales each modality by N * alpha_i so that uniform trust (1/N)
    preserves expected embedding magnitude.
"""

from __future__ import annotations
from typing import List, Tuple, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityTrustGating(nn.Module):
    """Computes softmax trust scores for N modalities.

    The gating network takes the concatenation of all N modality embeddings and
    outputs N scalar trust weights (after Softmax) per sample. Each modality
    embedding is then scaled by its trust weight * N to preserve expected
    embedding magnitude when all modalities are equally trusted.
    """

    def __init__(
        self,
        num_modalities: int = 2,
        embed_dim: int = 128,
        hidden_dim: Optional[int] = None,
    ):
        """
        Args:
            num_modalities : Number of modalities the gating network will handle (N).
                             Defaults to 2 for backward compatibility.
            embed_dim      : Dimensionality of each modality embedding.
            hidden_dim     : Hidden size of the MLP gating network.
        """
        super().__init__()
        assert num_modalities >= 1, "num_modalities must be at least 1"
        self.num_modalities = num_modalities
        self.embed_dim = embed_dim
        if hidden_dim is None:
            hidden_dim = max(64, embed_dim // 2)
        self.hidden_dim = hidden_dim

        self.gating_network = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_modalities),
        )

    def forward(
        self, *args, **kwargs
    ) -> Union[Tuple[List[torch.Tensor], torch.Tensor], Tuple[torch.Tensor, ...]]:
        """
        Supports two invocation signatures:
          1. List input:
             gating([z1, z2, ...]) -> (gated_list, trust_scores)
          2. Multi-argument unpacked input:
             gating(z1, z2, ...)   -> (g1, g2, ..., trust_scores)
        """
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            embeddings = list(args[0])
            is_unpacked_call = False
        elif len(args) > 0:
            embeddings = list(args)
            is_unpacked_call = True
        elif "embeddings" in kwargs:
            embeddings = list(kwargs["embeddings"])
            is_unpacked_call = False
        else:
            raise ValueError("ModalityTrustGating expects embeddings as a list or positional arguments.")

        assert len(embeddings) == self.num_modalities, (
            f"Expected {self.num_modalities} modality embeddings, got {len(embeddings)}"
        )

        # Concatenate all modality embeddings: (B, N * embed_dim)
        combined = torch.cat(embeddings, dim=-1)

        # Compute trust scores: (B, N)
        logits = self.gating_network(combined)
        trust_scores = F.softmax(logits, dim=-1)  # (B, N)

        # Scale factor: N * alpha_i so uniform trust preserves magnitude
        N = float(self.num_modalities)
        gated = []
        for i, z in enumerate(embeddings):
            alpha_i = trust_scores[:, i : i + 1]  # (B, 1)
            gated.append(N * alpha_i * z)

        if is_unpacked_call:
            # Return tuple compatible with `g_img, g_aud, trust = gating(z_img, z_aud)`
            return (*gated, trust_scores)
        return gated, trust_scores

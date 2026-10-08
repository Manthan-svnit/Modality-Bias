"""
Generalized Modality Trust Gating for N modalities.

Given a list of modality embeddings [z_0, z_1, ..., z_{N-1}], each of shape (B, embed_dim),
computes per-sample, per-modality softmax trust scores and returns scaled gated embeddings.

Key generalization over the original 2-modality version:
  - Accepts any number of modalities (N >= 1).
  - Concatenates all present embeddings, passes through a shared gating MLP,
    and applies per-modality Softmax normalization.
  - Scales each modality by N * alpha_i so that when all modalities are equally
    trusted (1/N each), the magnitude is preserved (analogous to the 2.0 factor
    in the original 2-modality version).
"""

from typing import List, Tuple
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

    def __init__(self, num_modalities: int, embed_dim: int = 128, hidden_dim: int = 64):
        """
        Args:
            num_modalities : Number of modalities the gating network will handle (N).
                             Must match the number of embeddings passed to forward().
            embed_dim      : Dimensionality of each modality embedding.
            hidden_dim     : Hidden size of the MLP gating network.
        """
        super().__init__()
        assert num_modalities >= 1, "num_modalities must be at least 1"
        self.num_modalities = num_modalities
        self.embed_dim = embed_dim

        self.gating_network = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_modalities),
        )

    def forward(
        self, embeddings: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            embeddings: List of N tensors, each of shape (B, embed_dim).
                        All tensors must be on the same device.

        Returns:
            gated_embeddings : List of N tensors (B, embed_dim) -- trust-weighted embeddings.
            trust_scores     : FloatTensor (B, N) -- per-sample per-modality trust weights.
        """
        assert len(embeddings) == self.num_modalities, (
            f"Expected {self.num_modalities} modality embeddings, got {len(embeddings)}"
        )

        # Concatenate all modality embeddings: (B, N * embed_dim)
        combined = torch.cat(embeddings, dim=-1)

        # Compute trust scores: (B, N)
        logits = self.gating_network(combined)
        trust_scores = F.softmax(logits, dim=-1)  # (B, N)

        # Scale factor: N * alpha_i so that uniform trust preserves magnitude
        N = float(self.num_modalities)
        gated = []
        for i, z in enumerate(embeddings):
            alpha_i = trust_scores[:, i : i + 1]  # (B, 1)
            gated.append(N * alpha_i * z)

        return gated, trust_scores

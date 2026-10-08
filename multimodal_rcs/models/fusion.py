"""
Generalized N-Modality Cross-Attention Fusion.

Replaces the hardcoded 2-modality (image, audio) CrossAttentionFusion with a
dynamic module that:

  1. Accepts a list of N modality embeddings.
  2. Applies pairwise cross-attention between all unique (i, j) modality pairs.
  3. Aggregates the refined embeddings (mean pooling).
  4. Projects to a fixed embed_dim output vector.

Supports both Early and Late fusion variants for arbitrary N.

--- Architecture ---

EarlyFusionModule  (N modalities):
  Cross-attention applied directly after encoders and trust gating.

LateFusionModule   (N modalities):
  Each modality first passes through a dedicated intermediate FFN before
  cross-attention, allowing deeper modality-specific processing.
"""

from typing import List, Tuple
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Pairwise Cross-Attention Layer
# ---------------------------------------------------------------------------

class PairwiseCrossAttentionLayer(nn.Module):
    """Applies bidirectional cross-attention between every pair of modalities.

    For N modalities: processes N*(N-1) directed attention operations.
    Each modality i attends to all other modalities j, accumulating context
    through residual connections and a position-wise FFN.
    """

    def __init__(self, num_modalities: int, dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_modalities = num_modalities
        self.dim = dim

        # For each (query_modality, key_modality) pair where i != j,
        # create a dedicated cross-attention + FFN block.
        # Stored in a flat ModuleList indexed by i * N + j.
        N = num_modalities
        self.cross_attns = nn.ModuleList()
        self.attn_norms = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()

        for i in range(N):
            for j in range(N):
                if i == j:
                    # Placeholder — not used
                    self.cross_attns.append(nn.Identity())
                    self.attn_norms.append(nn.Identity())
                    self.ffns.append(nn.Identity())
                    self.ffn_norms.append(nn.Identity())
                else:
                    self.cross_attns.append(
                        nn.MultiheadAttention(
                            embed_dim=dim, num_heads=num_heads,
                            dropout=dropout, batch_first=True
                        )
                    )
                    self.attn_norms.append(nn.LayerNorm(dim))
                    self.ffns.append(nn.Sequential(
                        nn.Linear(dim, dim * 2),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(dim * 2, dim),
                        nn.Dropout(dropout),
                    ))
                    self.ffn_norms.append(nn.LayerNorm(dim))

        self.drop = nn.Dropout(dropout)

    def _idx(self, i: int, j: int) -> int:
        return i * self.num_modalities + j

    def forward(self, modalities: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            modalities: List of N tensors, each (B, 1, dim)
        Returns:
            Updated list of N tensors, each (B, 1, dim)
        """
        N = self.num_modalities
        updated = list(modalities)  # Start with current values

        # Accumulate cross-attention updates independently (all from previous step)
        updates = [torch.zeros_like(z) for z in modalities]
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                idx = self._idx(i, j)
                q = modalities[i]   # (B, 1, dim)
                k = modalities[j]   # (B, 1, dim)
                attn_out, _ = self.cross_attns[idx](query=q, key=k, value=k)
                updates[i] = updates[i] + self.drop(attn_out)

        # Apply accumulated residual + FFN
        for i in range(N):
            # Count how many other modalities attended to i
            num_others = N - 1
            avg_update = updates[i] / max(1, num_others)
            h = self.attn_norms[self._idx(i, (i + 1) % N)](modalities[i] + avg_update)
            ffn_out = self.ffns[self._idx(i, (i + 1) % N)](h)
            h = self.ffn_norms[self._idx(i, (i + 1) % N)](h + ffn_out)
            updated[i] = h

        return updated


# ---------------------------------------------------------------------------
# Cross-Attention Fusion Stack
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """Stack of PairwiseCrossAttentionLayers followed by mean-pool and projection.

    Output shape: (B, dim) regardless of number of modalities N.
    """

    def __init__(
        self,
        num_modalities: int,
        dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.dim = dim

        self.layers = nn.ModuleList([
            PairwiseCrossAttentionLayer(
                num_modalities=num_modalities,
                dim=dim, num_heads=num_heads, dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Project concatenated/mean-pooled modalities to embed_dim
        self.fusion_projection = nn.Sequential(
            nn.Linear(dim * num_modalities, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            embeddings: List of N tensors, each (B, dim)
        Returns:
            z_fused: (B, dim)
        """
        # Expand token dimension for cross-attention: (B, 1, dim)
        h = [z.unsqueeze(1) for z in embeddings]

        for layer in self.layers:
            h = layer(h)

        # Squeeze and concat: (B, N * dim)
        h_flat = [x.squeeze(1) for x in h]
        concat = torch.cat(h_flat, dim=-1)

        return self.fusion_projection(concat)


# ---------------------------------------------------------------------------
# Early and Late Fusion Modules (generalized to N modalities)
# ---------------------------------------------------------------------------

class EarlyFusionModule(nn.Module):
    """Early Fusion: Cross-attention directly after encoders and trust gating.

    Supports N modalities. Replaces the original 2-modality EarlyFusionModule.
    """

    def __init__(
        self,
        num_modalities: int,
        dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_attention = CrossAttentionFusion(
            num_modalities=num_modalities,
            dim=dim, num_heads=num_heads,
            num_layers=num_layers, dropout=dropout,
        )

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            embeddings: List of N tensors, each (B, dim)
        Returns:
            z_fused: (B, dim)
        """
        return self.cross_attention(embeddings)


class LateFusionModule(nn.Module):
    """Late Fusion: Each modality passes through its own intermediate FFN
    before cross-attention, allowing deeper modality-specific processing.

    Supports N modalities. Replaces the original 2-modality LateFusionModule.
    """

    def __init__(
        self,
        num_modalities: int,
        dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # One intermediate FFN per modality
        self.intermediate_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
            )
            for _ in range(num_modalities)
        ])

        self.cross_attention = CrossAttentionFusion(
            num_modalities=num_modalities,
            dim=dim, num_heads=num_heads,
            num_layers=num_layers, dropout=dropout,
        )

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            embeddings: List of N tensors, each (B, dim)
        Returns:
            z_fused: (B, dim)
        """
        h = [ffn(z) for ffn, z in zip(self.intermediate_ffns, embeddings)]
        return self.cross_attention(h)

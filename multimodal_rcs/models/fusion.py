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
        for i in range(N):
            for j in range(N):
                if i == j:
                    self.cross_attns.append(nn.Identity())
                else:
                    self.cross_attns.append(
                        nn.MultiheadAttention(
                            embed_dim=dim, num_heads=num_heads,
                            dropout=dropout, batch_first=True
                        )
                    )

        # One post-attention norm, FFN, and FFN norm per modality
        self.attn_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(N)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 2, dim),
                nn.Dropout(dropout),
            )
            for _ in range(N)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(N)])
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
        if N == 1:
            return modalities

        updated = list(modalities)

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

        # Apply accumulated residual + FFN per modality
        for i in range(N):
            num_others = N - 1
            avg_update = updates[i] / max(1, num_others)
            h = self.attn_norms[i](modalities[i] + avg_update)
            ffn_out = self.ffns[i](h)
            h = self.ffn_norms[i](h + ffn_out)
            updated[i] = h

        return updated


# ---------------------------------------------------------------------------
# Cross-Attention Fusion Stack
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Cross-Attention Fusion Stack
# ---------------------------------------------------------------------------

def _extract_embeddings_arg(*args, **kwargs) -> List[torch.Tensor]:
    """Helper to support both list input and unpacked *args inputs."""
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        return list(args[0])
    if len(args) > 0:
        return list(args)
    if "embeddings" in kwargs:
        return list(kwargs["embeddings"])
    raise ValueError("Expected modality embeddings as a list or positional arguments.")


class CrossAttentionFusion(nn.Module):
    """Stack of PairwiseCrossAttentionLayers followed by projection.

    Output shape: (B, dim) regardless of number of modalities N.
    """

    def __init__(
        self,
        num_modalities: int = 2,
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

        # Project concatenated modalities to embed_dim
        self.fusion_projection = nn.Sequential(
            nn.Linear(dim * num_modalities, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Args:
            embeddings: List of N tensors, each (B, dim), OR positional *args
        Returns:
            z_fused: (B, dim)
        """
        embeddings = _extract_embeddings_arg(*args, **kwargs)
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

    Supports N modalities.
    """

    def __init__(
        self,
        num_modalities: int = 2,
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

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Args:
            embeddings: List of N tensors, each (B, dim), or unpacked *args
        Returns:
            z_fused: (B, dim)
        """
        embeddings = _extract_embeddings_arg(*args, **kwargs)
        return self.cross_attention(embeddings)


class LateFusionModule(nn.Module):
    """Late Fusion: Each modality passes through its own intermediate FFN
    before cross-attention, allowing deeper modality-specific processing.

    Supports N modalities.
    """

    def __init__(
        self,
        num_modalities: int = 2,
        dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_modalities = num_modalities

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

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Args:
            embeddings: List of N tensors, each (B, dim), or unpacked *args
        Returns:
            z_fused: (B, dim)
        """
        embeddings = _extract_embeddings_arg(*args, **kwargs)
        h = [ffn(z) for ffn, z in zip(self.intermediate_ffns, embeddings)]
        return self.cross_attention(h)


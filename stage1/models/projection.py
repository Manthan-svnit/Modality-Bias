"""
projection.py — Projection heads, decoders, and latent classifier.
Exact architecture aligned with the actual RCS paper (Mondal et al., 2026).

- Projection heads: z → p (used by contrastive losses)
- Decoders       : z_fused → z_m with BatchNorm layers (MSE reconstruction)
- ClassifierHead : 4-hidden-layer MLP with BatchNorm layers on z_fused
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head (SimCLR / SupCon style).
    z ∈ ℝ^d  →  p ∈ ℝ^proj_dim  (L2-normalised)
    """

    def __init__(self, in_dim: int, proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, in_dim] → p: [B, proj_dim] (L2-normalised)"""
        return F.normalize(self.mlp(z), dim=-1)


class ModalityDecoder(nn.Module):
    """
    RCS-aligned Decoder with BatchNorm: z_fused ∈ ℝ^d → z_m ∈ ℝ^d
    Per actual RCS paper (Section 4.2):
    "The Decoder network consisted of the deconvolution layers and the batch normalization layers."
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        cur = in_dim
        for _ in range(num_layers):
            layers += [
                nn.Linear(cur, hidden_dim),
                SafeBatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            cur = hidden_dim
        layers.append(nn.Linear(cur, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z_fused: torch.Tensor) -> torch.Tensor:
        """z_fused: [B, in_dim] → reconstructed: [B, out_dim]"""
        return self.mlp(z_fused)


class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.size(0) == 1:
            return x
        return super().forward(x)


class ClassifierHead(nn.Module):
    """
    RCS-aligned 4-hidden-layer MLP Latent Classifier on top of z_fused.
    Per actual RCS paper (Section 4.2):
    "The Latent Classifier Network consisted of a Multi-layered Perceptron with 4 hidden layers."
    """

    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.2, hidden_dim: int = 256):
        super().__init__()
        if isinstance(dropout, int) and dropout > 1 and (isinstance(hidden_dim, float) or hidden_dim <= 1):
            hidden_dim, dropout = dropout, hidden_dim
        hidden_dim = int(hidden_dim) if hidden_dim else 256
        dropout = float(dropout) if dropout is not None else 0.2

        layers: List[nn.Module] = []
        cur = in_dim
        for _ in range(4):
            layers += [
                nn.Linear(cur, hidden_dim),
                SafeBatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            cur = hidden_dim
        layers.append(nn.Linear(cur, num_classes))
        self.clf = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, in_dim] → logits: [B, num_classes]"""
        return self.clf(z)

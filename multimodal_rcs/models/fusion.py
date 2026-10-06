import torch
import torch.nn as nn
from typing import Tuple


class CrossAttentionLayer(nn.Module):
    """Bidirectional Cross-Attention Layer between two modalities with residual connections and FFN."""

    def __init__(self, dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # Image attends to Audio
        self.attn_img_to_aud = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_img_attn = nn.LayerNorm(dim)
        self.ffn_img = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm_img_ffn = nn.LayerNorm(dim)

        # Audio attends to Image
        self.attn_aud_to_img = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_aud_attn = nn.LayerNorm(dim)
        self.ffn_aud = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm_aud_ffn = nn.LayerNorm(dim)

        self.drop = nn.Dropout(dropout)

    def forward(
        self, z_img: torch.Tensor, z_aud: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_img: (B, 1, dim)
            z_aud: (B, 1, dim)
        Returns:
            Tuple of updated (z_img, z_aud) each (B, 1, dim)
        """
        # 1. Image attends to Audio
        out_img_attn, _ = self.attn_img_to_aud(query=z_img, key=z_aud, value=z_aud)
        z_img = self.norm_img_attn(z_img + self.drop(out_img_attn))
        z_img = self.norm_img_ffn(z_img + self.ffn_img(z_img))

        # 2. Audio attends to Image
        out_aud_attn, _ = self.attn_aud_to_img(query=z_aud, key=z_img, value=z_img)
        z_aud = self.norm_aud_attn(z_aud + self.drop(out_aud_attn))
        z_aud = self.norm_aud_ffn(z_aud + self.ffn_aud(z_aud))

        return z_img, z_aud


class CrossAttentionFusion(nn.Module):
    """Stack of 2 Cross-Attention Layers (4 heads, dim 128, dropout 0.1) yielding fused 128-d vector."""

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CrossAttentionLayer(dim=dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.fusion_projection = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, z_img: torch.Tensor, z_aud: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_img: (B, dim)
            z_aud: (B, dim)
        Returns:
            z_fused: (B, dim)
        """
        # Expand token dimension: (B, 1, dim)
        h_img = z_img.unsqueeze(1)
        h_aud = z_aud.unsqueeze(1)

        for layer in self.layers:
            h_img, h_aud = layer(h_img, h_aud)

        # Squeeze token dimension: (B, dim)
        h_img = h_img.squeeze(1)
        h_aud = h_aud.squeeze(1)

        # Merge representations
        concat = torch.cat([h_img, h_aud], dim=-1)  # (B, dim * 2)
        z_fused = self.fusion_projection(concat)     # (B, dim)
        return z_fused


class EarlyFusionModule(nn.Module):
    """Variant 1 (Early Fusion): Cross-attention directly applied after encoders and trust gating."""

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_attention = CrossAttentionFusion(
            dim=dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout
        )

    def forward(self, z_img: torch.Tensor, z_aud: torch.Tensor) -> torch.Tensor:
        return self.cross_attention(z_img, z_aud)


class LateFusionModule(nn.Module):
    """Variant 2 (Late Fusion): Intermediate feed-forward processing on each modality before cross-attention."""

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Intermediate feed-forward processing on each modality
        self.img_intermediate_ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.aud_intermediate_ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.cross_attention = CrossAttentionFusion(
            dim=dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout
        )

    def forward(self, z_img: torch.Tensor, z_aud: torch.Tensor) -> torch.Tensor:
        # Intermediate modality representations
        h_img = self.img_intermediate_ffn(z_img)
        h_aud = self.aud_intermediate_ffn(z_aud)
        # Cross-attention fusion
        return self.cross_attention(h_img, h_aud)

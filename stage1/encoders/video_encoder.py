"""
video_encoder.py — Per-frame CNN + temporal pooling encoder for video.

Input : [B, max_frames, 3, H, W]
Output: [B, d]   (L2-normalised embedding)

Architecture:
  - Shared per-frame CNN (same structure as ImageEncoder) applied to every frame
  - Temporal pooling: "mean" | "max" | "attention"
  - Linear → dropout → L2-norm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from stage1.encoders.registry import register_encoder


def _frame_cnn_block(in_ch: int, out_ch: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Dropout2d(dropout),
    )


class TemporalAttentionPool(nn.Module):
    """Soft attention over the T-frame dimension."""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.attn = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, T, feat_dim]
        returns: [B, feat_dim]
        """
        weights = self.attn(x)              # [B, T, 1]
        weights = torch.softmax(weights, dim=1)
        return (x * weights).sum(dim=1)     # [B, feat_dim]


@register_encoder("video")
class VideoEncoder(nn.Module):
    """
    Per-frame CNN encoder with configurable temporal pooling.

    Parameters from ``encoder_cfg``:
      - ``d``                         : output embedding dimension
      - ``video.frame_cnn_depth``     : per-frame CNN depth
      - ``video.frame_cnn_width``     : base channels
      - ``video.temporal_pool``       : "mean" | "max" | "attention"
      - ``video.dropout``             : dropout rate
    """

    def __init__(self, encoder_cfg):
        super().__init__()
        d = encoder_cfg.d
        depth = encoder_cfg.video.frame_cnn_depth
        width = encoder_cfg.video.frame_cnn_width
        dropout = encoder_cfg.video.dropout
        self.pool_mode: str = getattr(encoder_cfg.video, "temporal_pool", "mean")

        # Build per-frame CNN
        layers = []
        in_ch = 3
        out_ch = width
        for i in range(depth):
            layers.append(_frame_cnn_block(in_ch, out_ch, dropout))
            in_ch = out_ch
            if (i + 1) % 2 == 0:
                out_ch = min(out_ch * 2, 512)

        self.frame_cnn = nn.Sequential(*layers)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = in_ch

        # Temporal pooling
        if self.pool_mode == "attention":
            self.temporal_pool = TemporalAttentionPool(self.feat_dim)

        # Projection to d
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, d),
            nn.Dropout(dropout),
        )
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, T, 3, H, W]

        Returns
        -------
        z : [B, d]  (L2-normalised)
        """
        B, T, C, H, W = x.shape

        # Merge batch and time for efficient parallel CNN pass
        x_flat = x.view(B * T, C, H, W)             # [B*T, 3, H, W]
        feat = self.frame_cnn(x_flat)                # [B*T, feat_dim, h, w]
        feat = self.spatial_pool(feat).flatten(1)    # [B*T, feat_dim]
        feat = feat.view(B, T, self.feat_dim)        # [B, T, feat_dim]

        # Temporal aggregation
        if self.pool_mode == "mean":
            agg = feat.mean(dim=1)
        elif self.pool_mode == "max":
            agg, _ = feat.max(dim=1)
        elif self.pool_mode == "attention":
            agg = self.temporal_pool(feat)
        else:
            raise ValueError(f"Unknown temporal_pool mode: {self.pool_mode}")

        z = self.proj(agg)                           # [B, d]
        return F.normalize(z, dim=-1)

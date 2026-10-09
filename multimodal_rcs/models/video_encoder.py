"""
Video Encoder: Spatiotemporal 3D / R(2+1)D CNN mapping video volumes
(B, C, T, H, W) of arbitrary frame count and spatial resolution
to a fixed embed_dim-dimensional embedding.

Uses spatiotemporal decomposed convolutions (spatial 1x3x3 + temporal 3x1x1)
and AdaptiveAvgPool3d((1, 1, 1)) to achieve temporal and spatial resolution
invariance, projecting seamlessly to the pipeline's uniform embedding space (e.g. d=128).
"""

from __future__ import annotations
import torch
import torch.nn as nn


class SpatioTemporalConvBlock(nn.Module):
    """R(2+1)D style decomposed spatiotemporal convolution block:
    Spatial Conv (1 x K x K) followed by Temporal Conv (K x 1 x 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        inter_channels: int | None = None,
        stride: int | tuple[int, int, int] = 1,
    ):
        super().__init__()
        if inter_channels is None:
            # Standard R(2+1)D middle dimension calculation
            inter_channels = max(in_channels, out_channels)

        if isinstance(stride, int):
            spatial_stride = (1, stride, stride)
            temporal_stride = (stride, 1, 1)
        else:
            temporal_stride = (stride[0], 1, 1)
            spatial_stride = (1, stride[1], stride[2])

        # 1. Spatial convolution (1 x 3 x 3)
        self.spatial_conv = nn.Conv3d(
            in_channels,
            inter_channels,
            kernel_size=(1, 3, 3),
            stride=spatial_stride,
            padding=(0, 1, 1),
            bias=False,
        )
        self.bn_s = nn.BatchNorm3d(inter_channels)
        self.relu_s = nn.ReLU(inplace=True)

        # 2. Temporal convolution (3 x 1 x 1)
        self.temporal_conv = nn.Conv3d(
            inter_channels,
            out_channels,
            kernel_size=(3, 1, 1),
            stride=temporal_stride,
            padding=(1, 0, 0),
            bias=False,
        )
        self.bn_t = nn.BatchNorm3d(out_channels)
        self.relu_t = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu_s(self.bn_s(self.spatial_conv(x)))
        out = self.relu_t(self.bn_t(self.temporal_conv(out)))
        return out


class VideoEncoder(nn.Module):
    """Dataset-agnostic Spatiotemporal 3D CNN video encoder.

    Accepts video volumes of shape (B, C, T, H, W) for any C, T, H, W
    and produces a fixed (B, embed_dim) embedding.
    AdaptiveAvgPool3d((1, 1, 1)) guarantees dimensional invariance.
    """

    def __init__(self, embed_dim: int = 128, in_channels: int = 3):
        """
        Args:
            embed_dim   : Dimensionality of the output embedding (e.g. 128).
            in_channels : Number of input color channels (3 for RGB, 1 for grayscale).
        """
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.features = nn.Sequential(
            # Stage 1: Initial spatiotemporal feature extraction
            SpatioTemporalConvBlock(in_channels, 32, stride=1),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),  # (T, H/2, W/2)

            # Stage 2: Temporal + spatial downsampling
            SpatioTemporalConvBlock(32, 64, stride=1),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),  # (T/2, H/4, W/4)

            # Stage 3: Deep spatiotemporal features
            SpatioTemporalConvBlock(64, 128, stride=1),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),  # (T/4, H/8, W/8)

            # Stage 4: High-level representation
            SpatioTemporalConvBlock(128, 128, stride=1),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # (B, 128, 1, 1, 1) -- fully resolution invariant
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor of shape (B, in_channels, T, H, W).
               Supports variable T (frames) and spatial dimensions (H, W).
        Returns:
            Video embedding of shape (B, embed_dim).
        """
        if x.dim() == 4:
            # In case (B, T, H, W) single channel is passed
            x = x.unsqueeze(1)
        feats = self.features(x)
        embed = self.fc(feats)
        return embed

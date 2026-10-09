"""
Video Decoder: Spatiotemporal 3D Transposed CNN reconstructing video volumes
(B, out_channels, T, H, W) from a fixed embed_dim-dimensional fused latent vector.

Takes z_fused (B, embed_dim) and reconstructs video spatiotemporal representations
via 3D transposed convolutions and trilinear interpolation to target dimensions.
Compatible with MSE reconstruction loss in MultiTaskLoss.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoDecoder(nn.Module):
    """Reconstructs a spatiotemporal video tensor from a 128-d latent representation.

    Default output resolution: (B, out_channels, 8, 28, 28).
    Supports dynamic target temporal and spatial dimensions (T, H, W).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        out_channels: int = 3,
        out_frames: int = 8,
        out_height: int = 28,
        out_width: int = 28,
        use_sigmoid: bool = False,
    ):
        """
        Args:
            embed_dim    : Dimensionality of the input latent vector.
            out_channels : Number of output video channels (e.g. 3 for RGB, 1 for grayscale).
            out_frames   : Number of reconstructed video frames (T).
            out_height   : Reconstructed spatial height (H).
            out_width    : Reconstructed spatial width (W).
            use_sigmoid  : Whether to apply Sigmoid activation to clamp values in [0, 1].
        """
        super().__init__()
        self.out_channels = out_channels
        self.out_frames = out_frames
        self.out_height = out_height
        self.out_width = out_width
        self.use_sigmoid = use_sigmoid

        # Project 1D vector into initial 3D spatiotemporal grid (64 channels, 2 frames, 7x7 spatial)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64 * 2 * 7 * 7),
            nn.ReLU(inplace=True),
        )

        # Transposed 3D convolutions:
        # (B, 64, 2, 7, 7) -> (B, 32, 4, 14, 14) -> (B, out_channels, 8, 28, 28)
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(
                64, 32,
                kernel_size=(4, 4, 4),
                stride=(2, 2, 2),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(
                32, out_channels,
                kernel_size=(4, 4, 4),
                stride=(2, 2, 2),
                padding=(1, 1, 1),
            ),
        )

    def forward(
        self,
        z: torch.Tensor,
        target_shape: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z            : (B, embed_dim) fused latent vector.
            target_shape : Optional tuple (T, H, W) to resize the output to match
                           exact video input dimensions for MSE loss.
        Returns:
            Reconstructed video: (B, out_channels, T, H, W).
        """
        h = self.fc(z)
        h = h.view(-1, 64, 2, 7, 7)
        recon = self.deconv(h)

        target_t = target_shape[0] if target_shape is not None else self.out_frames
        target_h = target_shape[1] if target_shape is not None else self.out_height
        target_w = target_shape[2] if target_shape is not None else self.out_width

        if recon.shape[2:] != (target_t, target_h, target_w):
            recon = F.interpolate(
                recon,
                size=(target_t, target_h, target_w),
                mode="trilinear",
                align_corners=False,
            )

        if self.use_sigmoid:
            recon = torch.sigmoid(recon)

        return recon

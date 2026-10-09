"""
Audio Encoder: 2D CNN mapping spectrograms of arbitrary size and channel count
to a fixed embed_dim-dimensional embedding.

Uses AdaptiveAvgPool2d to handle any input spectrogram resolution (e.g. 112x112,
64x64, mel-spectrogram of shape 128x256 etc.), making this encoder dataset-agnostic.
"""

import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """Dataset-agnostic 2D CNN audio spectrogram encoder.

    Accepts spectrograms of shape (B, C, H, W) for any C, H, W and produces
    a (B, embed_dim) embedding. AdaptiveAvgPool2d(1,1) ensures spatial
    invariance to input resolution.
    """

    def __init__(self, embed_dim: int = 128, in_channels: int = 1):
        """
        Args:
            embed_dim   : Dimensionality of the output embedding.
            in_channels : Number of input channels (1 for single-channel spectrograms,
                          3 for log-mel + delta + delta-delta, etc.).
        """
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 128, 1, 1) -- resolution-invariant
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor of shape (B, in_channels, H, W) -- any H, W >= 16.
        Returns:
            Embedding of shape (B, embed_dim)
        """
        feats = self.conv_net(x)
        embed = self.fc(feats)
        return embed

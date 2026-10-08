"""
Reconstruction Decoders for Image, Audio, and Text modalities.

Each decoder takes the fused latent vector z_fused (B, embed_dim) and
reconstructs the original modality representation.

Image/Audio Decoders: Reconstruct pixel/spectrogram grids via transposed convolutions.
Text Decoder       : Reconstructs the mean token embedding (in embedding space).
                     See text_decoder.py for the canonical TextDecoder definition.
"""

import torch
import torch.nn as nn


class ImageDecoder(nn.Module):
    """Reconstructs an image tensor from a 128-d latent representation.

    The decoder up-samples from a 7x7 spatial grid to 28x28 by default.
    Using AdaptiveAvgPool + ConvTranspose sequence keeps it compatible with
    any target resolution through the transposed convolutions.

    Output shape: (B, out_channels, 28, 28)
    """

    def __init__(self, embed_dim: int = 128, out_channels: int = 1):
        """
        Args:
            embed_dim   : Dimensionality of the input latent vector.
            out_channels: Number of output image channels (1=grayscale, 3=RGB).
        """
        super().__init__()
        self.out_channels = out_channels
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64 * 7 * 7),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 14x14
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),     # 28x28
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, embed_dim)
        Returns:
            Reconstructed image: (B, out_channels, 28, 28)
        """
        h = self.fc(z)
        h = h.view(-1, 64, 7, 7)
        recon = self.deconv(h)
        return recon


class AudioDecoder(nn.Module):
    """Reconstructs a spectrogram tensor from a 128-d latent representation.

    Outputs a (B, out_channels, 112, 112) spectrogram by default.
    Sigmoid activation normalizes values to [0, 1] (normalized spectrograms).
    """

    def __init__(self, embed_dim: int = 128, out_channels: int = 1):
        """
        Args:
            embed_dim   : Dimensionality of the input latent vector.
            out_channels: Number of output spectrogram channels.
        """
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64 * 7 * 7),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),  # 56x56
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1),    # 112x112
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, embed_dim)
        Returns:
            Reconstructed spectrogram: (B, out_channels, 112, 112)
        """
        h = self.fc(z)
        h = h.view(-1, 64, 7, 7)
        recon = self.deconv(h)
        return recon

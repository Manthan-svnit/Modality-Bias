import torch
import torch.nn as nn


class ImageDecoder(nn.Module):
    """Reconstructs 28x28 grayscale images from a 128-d latent representation."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64 * 7 * 7),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),  # (B, 32, 14, 14)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),                # (B, 1, 28, 28)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, embed_dim)
        Returns:
            Reconstructed image: (B, 1, 28, 28)
        """
        h = self.fc(z)
        h = h.view(-1, 64, 7, 7)
        recon = self.deconv(h)
        return recon


class AudioDecoder(nn.Module):
    """Reconstructs 112x112 spectrograms from a 128-d latent representation."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64 * 7 * 7),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),  # (B, 64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),  # (B, 32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),  # (B, 16, 56, 56)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),                # (B, 1, 112, 112)
            nn.Sigmoid(),  # Spectrograms are normalized in [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, embed_dim)
        Returns:
            Reconstructed spectrogram: (B, 1, 112, 112)
        """
        h = self.fc(z)
        h = h.view(-1, 64, 7, 7)
        recon = self.deconv(h)
        return recon

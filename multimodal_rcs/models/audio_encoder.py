import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """2D CNN mapping spectrograms (1, 112, 112) to a 128-dimensional embedding."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),  # (B, 32, 56, 56)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  # (B, 64, 28, 28)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),  # (B, 128, 14, 14)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),  # (B, 128, 7, 7)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 128, 1, 1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 1, 112, 112)
        Returns:
            Embedding of shape (B, embed_dim)
        """
        feats = self.conv_net(x)
        embed = self.fc(feats)
        return embed

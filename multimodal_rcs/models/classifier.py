import torch
import torch.nn as nn


class Stage3Classifier(nn.Module):
    """Stage 3 Final Classifier:
    2 hidden layers (in_dim=128 -> 256 -> 128, dropout 0.1) mapping to 10 classes.
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden1: int = 256,
        hidden2: int = 128,
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Fused features tensor of shape (B, in_dim)
        Returns:
            Class logits of shape (B, num_classes)
        """
        return self.net(x)

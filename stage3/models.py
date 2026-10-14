"""
models.py — Stage 3 Final MLP Classifier.
Per actual RCS paper (Section 4.2):
"For classification of the oversampled feature vectors, we used a Multi-layered
Perceptron, trained over 100 epochs with Cross-entropy loss with Adam optimizer,
with a learning rate of 1e-3. The Latent Classifier Network consisted of a
Multi-layered Perceptron with 4 hidden layers."
"""

from typing import List, Optional
import torch
import torch.nn as nn


class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.size(0) == 1:
            return x
        return super().forward(x)

class MLPClassifier(nn.Module):
    """
    RCS 4-hidden-layer MLP Classifier for balanced z_fused representations.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.2,
        activation: str = "relu",
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        act_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "leaky_relu": nn.LeakyReLU,
        }
        act_fn = act_map.get(activation.lower(), nn.ReLU)

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(SafeBatchNorm1d(hidden_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final classification head
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

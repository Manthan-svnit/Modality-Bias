"""
Text Decoder: Reconstructs a mean-pooled token embedding from the fused latent vector z_fused.

Since text reconstruction to discrete tokens (language model decoding) is not the goal here,
the decoder reconstructs the TEXT ENCODER'S INPUT SPACE -- specifically the dense embedding
that the token embedding layer would have produced (before positional encoding and transformer),
averaged over the real (non-padding) tokens. This is a regression target in embedding space
rather than a token-id prediction, which:

  - Avoids requiring a generative LM head (vocab-scale softmax)
  - Remains differentiable and directly compatible with MSE reconstruction loss
  - Preserves the same training contract as the ImageDecoder and AudioDecoder

Input:  z      : FloatTensor (B, embed_dim)
Output: recon  : FloatTensor (B, embed_dim)   -- reconstructed mean token embedding
"""

import torch
import torch.nn as nn


class TextDecoder(nn.Module):
    """Reconstructs a mean token embedding from the fused latent vector.

    Architecture:
        z (B, embed_dim)
        -> Linear(embed_dim, ff_dim) -> GELU
        -> Linear(ff_dim, ff_dim)    -> GELU -> Dropout
        -> Linear(ff_dim, embed_dim) -> LayerNorm

    The output matches the shape of the mean pooled token embedding produced
    by TextEncoder.token_embedding, enabling MSE reconstruction supervision.
    """

    def __init__(self, embed_dim: int = 128, ff_dim: int = 256, dropout: float = 0.1):
        """
        Args:
            embed_dim : Must match the pipeline embed_dim and TextEncoder's embed_dim.
            ff_dim    : Hidden layer width (default 2x embed_dim).
            dropout   : Dropout probability on the hidden layer.
        """
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z    : (B, embed_dim) -- fused latent vector from Stage1Model
        Returns:
            recon: (B, embed_dim) -- reconstructed text embedding in embedding space
        """
        return self.decoder(z)

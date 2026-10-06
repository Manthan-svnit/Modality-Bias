from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityTrustGating(nn.Module):
    """Computes softmax trust scores between image and audio representations before fusion.
    
    Given image embedding z_img and audio embedding z_aud, computes normalized
    trust weights alpha_img, alpha_aud in [0, 1] summing to 1.
    """

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.gating_network = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self, z_img: torch.Tensor, z_aud: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_img: Tensor of shape (B, embed_dim)
            z_aud: Tensor of shape (B, embed_dim)

        Returns:
            Tuple of:
                - gated_img: (B, embed_dim)
                - gated_aud: (B, embed_dim)
                - trust_scores: (B, 2) where column 0 is img trust and column 1 is aud trust.
        """
        combined = torch.cat([z_img, z_aud], dim=-1)  # (B, 256)
        logits = self.gating_network(combined)        # (B, 2)
        trust_scores = F.softmax(logits, dim=-1)      # (B, 2)

        alpha_img = trust_scores[:, 0:1]              # (B, 1)
        alpha_aud = trust_scores[:, 1:2]              # (B, 1)

        # Scale by 2.0 so that when modalities are equally trusted (0.5, 0.5), representation magnitude is preserved
        gated_img = 2.0 * alpha_img * z_img
        gated_aud = 2.0 * alpha_aud * z_aud

        return gated_img, gated_aud, trust_scores

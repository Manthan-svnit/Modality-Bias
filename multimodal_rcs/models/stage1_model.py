from typing import Dict, Literal
import torch
import torch.nn as nn
from .image_encoder import ImageEncoder
from .audio_encoder import AudioEncoder
from .trust_gating import ModalityTrustGating
from .fusion import EarlyFusionModule, LateFusionModule
from .decoders import ImageDecoder, AudioDecoder


class Stage1Model(nn.Module):
    """Stage 1 Multimodal Architecture integrating Encoders, Modality Trust Gating,
    Cross-Attention Fusion (Early or Late), Decoders, and Auxiliary Classification Head.
    """

    def __init__(
        self,
        fusion_variant: Literal["early", "late"] = "early",
        embed_dim: int = 128,
        num_classes: int = 10,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fusion_variant = fusion_variant
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Modality Encoders
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.audio_encoder = AudioEncoder(embed_dim=embed_dim)

        # Modality Trust Gating
        self.trust_gating = ModalityTrustGating(embed_dim=embed_dim)

        # Fusion Module
        if fusion_variant.lower() == "early":
            self.fusion_module = EarlyFusionModule(
                dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif fusion_variant.lower() == "late":
            self.fusion_module = LateFusionModule(
                dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown fusion variant: {fusion_variant}")

        # Reconstruction Decoders
        self.image_decoder = ImageDecoder(embed_dim=embed_dim)
        self.audio_decoder = AudioDecoder(embed_dim=embed_dim)

        # Auxiliary Classification Head for Stage 1 Multi-Task Learning
        self.classifier_head = nn.Linear(embed_dim, num_classes)

    def forward(
        self, images: torch.Tensor, audios: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (B, 1, 28, 28)
            audios: (B, 1, 112, 112)
        Returns:
            Dictionary containing latent representations, trust scores, logits, and reconstructions.
        """
        # 1. Modality Encoders
        z_img = self.image_encoder(images)  # (B, 128)
        z_aud = self.audio_encoder(audios)  # (B, 128)

        # 2. Modality Trust Gating
        gated_img, gated_aud, trust_scores = self.trust_gating(z_img, z_aud)

        # 3. Cross-Attention Fusion
        z_fused = self.fusion_module(gated_img, gated_aud)  # (B, 128)

        # 4. Multi-Task Heads
        logits = self.classifier_head(z_fused)              # (B, 10)
        recon_img = self.image_decoder(z_fused)             # (B, 1, 28, 28)
        recon_aud = self.audio_decoder(z_fused)             # (B, 1, 112, 112)

        return {
            "z_img": z_img,
            "z_aud": z_aud,
            "gated_img": gated_img,
            "gated_aud": gated_aud,
            "trust_scores": trust_scores,
            "z_fused": z_fused,
            "logits": logits,
            "recon_img": recon_img,
            "recon_aud": recon_aud,
        }

    @torch.no_grad()
    def extract_fused(
        self, images: torch.Tensor, audios: torch.Tensor
    ) -> torch.Tensor:
        """Extracts frozen fused representations z_fused."""
        z_img = self.image_encoder(images)
        z_aud = self.audio_encoder(audios)
        gated_img, gated_aud, _ = self.trust_gating(z_img, z_aud)
        z_fused = self.fusion_module(gated_img, gated_aud)
        return z_fused

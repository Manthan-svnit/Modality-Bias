"""
Stage1Model: Generalized Multimodal Architecture

Supports any subset of 3 modalities: {image, audio, text}.
Only the modalities listed in `active_modalities` are constructed and used.

Forward pass:
  - Accepts keyword arguments only for the active modalities.
  - Missing modalities are silently ignored.
  - Dynamically routes through encoders -> trust gating -> fusion.

Output dictionary always contains:
  - "z_fused"      : (B, embed_dim) -- final fused representation
  - "logits"       : (B, num_classes) -- classification head output
  - "trust_scores" : (B, N)           -- trust weights for active modalities
  - Per-modality keys: "z_image", "z_audio", "z_text" (only for active ones)
  - Reconstruction keys: "recon_image", "recon_audio", "recon_text" (only for active ones)
  - "gated_{mod}"  : gated embedding for each active modality

Image input:   FloatTensor (B, C, H, W)  -- accepts any resolution via AdaptiveAvgPool
Audio input:   FloatTensor (B, C, H, W)  -- any spectrogram size
Text input:    LongTensor  (B, seq_len)  -- token ids with optional padding_idx=0

All encoders share the same embed_dim, enabling direct cross-modal attention.
"""

from __future__ import annotations
from typing import Dict, List, Literal, Optional, Tuple
import torch
import torch.nn as nn

from .image_encoder import ImageEncoder
from .audio_encoder import AudioEncoder
from .text_encoder import TextEncoder
from .text_decoder import TextDecoder
from .trust_gating import ModalityTrustGating
from .fusion import EarlyFusionModule, LateFusionModule
from .decoders import ImageDecoder, AudioDecoder

# Supported modality names (canonical keys)
VALID_MODALITIES = {"image", "audio", "text"}


class Stage1Model(nn.Module):
    """Generalized multimodal Stage 1 model.

    Constructed with a list of active modalities. At forward time, only the
    tensors corresponding to active modalities need to be provided.
    """

    def __init__(
        self,
        active_modalities: List[str],
        fusion_variant: Literal["early", "late"] = "early",
        embed_dim: int = 128,
        num_classes: int = 10,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        # Image encoder config
        image_in_channels: int = 1,
        # Audio encoder config
        audio_in_channels: int = 1,
        # Text encoder config
        text_vocab_size: int = 30000,
        text_max_seq_len: int = 128,
        text_ff_dim: int = 256,
        text_padding_idx: int = 0,
    ):
        """
        Args:
            active_modalities   : Subset of ["image", "audio", "text"] to use.
                                  Must have at least 1 element.
            fusion_variant      : "early" or "late" fusion.
            embed_dim           : Shared embedding dimensionality across all modalities.
            num_classes         : Number of output classes for the classification head.
            num_heads           : Number of attention heads in cross-attention.
            num_layers          : Number of cross-attention layers.
            dropout             : Dropout probability.
            image_in_channels   : Input channels for images (1=grayscale, 3=RGB).
            audio_in_channels   : Input channels for audio spectrograms.
            text_vocab_size     : Vocabulary size for the text tokenizer.
            text_max_seq_len    : Maximum token sequence length.
            text_ff_dim         : Feed-forward dim inside the Transformer text encoder.
            text_padding_idx    : Padding token id (ignored in text attention mask).
        """
        super().__init__()

        # Validate and store active modalities (preserve order for reproducibility)
        for m in active_modalities:
            assert m in VALID_MODALITIES, (
                f"Unknown modality '{m}'. Valid: {VALID_MODALITIES}"
            )
        assert len(active_modalities) >= 1, "Must specify at least 1 active modality."
        # Deduplicate while preserving order
        seen = set()
        self.active_modalities: List[str] = [
            m for m in active_modalities if not (m in seen or seen.add(m))
        ]
        self.num_modalities = len(self.active_modalities)
        self.fusion_variant = fusion_variant
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # ------------------------------------------------------------------ #
        # Encoders (only for active modalities)
        # ------------------------------------------------------------------ #
        if "image" in self.active_modalities:
            self.image_encoder = ImageEncoder(
                embed_dim=embed_dim, in_channels=image_in_channels
            )
        if "audio" in self.active_modalities:
            self.audio_encoder = AudioEncoder(
                embed_dim=embed_dim, in_channels=audio_in_channels
            )
        if "text" in self.active_modalities:
            self.text_encoder = TextEncoder(
                vocab_size=text_vocab_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                ff_dim=text_ff_dim,
                dropout=dropout,
                max_seq_len=text_max_seq_len,
                padding_idx=text_padding_idx,
            )

        # ------------------------------------------------------------------ #
        # Modality Trust Gating (N-way)
        # ------------------------------------------------------------------ #
        self.trust_gating = ModalityTrustGating(
            num_modalities=self.num_modalities,
            embed_dim=embed_dim,
            hidden_dim=max(64, embed_dim // 2),
        )

        # ------------------------------------------------------------------ #
        # Fusion Module
        # ------------------------------------------------------------------ #
        FusionClass = (
            EarlyFusionModule if fusion_variant.lower() == "early" else LateFusionModule
        )
        self.fusion_module = FusionClass(
            num_modalities=self.num_modalities,
            dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        # ------------------------------------------------------------------ #
        # Decoders (only for active modalities)
        # ------------------------------------------------------------------ #
        if "image" in self.active_modalities:
            self.image_decoder = ImageDecoder(embed_dim=embed_dim)
        if "audio" in self.active_modalities:
            self.audio_decoder = AudioDecoder(embed_dim=embed_dim)
        if "text" in self.active_modalities:
            self.text_decoder = TextDecoder(embed_dim=embed_dim, ff_dim=text_ff_dim, dropout=dropout)

        # ------------------------------------------------------------------ #
        # Classification Head
        # ------------------------------------------------------------------ #
        self.classifier_head = nn.Linear(embed_dim, num_classes)

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            image            : FloatTensor (B, C, H, W) -- required if "image" is active.
            audio            : FloatTensor (B, C, H, W) -- required if "audio" is active.
            text             : LongTensor  (B, seq_len) -- required if "text" is active.
            text_padding_mask: Optional BoolTensor (B, seq_len) -- True = padding.
        Returns:
            Dictionary with keys:
                "z_fused", "logits", "trust_scores",
                "z_image"/"z_audio"/"z_text"     (for active modalities),
                "gated_image"/"gated_audio"/"gated_text" (for active modalities),
                "recon_image"/"recon_audio"/"recon_text" (for active modalities)
        """
        # 1. Encode active modalities
        raw_embeddings: List[torch.Tensor] = []
        mod_keys: List[str] = []

        for mod in self.active_modalities:
            if mod == "image":
                assert image is not None, "image tensor required for 'image' modality"
                z = self.image_encoder(image)
            elif mod == "audio":
                assert audio is not None, "audio tensor required for 'audio' modality"
                z = self.audio_encoder(audio)
            elif mod == "text":
                assert text is not None, "text tensor required for 'text' modality"
                z = self.text_encoder(text, src_key_padding_mask=text_padding_mask)
            raw_embeddings.append(z)
            mod_keys.append(mod)

        # 2. Trust Gating
        gated_embeddings, trust_scores = self.trust_gating(raw_embeddings)

        # 3. Cross-Attention Fusion
        z_fused = self.fusion_module(gated_embeddings)

        # 4. Classification Head
        logits = self.classifier_head(z_fused)

        # 5. Reconstruction Decoders
        outputs: Dict[str, torch.Tensor] = {
            "z_fused": z_fused,
            "logits": logits,
            "trust_scores": trust_scores,
        }

        for i, mod in enumerate(mod_keys):
            outputs[f"z_{mod}"] = raw_embeddings[i]
            outputs[f"gated_{mod}"] = gated_embeddings[i]

            if mod == "image":
                outputs["recon_image"] = self.image_decoder(z_fused)
            elif mod == "audio":
                outputs["recon_audio"] = self.audio_decoder(z_fused)
            elif mod == "text":
                # Target for text reconstruction = mean token embedding of input
                # Computed here as the stop-gradient mean embedding (no teacher-forcing needed)
                with torch.no_grad():
                    mean_tok_emb = self.text_encoder.token_embedding(text).mean(dim=1)  # (B, D)
                outputs["recon_text"] = self.text_decoder(z_fused)
                outputs["text_mean_emb"] = mean_tok_emb  # reconstruction target

        return outputs

    # ---------------------------------------------------------------------- #
    # Feature Extraction (frozen, no_grad)
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def extract_fused(
        self,
        image: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extracts frozen fused representations z_fused for Stage 2.

        Returns:
            z_fused: (B, embed_dim)
        """
        outputs = self.forward(
            image=image, audio=audio,
            text=text, text_padding_mask=text_padding_mask
        )
        return outputs["z_fused"]

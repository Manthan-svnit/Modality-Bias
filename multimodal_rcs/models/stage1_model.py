"""
Stage1Model: Generalized Multimodal Architecture supporting dynamic N-modality inputs.

Supports any subset of modalities: {image, audio, text, video}.
Only the modalities listed in `active_modalities` are constructed and used.

Forward pass:
  - Accepts keyword arguments for any active modality (e.g. image=..., video=..., audio=..., text=...)
    or a dictionary/positional arguments.
  - Dynamically routes through encoders -> trust gating -> fusion.
  - Reconstructs inputs using modality-specific decoders.

Output dictionary contains:
  - "z_fused"      : (B, embed_dim) -- final fused representation
  - "logits"       : (B, num_classes) -- classification head output
  - "trust_scores" : (B, N)           -- trust weights for active modalities
  - Per-modality keys: "z_image", "z_audio", "z_text", "z_video" (only for active ones)
  - Reconstruction keys: "recon_image", "recon_audio", "recon_text", "recon_video" (for active ones)
  - Legacy convenience aliases: "z_img", "z_aud", "recon_img", "recon_aud"
  - "gated_{mod}"  : gated embedding for each active modality

All encoders project to a uniform embed_dim (e.g. d=128), enabling direct cross-modal attention.
"""

from __future__ import annotations
from typing import Dict, List, Literal, Optional, Tuple, Any
import torch
import torch.nn as nn

from .image_encoder import ImageEncoder
from .audio_encoder import AudioEncoder
from .text_encoder import TextEncoder
from .text_decoder import TextDecoder
from .video_encoder import VideoEncoder
from .video_decoder import VideoDecoder
from .trust_gating import ModalityTrustGating
from .fusion import EarlyFusionModule, LateFusionModule
from .decoders import ImageDecoder, AudioDecoder

# Supported modality names (canonical keys)
VALID_MODALITIES = {"image", "audio", "text", "video"}


class Stage1Model(nn.Module):
    """Generalized multimodal Stage 1 model.

    Constructed with a list of active modalities. Dynamically supports any combination
    (e.g., Image + Audio, Image + Text, Video + Audio, Image + Video + Audio, etc.).
    """

    def __init__(
        self,
        active_modalities: Optional[List[str]] = None,
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
        # Video encoder / decoder config
        video_in_channels: int = 3,
        video_recon_frames: int = 8,
        video_recon_height: int = 28,
        video_recon_width: int = 28,
    ):
        """
        Args:
            active_modalities   : Subset of ["image", "audio", "text", "video"] to use.
                                  Defaults to ["image", "audio"] for backward compatibility.
            fusion_variant      : "early" or "late" fusion.
            embed_dim           : Shared embedding dimensionality across all modalities.
            num_classes         : Number of output classes for the classification head.
            num_heads           : Number of attention heads in cross-attention.
            num_layers          : Number of cross-attention layers.
            dropout             : Dropout probability.
            image_in_channels   : Input channels for images (1=grayscale, 3=RGB).
            audio_in_channels   : Input channels for audio spectrograms.
            text_vocab_size     : Vocabulary size for text tokenizer.
            text_max_seq_len    : Maximum token sequence length.
            text_ff_dim         : Feed-forward dim inside Transformer text encoder.
            text_padding_idx    : Padding token id.
            video_in_channels   : Input channels for video (3=RGB, 1=grayscale).
            video_recon_frames  : Reconstructed video frames.
            video_recon_height  : Reconstructed video height.
            video_recon_width   : Reconstructed video width.
        """
        super().__init__()

        if active_modalities is None:
            active_modalities = ["image", "audio"]

        # Validate and store active modalities (preserve order for reproducibility)
        for m in active_modalities:
            assert m in VALID_MODALITIES, (
                f"Unknown modality '{m}'. Valid: {VALID_MODALITIES}"
            )
        assert len(active_modalities) >= 1, "Must specify at least 1 active modality."

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
        if "video" in self.active_modalities:
            self.video_encoder = VideoEncoder(
                embed_dim=embed_dim, in_channels=video_in_channels
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
            self.text_decoder = TextDecoder(
                embed_dim=embed_dim, ff_dim=text_ff_dim, dropout=dropout
            )
        if "video" in self.active_modalities:
            self.video_decoder = VideoDecoder(
                embed_dim=embed_dim,
                out_channels=video_in_channels,
                out_frames=video_recon_frames,
                out_height=video_recon_height,
                out_width=video_recon_width,
            )

        # ------------------------------------------------------------------ #
        # Classification Head
        # ------------------------------------------------------------------ #
        self.classifier_head = nn.Linear(embed_dim, num_classes)

    def _normalize_inputs(
        self, *args, **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Normalizes positional args or kwargs into a clean dictionary of modality tensors."""
        inputs: Dict[str, torch.Tensor] = {}

        # 1. Positional arguments map to active modalities in order
        if len(args) > 0:
            for i, arg in enumerate(args):
                if i < len(self.active_modalities) and isinstance(arg, torch.Tensor):
                    inputs[self.active_modalities[i]] = arg

        # 2. Kwargs mapping with plural and short aliases
        alias_map = {
            "image": "image", "images": "image", "img": "image",
            "audio": "audio", "audios": "audio", "aud": "audio",
            "text": "text", "texts": "text", "txt": "text",
            "video": "video", "videos": "video", "vid": "video",
        }
        for k, v in kwargs.items():
            if k == "text_padding_mask":
                continue
            canonical = alias_map.get(k, k)
            if v is not None and isinstance(v, torch.Tensor):
                inputs[canonical] = v

        text_padding_mask = kwargs.get("text_padding_mask", None)
        return inputs, text_padding_mask

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Accepts dynamic inputs for any active modalities:
            model(image=x_img, video=x_vid)
            model(images, audios)
            model(**batch_dict)

        Returns:
            Dictionary with keys:
                "z_fused", "logits", "trust_scores",
                "z_{mod}", "gated_{mod}", "recon_{mod}" for each active modality,
                plus backwards-compatible aliases ("z_img", "z_aud", "recon_img", "recon_aud").
        """
        inputs, text_padding_mask = self._normalize_inputs(*args, **kwargs)

        # 1. Encode active modalities
        raw_embeddings: List[torch.Tensor] = []
        mod_keys: List[str] = []

        for mod in self.active_modalities:
            assert mod in inputs, f"Missing input tensor for active modality '{mod}'"
            x = inputs[mod]

            if mod == "image":
                z = self.image_encoder(x)
            elif mod == "audio":
                z = self.audio_encoder(x)
            elif mod == "text":
                z = self.text_encoder(x, src_key_padding_mask=text_padding_mask)
            elif mod == "video":
                z = self.video_encoder(x)
            else:
                raise ValueError(f"Unsupported modality: {mod}")

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
                outputs["recon_img"] = outputs["recon_image"]
                outputs["z_img"] = raw_embeddings[i]
            elif mod == "audio":
                outputs["recon_audio"] = self.audio_decoder(z_fused)
                outputs["recon_aud"] = outputs["recon_audio"]
                outputs["z_aud"] = raw_embeddings[i]
            elif mod == "text":
                with torch.no_grad():
                    mean_tok_emb = self.text_encoder.token_embedding(inputs["text"]).mean(dim=1)
                outputs["recon_text"] = self.text_decoder(z_fused)
                outputs["text_mean_emb"] = mean_tok_emb
                outputs["z_txt"] = raw_embeddings[i]
            elif mod == "video":
                # If target video shape is available, pass (T, H, W) to match exactly
                target_shape = inputs["video"].shape[2:] if "video" in inputs else None
                outputs["recon_video"] = self.video_decoder(z_fused, target_shape=target_shape)
                outputs["z_vid"] = raw_embeddings[i]

        return outputs

    # ---------------------------------------------------------------------- #
    # Feature Extraction (frozen, no_grad)
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def extract_fused(self, *args, **kwargs) -> torch.Tensor:
        """Extracts frozen fused representations z_fused for Stage 2.

        Returns:
            z_fused: (B, embed_dim)
        """
        outputs = self.forward(*args, **kwargs)
        return outputs["z_fused"]

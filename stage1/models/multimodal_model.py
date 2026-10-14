"""
multimodal_model.py — Top-level Stage 1 model.

Wires together:
  Encoders → Gating → Fusion → {Projection heads, Decoders, Classifier}

forward() returns a structured dict with all intermediate representations
needed for loss computation and evaluation.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from stage1.encoders import build_encoder
from stage1.models.gating import GatingNetwork
from stage1.models.fusion import FusionModule
from stage1.models.projection import ProjectionHead, ModalityDecoder, ClassifierHead


class MultimodalModel(nn.Module):
    """
    Parameters
    ----------
    modalities : list of str
    cfg : ConfigNode
        Full config object.
    num_classes : int
    """

    def __init__(self, modalities: List[str], cfg, num_classes: int):
        super().__init__()
        self.modalities = modalities
        self.cfg = cfg
        self.num_classes = num_classes
        d = cfg.encoder.d

        # ── Encoders ──────────────────────────────────────────────────────────
        self.encoders = nn.ModuleDict({
            mod: build_encoder(mod, cfg.encoder, num_classes)
            for mod in modalities
        })

        # ── Gating ────────────────────────────────────────────────────────────
        self.gating = GatingNetwork(
            num_modalities=len(modalities),
            embed_dim=d,
            hidden_dim=cfg.gating.hidden_dim,
            dropout=cfg.gating.dropout,
        )

        # ── Fusion ────────────────────────────────────────────────────────────
        self.fusion = FusionModule(
            modalities=modalities,
            d=d,
            fusion_cfg=cfg.fusion,
            timing=cfg.fusion_timing,
        )

        # ── Projection heads ──────────────────────────────────────────────────
        proj_dim = cfg.projection.proj_dim
        dropout_proj = cfg.fusion.dropout

        # Fused projection head
        self.proj_fused = ProjectionHead(d, proj_dim, dropout_proj)

        # Per-modality projection heads
        self.proj_modality = nn.ModuleDict({
            mod: ProjectionHead(d, proj_dim, dropout_proj)
            for mod in modalities
        })

        # ── Decoders (reconstruction) ─────────────────────────────────────────
        dec_cfg = cfg.decoder
        self.decoders = nn.ModuleDict({
            mod: ModalityDecoder(
                in_dim=d,
                hidden_dim=dec_cfg.hidden_dim,
                out_dim=d,
                num_layers=dec_cfg.num_layers,
                dropout=dropout_proj,
            )
            for mod in modalities
        })

        # ── Classifier head ───────────────────────────────────────────────────
        self.classifier = ClassifierHead(d, num_classes, dropout_proj)

    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        batch : dict
            Must contain keys for each active modality (modality_name → tensor)
            plus ``"label"`` → [B].

        Returns
        -------
        out : dict with keys:
          ``z``        : dict modality → [B, d]   raw embeddings
          ``z_gated``  : dict modality → [B, d]   gated embeddings
          ``gates``    : [B, M]  softmax trust scores
          ``z_fused``  : [B, d]  fused representation
          ``p_fused``  : [B, proj_dim]  fused projection
          ``p_mod``    : dict modality → [B, proj_dim]  per-modality projections
          ``z_rec``    : dict modality → [B, d]  reconstructed modality embeddings
          ``logits``   : [B, num_classes]  classification logits
        """
        # ── Encode ────────────────────────────────────────────────────────────
        z: Dict[str, torch.Tensor] = {}
        for mod in self.modalities:
            z[mod] = self.encoders[mod](batch[mod])   # [B, d]

        # ── Gate ──────────────────────────────────────────────────────────────
        z_gated, gates = self.gating(z)              # dict, [B, M]

        # ── Fuse ──────────────────────────────────────────────────────────────
        z_fused = self.fusion(z_gated)               # [B, d]

        # ── Project ───────────────────────────────────────────────────────────
        p_fused = self.proj_fused(z_fused)           # [B, proj_dim]
        p_mod = {
            mod: self.proj_modality[mod](z[mod])
            for mod in self.modalities
        }

        # ── Reconstruct ───────────────────────────────────────────────────────
        z_rec = {
            mod: self.decoders[mod](z_fused)
            for mod in self.modalities
        }

        # ── Classify ──────────────────────────────────────────────────────────
        logits = self.classifier(z_fused)            # [B, num_classes]

        return {
            "z": z,
            "z_gated": z_gated,
            "gates": gates,
            "z_fused": z_fused,
            "p_fused": p_fused,
            "p_mod": p_mod,
            "z_rec": z_rec,
            "logits": logits,
        }

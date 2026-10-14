"""
fusion.py — Cross-attention fusion module (early & late).

Early fusion  : cross-attend the raw encoder outputs z_m.
Late fusion   : apply per-modality MLP → then cross-attend.

Output: z_fused ∈ ℝ^[B, d]  (the fused representation saved at end of Stage 1)
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Cross-attention block
# ──────────────────────────────────────────────────────────────────────────────

class CrossAttentionLayer(nn.Module):
    """
    Single cross-attention layer: query from one modality, key/value from
    all other modalities (including itself → self-attend).

    Uses PyTorch's ``nn.MultiheadAttention`` with batch-first=True.
    """

    def __init__(self, d: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d),
        )
        self.ln2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        query : [B, 1, d]
        kv    : [B, M, d]   (all modality tokens, including query)
        returns: [B, 1, d]
        """
        attended, _ = self.attn(query, kv, kv)
        query = self.ln1(query + self.drop(attended))
        query = self.ln2(query + self.drop(self.ffn(query)))
        return query


# ──────────────────────────────────────────────────────────────────────────────
# Fusion module
# ──────────────────────────────────────────────────────────────────────────────

class FusionModule(nn.Module):
    """
    Multi-layer cross-attention fusion.

    All modality embeddings form a token sequence [B, M, d].
    Each modality attends over all others through ``num_layers`` layers.
    The final fused vector is the mean of all modality tokens.

    For late fusion, a per-modality MLP is applied before attention.

    Parameters
    ----------
    modalities : list of str
        Active modality names (determines M).
    d : int
        Embedding dimension.
    fusion_cfg : ConfigNode
        ``cfg.fusion`` sub-config.
    timing : str
        ``"early"`` or ``"late"``.
    """

    def __init__(
        self,
        modalities: List[str],
        d: int,
        fusion_cfg,
        timing: str = "early",
    ):
        super().__init__()
        self.M = len(modalities)
        self.d = d
        self.timing = timing
        num_heads = fusion_cfg.num_heads
        num_layers = fusion_cfg.num_layers
        dropout = fusion_cfg.dropout
        ffn_dim = fusion_cfg.ffn_dim

        # Late fusion: per-modality MLP before cross-attention
        if timing == "late":
            extra_layers = getattr(fusion_cfg, "late_extra_layers", 2)
            extra_dim = getattr(fusion_cfg, "late_extra_dim", 256)
            self.pre_mlps = nn.ModuleList([
                self._make_mlp(d, extra_dim, d, extra_layers, dropout)
                for _ in modalities
            ])

        # Cross-attention layers
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final projection
        self.out_proj = nn.Linear(d, d)
        self.out_ln = nn.LayerNorm(d)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_mlp(
        in_dim: int, hidden: int, out_dim: int, depth: int, dropout: float
    ) -> nn.Sequential:
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, out_dim))
        return nn.Sequential(*layers)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : dict   modality → [B, d]
            (weighted by gating, or raw encoder outputs)

        Returns
        -------
        z_fused : [B, d]   (L2-normalised fused embedding)
        """
        tokens_list = list(embeddings.values())     # list of [B, d]

        # Late fusion: run per-modality MLPs
        if self.timing == "late":
            tokens_list = [
                mlp(t) for mlp, t in zip(self.pre_mlps, tokens_list)
            ]

        # Stack to [B, M, d]
        tokens = torch.stack(tokens_list, dim=1)    # [B, M, d]

        # Multi-layer cross-attention
        for layer in self.layers:
            new_tokens = []
            for i in range(self.M):
                q = tokens[:, i : i + 1, :]         # [B, 1, d]
                updated = layer(q, tokens)           # [B, 1, d]
                new_tokens.append(updated)
            tokens = torch.cat(new_tokens, dim=1)   # [B, M, d]

        # Mean pool over modalities → [B, d]
        z = tokens.mean(dim=1)
        z = self.out_ln(self.out_proj(z))
        return F.normalize(z, dim=-1)

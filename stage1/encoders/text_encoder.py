"""
text_encoder.py — Text encoder supporting HuggingFace transformers and
                  static embeddings (GloVe / FastText).

Input : dict with ``input_ids`` and ``attention_mask`` [B, seq_len]
        OR plain torch.Tensor [B, embed_dim] (pre-computed static embeddings)
Output: [B, d]   (L2-normalised embedding)

Modes (controlled by ``encoder_cfg.text.model_name``):
  - Any HuggingFace model name → transformer backbone (CLS token)
  - ``"glove"`` or ``"fasttext"`` → two-layer MLP over mean-pooled static vectors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from stage1.encoders.registry import register_encoder

try:
    from transformers import AutoModel
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ──────────────────────────────────────────────────────────────────────────────

class _TransformerTextEncoder(nn.Module):
    """
    HuggingFace transformer backbone → CLS-token → linear projection.

    Parameters
    ----------
    model_name : str
        HuggingFace model name (e.g., ``bert-base-uncased``).
    d : int
        Output embedding dimension.
    dropout : float
    freeze_backbone : bool
        If True, all transformer weights are frozen.
    """

    def __init__(self, model_name: str, d: int, dropout: float, freeze_backbone: bool):
        super().__init__()
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required for transformer text encoder.")
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(hidden_size, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: dict) -> torch.Tensor:
        out = self.backbone(
            input_ids=x["input_ids"],
            attention_mask=x["attention_mask"],
        )
        # CLS token representation
        cls = out.last_hidden_state[:, 0, :]   # [B, hidden_size]
        return self.proj(cls)                  # [B, d]


class _StaticTextEncoder(nn.Module):
    """
    Two-layer MLP over mean-pooled static (GloVe/FastText) embeddings.

    Expects ``x`` as a tensor [B, embed_dim].
    ``embed_dim`` is inferred on the first forward pass.
    """

    def __init__(self, d: int, dropout: float, embed_dim: int = 300):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, d * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d * 2, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)   # [B, d]


# ──────────────────────────────────────────────────────────────────────────────
# Registered encoder
# ──────────────────────────────────────────────────────────────────────────────

@register_encoder("text")
class TextEncoder(nn.Module):
    """
    Unified text encoder.  Dispatches to transformer or static sub-encoder
    based on ``encoder_cfg.text.model_name``.

    Parameters from ``encoder_cfg``:
      - ``d``                       : output embedding dimension
      - ``text.model_name``         : HuggingFace model name OR "glove"/"fasttext"
      - ``text.max_length``         : tokenizer max length (informational)
      - ``text.dropout``            : dropout rate
      - ``text.freeze_backbone``    : freeze transformer weights
    """

    _STATIC_NAMES = {"glove", "fasttext"}

    def __init__(self, encoder_cfg):
        super().__init__()
        d = encoder_cfg.d
        model_name: str = encoder_cfg.text.model_name
        dropout: float = encoder_cfg.text.dropout
        freeze: bool = getattr(encoder_cfg.text, "freeze_backbone", False)

        self.is_static = model_name.lower() in self._STATIC_NAMES

        if self.is_static:
            # Default GloVe/FastText dim is 300; can be overridden
            embed_dim = getattr(encoder_cfg.text, "embed_dim", 300)
            self._encoder = _StaticTextEncoder(d, dropout, embed_dim)
        else:
            self._encoder = _TransformerTextEncoder(model_name, d, dropout, freeze)

        self.d = d

    def forward(self, x) -> torch.Tensor:
        """
        Parameters
        ----------
        x : dict (input_ids, attention_mask) for transformers
            OR torch.Tensor [B, embed_dim] for static embeddings

        Returns
        -------
        z : [B, d]   (L2-normalised)
        """
        z = self._encoder(x)
        return F.normalize(z, dim=-1)

"""
Text Encoder: Transformer-based encoder mapping variable-length token sequences
to a fixed embed_dim-dimensional embedding.

Supports:
  - Arbitrary vocabulary size and sequence length (passed at construction time)
  - Token-level positional encoding
  - Learned [CLS] token for sequence-level representation
  - Padding mask for variable-length inputs

Input contract:
    x         : LongTensor of shape (B, seq_len)   -- token ids
    mask      : BoolTensor of shape (B, seq_len)   -- True where token is padding (optional)

Output:
    embedding : FloatTensor of shape (B, embed_dim)
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, embed_dim)
        Returns:
            x + positional_encoding: (B, seq_len, embed_dim)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TextEncoder(nn.Module):
    """Transformer-based text encoder producing a fixed-size sentence embedding.

    Workflow:
      1. Token embedding lookup  ->  (B, seq_len, embed_dim)
      2. Prepend learned [CLS] token
      3. Add sinusoidal positional encoding
      4. Pass through N Transformer encoder layers
      5. Extract CLS token output as sentence embedding  ->  (B, embed_dim)
      6. Project through LayerNorm

    The [CLS] token follows the BERT convention: it accumulates global context
    through self-attention over the whole sequence.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        padding_idx: int = 0,
    ):
        """
        Args:
            vocab_size  : Size of the token vocabulary (including special tokens).
            embed_dim   : Dimensionality of the output embedding (must match pipeline embed_dim).
            num_heads   : Number of multi-head attention heads.
            num_layers  : Number of Transformer encoder layers.
            ff_dim      : Feed-forward inner dimension inside each Transformer layer.
            dropout     : Dropout probability.
            max_seq_len : Maximum supported sequence length for positional encoding.
            padding_idx : Token id used for padding (masked out in attention).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx

        # Token embedding
        self.token_embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )

        # Learned [CLS] token (prepended at position 0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional encoding (accounts for +1 CLS position)
        self.pos_enc = PositionalEncoding(
            embed_dim=embed_dim, max_len=max_seq_len + 1, dropout=dropout
        )

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        # Output projection / normalization
        self.output_norm = nn.LayerNorm(embed_dim)

    def _build_padding_mask(
        self, token_ids: torch.Tensor, extra_token_count: int = 1
    ) -> torch.Tensor:
        """Creates a key_padding_mask for TransformerEncoder.

        Returns a BoolTensor of shape (B, seq_len + extra_token_count) where
        True means *ignore* (padding position). The extra positions (CLS) are
        never masked.
        """
        B, L = token_ids.shape
        # True where token == padding_idx
        pad_mask = token_ids.eq(self.padding_idx)  # (B, L)
        # Prepend False (CLS is never masked)
        cls_mask = torch.zeros(
            B, extra_token_count, dtype=torch.bool, device=token_ids.device
        )
        return torch.cat([cls_mask, pad_mask], dim=1)  # (B, 1 + L)

    def forward(
        self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x                   : LongTensor (B, seq_len) -- token ids.
            src_key_padding_mask: Optional BoolTensor (B, seq_len), True = padding.
                                  If None, inferred automatically from padding_idx.
        Returns:
            embedding: FloatTensor (B, embed_dim)
        """
        B, L = x.shape

        # 1. Token embeddings
        tok_emb = self.token_embedding(x)  # (B, L, D)

        # 2. Prepend [CLS]
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, D)
        h = torch.cat([cls, tok_emb], dim=1)      # (B, 1+L, D)

        # 3. Positional encoding
        h = self.pos_enc(h)  # (B, 1+L, D)

        # 4. Build padding mask if not provided
        if src_key_padding_mask is None:
            src_key_padding_mask = self._build_padding_mask(x, extra_token_count=1)

        # 5. Transformer encoder
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)  # (B, 1+L, D)

        # 6. Extract [CLS] and project
        cls_out = h[:, 0, :]        # (B, D)
        embedding = self.output_norm(cls_out)
        return embedding

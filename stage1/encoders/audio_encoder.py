"""
audio_encoder.py — Spectrogram CNN encoder for audio.
RCS-aligned 2-Conv, 2-BatchNorm architecture with projection to d.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from stage1.encoders.registry import register_encoder


@register_encoder("audio")
class AudioEncoder(nn.Module):
    """
    RCS-aligned Audio Spectrogram Encoder: 2 Conv layers + 2 BatchNorm layers + Linear to d.
    """

    def __init__(self, encoder_cfg):
        super().__init__()
        d = encoder_cfg.d
        width = getattr(encoder_cfg.audio, "cnn_width", 64)
        dropout = getattr(encoder_cfg.audio, "dropout", 0.1)

        self.conv1 = nn.Conv2d(1, width, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width * 2, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(width * 2)
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(width * 2, d),
        )
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, n_mels, T] or [B, 1, n_mels, T]
        Returns z : [B, d] (L2-normalised)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.pool(out).flatten(1)
        z = self.fc(out)
        return F.normalize(z, dim=-1)

"""
registry.py — Encoder registry.

Encoders are registered by modality name and built via ``build_encoder``.
All encoders share the same interface:

    forward(x) -> torch.Tensor  shape [B, d]

where ``d`` is ``cfg.encoder.d`` (the shared embedding dimension).
"""

from typing import Callable, Dict

import torch
import torch.nn as nn

_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_encoder(modality: str):
    """Class decorator to register an encoder for ``modality``."""
    def decorator(cls):
        _REGISTRY[modality] = cls
        return cls
    return decorator


def build_encoder(modality: str, encoder_cfg, num_classes: int = 0) -> nn.Module:
    """
    Instantiate the encoder for *modality* using *encoder_cfg*.

    Parameters
    ----------
    modality : str
        One of ``{"image", "audio", "video", "text"}``.
    encoder_cfg : ConfigNode
        The ``cfg.encoder`` sub-config.
    num_classes : int
        Passed to text encoders that need it (unused by CNN encoders).

    Returns
    -------
    nn.Module  —  forward(x) -> [B, d]
    """
    if modality not in _REGISTRY:
        raise ValueError(
            f"No encoder registered for modality '{modality}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[modality](encoder_cfg)


def list_encoders():
    return list(_REGISTRY.keys())

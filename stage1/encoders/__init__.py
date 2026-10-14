"""
Encoders package.
Importing this package auto-registers all built-in encoders.
"""

# Auto-register encoders by importing them
from stage1.encoders import image_encoder  # noqa: F401
from stage1.encoders import audio_encoder  # noqa: F401
from stage1.encoders import video_encoder  # noqa: F401
from stage1.encoders import text_encoder   # noqa: F401

from stage1.encoders.registry import build_encoder, list_encoders, register_encoder

__all__ = ["build_encoder", "list_encoders", "register_encoder"]

"""
config.py — Loads, validates, and exposes the YAML configuration.
All other modules import `from stage1.config.config import cfg`.
"""

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (in-place). Returns base."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _resolve(path: str, root: Path) -> str:
    """Make relative paths absolute relative to *root*."""
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    return str(p)


# ──────────────────────────────────────────────────────────────────────────────
# Flat-access config object (dict with attribute access)
# ──────────────────────────────────────────────────────────────────────────────

class ConfigNode(dict):
    """dict whose keys are also accessible as attributes."""

    def __getattr__(self, name: str):
        try:
            v = self[name]
        except KeyError:
            raise AttributeError(f"Config has no key '{name}'") from None
        return v

    def __setattr__(self, name: str, value):
        self[name] = value

    def __delattr__(self, name: str):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None

    @classmethod
    def from_dict(cls, d: dict) -> "ConfigNode":
        node = cls()
        for k, v in d.items():
            if isinstance(v, dict):
                node[k] = cls.from_dict(v)
            else:
                node[k] = v
        return node

    def to_dict(self) -> dict:
        out = {}
        for k, v in self.items():
            out[k] = v.to_dict() if isinstance(v, ConfigNode) else v
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULTS_PATH = Path(__file__).parent / "config_schema.yaml"

# Global config singleton (populated by load_config)
cfg: Optional[ConfigNode] = None


def load_config(user_yaml: str, overrides: Optional[Dict[str, Any]] = None) -> ConfigNode:
    """
    Load *user_yaml*, merge over schema defaults, apply *overrides* dict,
    validate, and set the global ``cfg``.

    Parameters
    ----------
    user_yaml : str
        Path to user-supplied YAML file.
    overrides : dict, optional
        Key-value pairs (supports dot-notation keys, e.g. ``"training.lr"``)
        that override values in the YAML.

    Returns
    -------
    ConfigNode
        Validated configuration object.
    """
    global cfg

    # 1. Load defaults
    with open(_DEFAULTS_PATH) as f:
        raw = yaml.safe_load(f)

    # 2. Load user YAML and merge
    user_path = Path(user_yaml)
    with open(user_path) as f:
        user_raw = yaml.safe_load(f) or {}
    _deep_update(raw, user_raw)

    # 3. Apply CLI overrides (dot-notation)
    if overrides:
        for key_path, value in overrides.items():
            _set_dotpath(raw, key_path, value)

    # 4. Expand {dataset_name} template strings
    raw = _expand_templates(raw)

    # 5. Validate
    _validate(raw)

    # 6. Convert and cache
    cfg = ConfigNode.from_dict(raw)
    return cfg


def _set_dotpath(d: dict, key_path: str, value) -> None:
    """Set d[a][b][c] = value for key_path = 'a.b.c'."""
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _expand_templates(raw: dict) -> dict:
    """Replace {dataset_name} tokens recursively."""
    dataset_name = raw.get("dataset_name", "")

    def _walk(obj):
        if isinstance(obj, str):
            return obj.replace("{dataset_name}", dataset_name)
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(i) for i in obj]
        return obj

    return _walk(raw)


def _validate(raw: dict) -> None:
    """Minimal sanity checks on the config."""
    errors = []

    # Modalities
    valid_mods = {"image", "audio", "video", "text"}
    mods = set(raw.get("modalities", []))
    unknown = mods - valid_mods
    if unknown:
        errors.append(f"Unknown modalities: {unknown}. Valid: {valid_mods}")
    if not mods:
        errors.append("At least one modality must be specified.")

    # Fusion timing
    if raw.get("fusion_timing") not in ("early", "late"):
        errors.append("fusion_timing must be 'early' or 'late'.")

    # Contrastive placement
    # Contrastive placement is locked to "both" (per-modality AND fused simultaneously)
    if raw.get("contrastive_placement", "both") != "both":
        errors.append(
            "contrastive_placement must be 'both'. "
            "Computes supervised contrastive loss on each modality vector AND on z_fused."
        )

    # Embedding dimension vs. num_heads
    d = raw.get("encoder", {}).get("d", 128)
    num_heads = raw.get("fusion", {}).get("num_heads", 4)
    if d % num_heads != 0:
        errors.append(
            f"encoder.d ({d}) must be divisible by fusion.num_heads ({num_heads})."
        )

    # Batch size
    bs = raw.get("training", {}).get("batch_size", 64)
    if bs < 2:
        errors.append("training.batch_size must be >= 2.")

    if errors:
        raise ValueError("Config validation failed:\n  " + "\n  ".join(errors))


def save_config(cfg: Any, path: Any) -> None:
    """Save a ConfigNode or dict to a YAML file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def _to_plain(obj):
        if isinstance(obj, dict):
            return {k: _to_plain(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_plain(x) for x in obj]
        return obj

    data = _to_plain(cfg)
    with open(p, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

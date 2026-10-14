"""
manifest.py — Reads CSV or JSON manifest files.

Expected manifest columns (CSV) or keys (JSON):
  - sample_id          : unique identifier
  - label              : integer class label (0-indexed)
  - split              : "train" | "val" | "test"
  - image_path (opt)   : path to image file
  - audio_path (opt)   : path to audio file (wav/mp3/flac)
  - video_path (opt)   : path to video file (mp4/avi)
  - text       (opt)   : raw text string  — OR —
    text_path  (opt)   : path to a .txt file

Only samples that have a valid entry for EVERY enabled modality are kept.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

MODALITY_COLUMNS: Dict[str, List[str]] = {
    "image": ["image_path"],
    "audio": ["audio_path"],
    "video": ["video_path"],
    "text": ["text", "text_path"],  # either column is acceptable
}


def load_manifest(
    manifest_path: str,
    modalities: Sequence[str],
    split: Optional[str] = None,
    create_artificial_imbalance: bool = False,
    imbalance_ratio: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load the manifest file and return a filtered DataFrame.

    Parameters
    ----------
    manifest_path : str
        Path to a CSV or JSON file.
    modalities : sequence of str
        Active modalities (subset of ``{"image","audio","video","text"}``).
    split : str, optional
        If given, filter to rows where ``split`` column equals this value.
        Pass ``None`` to return all splits.

    Returns
    -------
    pd.DataFrame
        Rows with columns: sample_id, label, split, + modality path columns.
        Only rows that have ALL required modality columns non-null are kept.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    # ── Load ──
    if path.suffix.lower() == ".json":
        df = _load_json(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported manifest format: {path.suffix}. Use .csv or .json.")

    # ── Normalise required columns ──
    _assert_columns(df, ["sample_id", "label", "split"])
    df["label"] = df["label"].astype(int)
    df["sample_id"] = df["sample_id"].astype(str)

    # ── Split filter ──
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No rows found for split='{split}' in {path}.")

    # ── Filter to samples with all required modality columns ──
    required_cols = _required_columns(modalities, df)
    before = len(df)
    mask = df[required_cols].notna().all(axis=1)
    df = df[mask].reset_index(drop=True)
    after = len(df)
    if before != after:
        print(
            f"[manifest] Dropped {before - after} rows missing modality columns "
            f"({required_cols}). Kept {after}."
        )

    if len(df) == 0:
        raise ValueError(
            "Manifest is empty after filtering for required modalities. "
            "Check that your CSV has the correct column names."
        )

    # ── Artificial Imbalance (train split only) ──
    if split == "train" and create_artificial_imbalance:
        df = apply_artificial_imbalance(df, imbalance_ratio=imbalance_ratio, seed=seed)

    return df


def apply_artificial_imbalance(
    df: pd.DataFrame,
    imbalance_ratio: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Artificially downsample classes in a balanced train set to create step/exponential imbalance.
    As per spec: "create_artificial_imbalance: false # only if dataset is balanced; train split only
    Drop samples missing any enabled modality (log per-class counts before and after)."
    """
    rng = np.random.default_rng(seed)
    counts_before = df["label"].value_counts().sort_index()
    print("\n[manifest] Creating artificial imbalance (train split only)...")
    print(f"  Class counts BEFORE artificial imbalance:\n{counts_before.to_string()}")

    classes = sorted(df["label"].unique())
    num_classes = len(classes)
    if num_classes <= 1:
        return df

    # Exponential imbalance profile: mu_i = mu_max * (imbalance_ratio ** (i / (num_classes - 1)))
    kept_dfs = []
    for idx, c in enumerate(classes):
        c_df = df[df["label"] == c]
        n_c = len(c_df)
        decay = imbalance_ratio ** (idx / float(num_classes - 1))
        target_n = max(2, int(round(n_c * decay)))
        if target_n < n_c:
            chosen_indices = rng.choice(c_df.index, size=target_n, replace=False)
            kept_dfs.append(c_df.loc[chosen_indices])
        else:
            kept_dfs.append(c_df)

    imbalanced_df = pd.concat(kept_dfs).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    counts_after = imbalanced_df["label"].value_counts().sort_index()
    print(f"  Class counts AFTER artificial imbalance:\n{counts_after.to_string()}\n")
    return imbalanced_df


def get_num_classes(df: pd.DataFrame) -> int:
    """Return the number of unique classes in the manifest."""
    return int(df["label"].nunique())


def class_counts(df: pd.DataFrame) -> pd.Series:
    """Return a Series mapping label -> count, sorted by label."""
    return df["label"].value_counts().sort_index()


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> pd.DataFrame:
    """Load a JSON manifest.  Accepts both a list of dicts and a dict-of-lists."""
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return pd.DataFrame(raw)
    if isinstance(raw, dict):
        return pd.DataFrame(raw)
    raise ValueError("JSON manifest must be a list of records or a dict of columns.")


def _assert_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}.")


def _required_columns(modalities: Sequence[str], df: pd.DataFrame) -> List[str]:
    """
    Determine which columns must be non-null for each modality.
    For text, either 'text' OR 'text_path' is sufficient.
    """
    cols = []
    for mod in modalities:
        options = MODALITY_COLUMNS.get(mod, [])
        if not options:
            raise ValueError(f"Unknown modality '{mod}'.")
        # Check which column(s) exist in the dataframe
        present = [c for c in options if c in df.columns]
        if not present:
            raise ValueError(
                f"Manifest has no column for modality '{mod}'. "
                f"Expected one of: {options}."
            )
        # For text: if both exist, 'text' takes priority; otherwise use whichever is present
        if mod == "text":
            if "text" in present:
                cols.append("text")
            else:
                cols.append("text_path")
        else:
            cols.append(present[0])
    return cols

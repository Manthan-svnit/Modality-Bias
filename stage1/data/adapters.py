"""
adapters.py — Generic Dataset Adapters & Manifest Generators.

Allows loading any dataset (not tied to CREMA-D) into the standard manifest format:
  Columns: sample_id, label, split, <modality_columns...>

Provides utilities to:
  1. Build manifest from folder structures:
     dataset_dir/
       split/ (train/val/test)
         class_name/
           sample_001_audio.wav, sample_001_video.mp4, sample_001_image.png, sample_001.txt
  2. Build manifest from tabular data / existing metadata CSVs.
  3. Validate and summarize any manifest across arbitrary modalities and label schemes.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
import pandas as pd


def create_manifest_from_directory(
    root_dir: Union[str, Path],
    modality_extensions: Dict[str, Sequence[str]],
    splits: Sequence[str] = ("train", "val", "test"),
    output_csv: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Build a standard manifest DataFrame from a structured directory.

    Expected layout:
      root_dir/
        {split}/
          {class_name_or_id}/
            {sample_stem}.{ext}   (e.g., sample1.wav, sample1.mp4, sample1.jpg, sample1.txt)

    Parameters
    ----------
    root_dir : str | Path
        Root path of the dataset.
    modality_extensions : dict
        Mapping of modality name -> allowed extensions.
        e.g. {"audio": [".wav", ".mp3", ".flac"], "video": [".mp4", ".avi"], "image": [".jpg", ".png"], "text": [".txt"]}
    splits : sequence of str
        Splits to look for (default: train, val, test).
    output_csv : str | Path, optional
        If provided, saves the manifest to this CSV path.

    Returns
    -------
    pd.DataFrame with sample_id, label, split, and modality paths.
    """
    root_dir = Path(root_dir)
    records = []

    # Map class names to integer labels dynamically
    class_to_id: Dict[str, int] = {}

    for split in splits:
        split_dir = root_dir / split
        if not split_dir.exists():
            continue

        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        for cdir in class_dirs:
            cname = cdir.name
            if cname not in class_to_id:
                class_to_id[cname] = len(class_to_id)
            label = class_to_id[cname]

            # Group files by stem
            sample_groups: Dict[str, Dict[str, str]] = {}
            for f in cdir.iterdir():
                if not f.is_file():
                    continue
                ext = f.suffix.lower()
                stem = f.stem

                # Clean common modality suffixes in stem (e.g., sample01_audio -> sample01)
                for mod in modality_extensions.keys():
                    if stem.endswith(f"_{mod}"):
                        stem = stem[: -len(f"_{mod}")]

                for mod, exts in modality_extensions.items():
                    if ext in exts:
                        if stem not in sample_groups:
                            sample_groups[stem] = {}
                        col = "image_path" if mod == "image" else f"{mod}_path" if mod in ("audio", "video") else "text_path"
                        sample_groups[stem][col] = str(f.resolve())

            for stem, paths in sample_groups.items():
                record = {
                    "sample_id": f"{split}_{cname}_{stem}",
                    "label": label,
                    "class_name": cname,
                    "split": split,
                }
                record.update(paths)
                records.append(record)

    df = pd.DataFrame(records)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[adapters] Generated manifest with {len(df)} samples saved to: {output_csv}")

    return df


def create_manifest_from_dataframe(
    df: pd.DataFrame,
    sample_id_col: str,
    label_col: str,
    split_col: str,
    modality_col_map: Dict[str, str],
    output_csv: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Standardize an existing DataFrame into the standard multimodal manifest format.

    Parameters
    ----------
    df : pd.DataFrame
    sample_id_col : str
    label_col : str
    split_col : str
    modality_col_map : dict
        Mapping from standard manifest column name (e.g. 'audio_path', 'video_path', 'text')
        to column name in the source DataFrame.
    output_csv : str | Path, optional
    """
    out_df = pd.DataFrame()
    out_df["sample_id"] = df[sample_id_col].astype(str)

    # Encode label as integer if categorical
    if df[label_col].dtype == object or str(df[label_col].dtype) == "category":
        classes = sorted(df[label_col].unique())
        c2i = {c: i for i, c in enumerate(classes)}
        out_df["label"] = df[label_col].map(c2i).astype(int)
        out_df["class_name"] = df[label_col].astype(str)
    else:
        out_df["label"] = df[label_col].astype(int)

    out_df["split"] = df[split_col].astype(str)

    for std_col, src_col in modality_col_map.items():
        if src_col in df.columns:
            out_df[std_col] = df[src_col]

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_csv, index=False)
        print(f"[adapters] Standardized manifest with {len(out_df)} samples saved to: {output_csv}")

    return out_df

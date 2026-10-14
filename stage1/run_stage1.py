"""
run_stage1.py — Entry point for Stage 1.

Usage
-----
# Minimal (all settings from your YAML):
python run_stage1.py --config path/to/my_config.yaml

# With Optuna Stage A tuning first:
python run_stage1.py --config path/to/my_config.yaml --tune

# With CLI overrides (dot-notation):
python run_stage1.py --config path/to/my_config.yaml \\
    training.lr=5e-4 fusion_timing=late contrastive_placement=both

Pipeline
--------
1. Load & validate config
2. Load manifest (train / val / test splits)
3. Build DataLoaders (with class-balanced sampler if enabled)
4. [Optional] Run Optuna Stage A to find best hyperparameters
5. Build model
6. Train with Stage1Trainer
7. All artifacts written to results/{dataset_name}/...
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Multimodal RCS")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna Stage A hyperparameter search before final training.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Key=value overrides (dot-notation). e.g. training.lr=1e-3",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_overrides(override_list):
    """Convert ['a.b=1', 'c=foo'] → {'a.b': 1, 'c': 'foo'}."""
    result = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be in key=value format.")
        k, v = item.split("=", 1)
        # Try to cast to int/float/bool
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        if v == "true":
            v = True
        elif v == "false":
            v = False
        result[k] = v
    return result


def main():
    args = parse_args()

    # ── 1. Config ─────────────────────────────────────────────────────────────
    from stage1.config.config import load_config
    overrides = parse_overrides(args.overrides)
    cfg = load_config(args.config, overrides)

    set_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_stage1] Device: {device}")
    print(f"[run_stage1] Dataset: {cfg.dataset_name}")
    print(f"[run_stage1] Modalities: {cfg.modalities}")
    print(f"[run_stage1] Fusion: {cfg.fusion_timing} | Placement: {cfg.contrastive_placement}")

    # ── 2. Manifest ───────────────────────────────────────────────────────────
    from stage1.data.manifest import load_manifest, get_num_classes
    modalities = list(cfg.modalities)

    create_imb = getattr(cfg, "create_artificial_imbalance", False)
    train_df = load_manifest(
        cfg.manifest, modalities, split="train",
        create_artificial_imbalance=create_imb, seed=cfg.training.seed
    )
    val_df   = load_manifest(cfg.manifest, modalities, split="val")
    test_df  = load_manifest(cfg.manifest, modalities, split="test")

    num_classes = get_num_classes(train_df)
    print(f"[run_stage1] Classes: {num_classes}  |  "
          f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    # ── 3. Datasets & DataLoaders ─────────────────────────────────────────────
    from stage1.data.dataset import MultimodalDataset
    from stage1.data.sampler import build_sampler

    text_model = cfg.encoder.text.model_name if "text" in modalities else "bert-base-uncased"
    text_len   = cfg.encoder.text.max_length  if "text" in modalities else 128

    train_ds = MultimodalDataset(train_df, modalities, cfg.encoder, train=True,
                                 tokenizer_name=text_model, max_text_length=text_len)
    val_ds   = MultimodalDataset(val_df,   modalities, cfg.encoder, train=False,
                                 tokenizer_name=text_model, max_text_length=text_len)
    test_ds  = MultimodalDataset(test_df,  modalities, cfg.encoder, train=False,
                                 tokenizer_name=text_model, max_text_length=text_len)

    bs      = cfg.training.batch_size
    nw      = cfg.training.num_workers
    pm      = cfg.training.pin_memory
    sampler = build_sampler(train_ds, bs, cfg.training.use_balanced_sampler, cfg.training.seed)

    train_loader = DataLoader(
        train_ds, batch_size=bs,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=nw, pin_memory=pm, drop_last=True,
    )
    val_loader = DataLoader(val_ds,  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm)
    test_loader= DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm)

    # ── 4. Optuna Stage A (optional) ──────────────────────────────────────────
    if args.tune and cfg.optuna.enabled:
        from stage1.training.optuna_tuner import OptunaStageA
        print("\n[run_stage1] Starting Optuna Stage A hyperparameter search...")
        tuner = OptunaStageA(
            base_cfg=cfg,
            train_df=train_df,
            val_df=val_df,
            modalities=modalities,
            num_classes=num_classes,
            device=device,
            results_root=cfg.results_root,
        )
        best_params = tuner.run()

        # Re-build config applying best params as overrides
        cfg = load_config(args.config, {**overrides, **best_params})

        # Rebuild datasets/loaders with potentially new batch_size
        bs = cfg.training.batch_size
        sampler = build_sampler(train_ds, bs, cfg.training.use_balanced_sampler, cfg.training.seed)
        train_loader = DataLoader(
            train_ds, batch_size=bs,
            sampler=sampler, shuffle=(sampler is None),
            num_workers=nw, pin_memory=pm, drop_last=True,
        )

    # ── 5. Model ──────────────────────────────────────────────────────────────
    from stage1.models.multimodal_model import MultimodalModel
    model = MultimodalModel(modalities, cfg, num_classes)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[run_stage1] Model parameters: {total_params:,}")

    # ── 6. Results layout (per PDF specification) ─────────────────────────────
    # results/<dataset>/<early|late>/<modality_only|fused_only|both>/<modalities>/
    modality_tag = "_".join(sorted(modalities))
    exp_dir = Path(cfg.results_root) / cfg.dataset_name / str(cfg.fusion_timing) / str(cfg.contrastive_placement) / modality_tag
    stage1_dir = exp_dir / "stage1_representation"
    stage1_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot in experiment folder
    from stage1.config.config import save_config
    save_config(cfg, exp_dir / "config_snapshot.yaml")

    # ── 7. Train ──────────────────────────────────────────────────────────────
    from stage1.training.trainer import Stage1Trainer
    trainer = Stage1Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        modalities=modalities,
        results_dir=str(stage1_dir),
        device=device,
    )
    best_metrics = trainer.train()

    print("\n[run_stage1] Stage 1 complete.")
    print(f"[run_stage1] Best metrics: {best_metrics}")


if __name__ == "__main__":
    main()

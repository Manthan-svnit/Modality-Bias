"""
optuna_tuner.py — Optuna Stage A hyperparameter search.

Runs N trials, each training the model for a shortened epoch count.
The objective is macro_f1 (or silhouette, or knn_accuracy) on the val set.

Usage (called from run_stage1.py or standalone):
    tuner = OptunaStageA(cfg, ...)
    best_params = tuner.run()
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import optuna
    _OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    _OPTUNA_AVAILABLE = False

import torch
from torch.utils.data import DataLoader

from stage1.config.config import ConfigNode
from stage1.data.dataset import MultimodalDataset
from stage1.data.sampler import build_sampler
from stage1.models.multimodal_model import MultimodalModel
from stage1.evaluation.metrics import (
    collect_embeddings,
    knn_accuracy,
    compute_silhouette,
    linear_probe_accuracy,
)
from stage1.losses import Stage1Loss
from stage1.training.trainer import build_optimizer, build_scheduler, Stage1Trainer


class OptunaStageA:
    """
    Optuna Stage A tuner.

    Parameters
    ----------
    base_cfg : ConfigNode
        Base configuration (will be overridden per trial).
    train_df, val_df : pd.DataFrame
        Training and validation manifests.
    modalities : list of str
    num_classes : int
    device : torch.device
    results_root : str | Path
    """

    def __init__(
        self,
        base_cfg: ConfigNode,
        train_df,
        val_df,
        modalities,
        num_classes: int,
        device: torch.device,
        results_root: str,
    ):
        if not _OPTUNA_AVAILABLE:
            raise ImportError("optuna is required for hyperparameter tuning. Please install optuna via `pip install optuna`.")
        self.base_cfg = base_cfg
        self.train_df = train_df
        self.val_df = val_df
        self.modalities = modalities
        self.num_classes = num_classes
        self.device = device
        self.results_root = Path(results_root)
        self.optuna_cfg = base_cfg.optuna

        # Ensure study DB directory exists
        db_uri: str = self.optuna_cfg.study_db
        if db_uri.startswith("sqlite:///"):
            db_path = Path(db_uri.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """
        Run the Optuna study.

        Returns
        -------
        best_params : dict
        """
        sampler = self._build_sampler()
        pruner  = self._build_pruner()

        study = optuna.create_study(
            study_name=f"stage_a_{self.base_cfg.dataset_name}",
            storage=self.optuna_cfg.study_db,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        n_trials = int(self.optuna_cfg.n_trials)
        n_jobs   = int(getattr(self.optuna_cfg, "n_jobs", 1))

        study.optimize(
            self._objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
        )

        print(f"\n[Optuna] Best trial: {study.best_trial.number}")
        print(f"[Optuna] Best value: {study.best_value:.4f}")
        print(f"[Optuna] Best params: {study.best_params}")

        # Save best params
        best_path = self.results_root / self.base_cfg.dataset_name / "optuna" / "best_params.json"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(best_path, "w") as f:
            json.dump(study.best_params, f, indent=2)

        return study.best_params

    # ──────────────────────────────────────────────────────────────────────────
    # Objective
    # ──────────────────────────────────────────────────────────────────────────

    def _objective(self, trial: optuna.Trial) -> float:
        """Single Optuna trial: sample params, train, return val metric."""
        cfg = self._apply_trial_params(trial)

        # Build datasets
        train_ds = MultimodalDataset(
            self.train_df, self.modalities, cfg.encoder, train=True,
            tokenizer_name=cfg.encoder.text.model_name if "text" in self.modalities else "bert-base-uncased",
            max_text_length=cfg.encoder.text.max_length if "text" in self.modalities else 128,
        )
        val_ds = MultimodalDataset(
            self.val_df, self.modalities, cfg.encoder, train=False,
            tokenizer_name=cfg.encoder.text.model_name if "text" in self.modalities else "bert-base-uncased",
            max_text_length=cfg.encoder.text.max_length if "text" in self.modalities else 128,
        )

        bs = int(cfg.training.batch_size)
        sampler = build_sampler(train_ds, bs, cfg.training.use_balanced_sampler, cfg.training.seed)
        shuffle = sampler is None

        train_loader = DataLoader(
            train_ds,
            batch_size=bs,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=int(cfg.training.num_workers),
            pin_memory=cfg.training.pin_memory,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=int(cfg.training.num_workers),
            pin_memory=cfg.training.pin_memory,
        )

        # Build model
        model = MultimodalModel(self.modalities, cfg, self.num_classes).to(self.device)

        # Quick training (short epoch budget for search)
        trial_epochs = min(int(cfg.training.epochs), 20)
        cfg_trial = copy.deepcopy(cfg)
        cfg_trial.training.epochs = trial_epochs

        criterion = Stage1Loss(
            loss_cfg=cfg.loss,
            contrastive_placement=cfg.contrastive_placement,
            modalities=self.modalities,
        )
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg, trial_epochs)

        for epoch in range(1, trial_epochs + 1):
            model.train()
            for batch in train_loader:
                batch_gpu = self._to_device(batch)
                optimizer.zero_grad()
                out = model(batch_gpu)
                loss_d = criterion(out, batch_gpu["label"], epoch=epoch)
                loss_d["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if scheduler:
                scheduler.step()

            # Pruning check
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Evaluate
        z_t, z_mod_t, y_t = collect_embeddings(model, train_loader, self.modalities, self.device)
        z_v, z_mod_v, y_v = collect_embeddings(model, val_loader, self.modalities, self.device)

        objective_name = str(self.optuna_cfg.objective)
        if objective_name == "macro_f1":
            score = linear_probe_accuracy(z_t, y_t, z_v, y_v)
        elif objective_name == "knn_accuracy":
            score = knn_accuracy(z_t, y_t, z_v, y_v, k=int(self.base_cfg.evaluation.knn_k))
        elif objective_name == "silhouette":
            score = compute_silhouette(z_v, y_v)
        else:
            score = linear_probe_accuracy(z_t, y_t, z_v, y_v)

        return score

    # ──────────────────────────────────────────────────────────────────────────
    # Search space → config
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_trial_params(self, trial: optuna.Trial) -> ConfigNode:
        """Sample from search space and return a modified config."""
        cfg = copy.deepcopy(self.base_cfg)
        ss = self.optuna_cfg.search_space

        def _suggest(name, spec):
            if isinstance(spec, list) and len(spec) == 2 and isinstance(spec[0], (int, float)):
                # Range: [min, max]
                lo, hi = spec
                if isinstance(lo, int) and isinstance(hi, int):
                    return trial.suggest_int(name, lo, hi)
                return trial.suggest_float(name, lo, hi, log=True)
            elif isinstance(spec, list):
                return trial.suggest_categorical(name, spec)
            return spec

        # Embedding dim
        if "d" in ss:
            d = _suggest("d", ss["d"])
            # Make divisible by max num_heads
            max_heads = max(ss.get("num_heads", [4])) if isinstance(ss.get("num_heads"), list) else 4
            d = max(max_heads, (d // max_heads) * max_heads)
            cfg.encoder.d = d

        # Encoder depth/width (applied uniformly across modalities)
        if "encoder_depth" in ss:
            depth = _suggest("encoder_depth", ss["encoder_depth"])
            for mod in self.modalities:
                if mod == "image":
                    cfg.encoder.image.cnn_depth = depth
                elif mod == "audio":
                    cfg.encoder.audio.cnn_depth = depth
                elif mod == "video":
                    cfg.encoder.video.frame_cnn_depth = depth

        if "encoder_width" in ss:
            width = _suggest("encoder_width", ss["encoder_width"])
            for mod in self.modalities:
                if mod == "image":
                    cfg.encoder.image.cnn_width = width
                elif mod == "audio":
                    cfg.encoder.audio.cnn_width = width
                elif mod == "video":
                    cfg.encoder.video.frame_cnn_width = width

        # Gating
        if "gating_hidden" in ss:
            cfg.gating.hidden_dim = _suggest("gating_hidden", ss["gating_hidden"])

        # Fusion
        if "num_heads" in ss:
            nh = _suggest("num_heads", ss["num_heads"])
            # Ensure divisibility
            d = cfg.encoder.d
            while d % nh != 0 and nh > 1:
                nh -= 1
            cfg.fusion.num_heads = nh

        if "num_layers" in ss:
            cfg.fusion.num_layers = _suggest("num_layers", ss["num_layers"])

        if "fusion_dropout" in ss:
            cfg.fusion.dropout = _suggest("fusion_dropout", ss["fusion_dropout"])

        # Projection
        if "proj_dim" in ss:
            cfg.projection.proj_dim = _suggest("proj_dim", ss["proj_dim"])

        # Loss weights
        for w_name in ["w_con_fused", "w_con_mod", "w_cls", "w_rec", "w_align"]:
            if w_name in ss:
                val = _suggest(w_name, ss[w_name])
                cfg.loss[w_name] = val

        if "supcon_temperature" in ss:
            cfg.loss.supcon_temperature = _suggest("supcon_temperature", ss["supcon_temperature"])

        # Training
        if "lr" in ss:
            cfg.training.lr = _suggest("lr", ss["lr"])

        if "batch_size" in ss:
            cfg.training.batch_size = _suggest("batch_size", ss["batch_size"])

        if "weight_decay" in ss:
            cfg.training.weight_decay = _suggest("weight_decay", ss["weight_decay"])

        if "epochs" in ss:
            cfg.training.epochs = _suggest("epochs", ss["epochs"])

        return cfg

    def _to_device(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device, non_blocking=True)
            elif isinstance(v, dict):
                out[k] = {
                    dk: dv.to(self.device, non_blocking=True)
                    if isinstance(dv, torch.Tensor) else dv
                    for dk, dv in v.items()
                }
            else:
                out[k] = v
        return out

    def _build_sampler(self):
        name = str(getattr(self.optuna_cfg, "sampler", "tpe")).lower()
        if name == "tpe":
            return optuna.samplers.TPESampler()
        if name == "random":
            return optuna.samplers.RandomSampler()
        if name == "cmaes":
            return optuna.samplers.CmaEsSampler()
        return optuna.samplers.TPESampler()

    def _build_pruner(self):
        name = str(getattr(self.optuna_cfg, "pruner", "median")).lower()
        if name == "median":
            return optuna.pruners.MedianPruner()
        if name == "hyperband":
            return optuna.pruners.HyperbandPruner()
        if name == "none":
            return optuna.pruners.NopPruner()
        return optuna.pruners.MedianPruner()

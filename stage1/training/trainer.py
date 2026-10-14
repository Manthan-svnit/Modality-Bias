"""
trainer.py — Stage 1 training loop.

Handles:
  - Optimizer and scheduler construction
  - Per-epoch train / val loops
  - Loss computation with warmup
  - CKA logging every N epochs
  - t-SNE / UMAP saving
  - Checkpoint saving (best val metric + latest)
  - z_fused saving (train/val/test) at end of training
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from stage1.losses import Stage1Loss
from stage1.evaluation.metrics import (
    collect_embeddings,
    run_separation_metrics,
    save_2d_projections,
    compute_cka,
)


# ──────────────────────────────────────────────────────────────────────────────
# Optimizer / scheduler factories
# ──────────────────────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    opt_name = cfg.training.optimizer.lower()
    lr = float(cfg.training.lr)
    wd = float(cfg.training.weight_decay)
    if opt_name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "sgd":
        return SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {opt_name}")


def build_scheduler(optimizer, cfg, num_epochs: int):
    sched_name = cfg.training.scheduler.lower()
    if sched_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    if sched_name == "step":
        return StepLR(optimizer, step_size=max(1, num_epochs // 3), gamma=0.1)
    if sched_name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {sched_name}")


# ──────────────────────────────────────────────────────────────────────────────
# Main Trainer
# ──────────────────────────────────────────────────────────────────────────────

class Stage1Trainer:
    """
    Parameters
    ----------
    model : MultimodalModel
    cfg   : ConfigNode  (full config)
    train_loader, val_loader, test_loader : DataLoader
    modalities : list of str
    results_dir : str | Path   (per-run output directory)
    device : torch.device
    """

    def __init__(
        self,
        model: nn.Module,
        cfg,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader],
        modalities: List[str],
        results_dir: str,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.modalities = modalities
        self.results_dir = Path(results_dir)
        self.device = device

        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "checkpoints").mkdir(exist_ok=True)
        (self.results_dir / "embeddings").mkdir(exist_ok=True)
        (self.results_dir / "projections").mkdir(exist_ok=True)

        # Loss
        self.criterion = Stage1Loss(
            loss_cfg=cfg.loss,
            contrastive_placement=cfg.contrastive_placement,
            modalities=modalities,
        )

        # Optimizer & scheduler
        self.optimizer = build_optimizer(model, cfg)
        self.num_epochs = cfg.training.epochs
        self.scheduler = build_scheduler(self.optimizer, cfg, self.num_epochs)

        # Tracking
        self.history: List[Dict] = []
        self.best_val_metric = float("-inf")
        self.best_epoch = 0
        self.cka_log_every = getattr(cfg.evaluation, "log_cka_every_n_epochs", 1)

    # ──────────────────────────────────────────────────────────────────────────
    # Train one epoch
    # ──────────────────────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = {}
        n_batches = 0

        for batch in self.train_loader:
            # Move tensors to device
            batch = self._to_device(batch)
            labels = batch["label"]

            self.optimizer.zero_grad()
            out = self.model(batch)
            loss_dict = self.criterion(out, labels, epoch=epoch)
            loss_dict["total"].backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k, v in loss_dict.items():
                totals[k] = totals.get(k, 0.0) + float(v.detach().cpu() if hasattr(v, "detach") else v)
            n_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    # ──────────────────────────────────────────────────────────────────────────
    # Validate one epoch
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        totals: Dict[str, float] = {}
        n_batches = 0

        for batch in self.val_loader:
            batch = self._to_device(batch)
            labels = batch["label"]
            out = self.model(batch)
            loss_dict = self.criterion(out, labels, epoch=epoch)
            for k, v in loss_dict.items():
                totals[k] = totals.get(k, 0.0) + float(v.detach().cpu() if hasattr(v, "detach") else v)
            n_batches += 1

        return {f"val_{k}": v / max(n_batches, 1) for k, v in totals.items()}

    # ──────────────────────────────────────────────────────────────────────────
    # Full training loop
    # ──────────────────────────────────────────────────────────────────────────

    def train(self) -> Dict[str, float]:
        """
        Run the full training loop.

        Returns
        -------
        best metrics dict from the best epoch
        """
        print(f"\n{'='*60}")
        print(f"Stage 1 Training - {self.num_epochs} epochs")
        print(f"Results dir: {self.results_dir}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.num_epochs + 1):
            t0 = time.time()

            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._val_epoch(epoch)

            # ── CKA logging ──────────────────────────────────────────────────
            cka_metrics: Dict[str, float] = {}
            if epoch % self.cka_log_every == 0:
                z_fused_v, z_mod_v, _ = collect_embeddings(
                    self.model, self.val_loader, self.modalities, self.device
                )
                cka_metrics = compute_cka(z_fused_v, z_mod_v)

            # ── Combine metrics ───────────────────────────────────────────────
            row = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **val_metrics,
                **cka_metrics,
                "lr": self.optimizer.param_groups[0]["lr"],
                "elapsed_s": time.time() - t0,
            }

            # ── Separation metrics (end of epoch, full val pass) ──────────────
            z_fused_t, z_mod_t, y_t = collect_embeddings(
                self.model, self.train_loader, self.modalities, self.device
            )
            z_fused_v, z_mod_v, y_v = collect_embeddings(
                self.model, self.val_loader, self.modalities, self.device
            )
            sep = run_separation_metrics(
                z_fused_t, y_t, z_fused_v, y_v, z_mod_t, z_mod_v,
                self.cfg.evaluation,
            )
            row.update(sep)

            self.history.append(row)
            self._log_epoch(row)

            # ── Best model checkpoint ─────────────────────────────────────────
            val_metric = row.get("linear_probe_macro_f1", row.get("val_total", 0.0))
            # For loss-based metric: negate (lower is better)
            if "linear_probe_macro_f1" not in row:
                val_metric = -val_metric
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self.best_epoch = epoch
                self._save_checkpoint("best.pt")

            # ── t-SNE / UMAP (every 10 epochs and last epoch) ─────────────────
            if epoch % 10 == 0 or epoch == self.num_epochs:
                save_2d_projections(
                    z_fused_v, y_v,
                    str(self.results_dir / "projections"),
                    epoch,
                    self.cfg.evaluation,
                )

            # ── Latest checkpoint ─────────────────────────────────────────────
            self._save_checkpoint("latest.pt")

        # ── Save history ──────────────────────────────────────────────────────
        self._save_history()

        # ── Save embeddings for all splits ────────────────────────────────────
        self._save_all_embeddings()

        print(f"\nTraining complete. Best epoch: {self.best_epoch} "
              f"(val metric: {self.best_val_metric:.4f})")
        return self.history[self.best_epoch - 1]

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _to_device(self, batch: dict) -> dict:
        """Move all tensor values in batch to device."""
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

    def _save_checkpoint(self, name: str) -> None:
        path = self.results_dir / "checkpoints" / name
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epoch": self.history[-1]["epoch"] if self.history else 0,
                "best_val_metric": self.best_val_metric,
            },
            path,
        )

    def _save_history(self) -> None:
        path = self.results_dir / "training_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)

        # Also save separation_metrics.csv and cka_log.csv per PDF spec
        import pandas as pd
        df = pd.DataFrame(self.history)
        sep_cols = [c for c in df.columns if any(k in c for k in ("knn", "f1", "silhouette", "linear_probe")) or c == "epoch"]
        if sep_cols:
            df[sep_cols].to_csv(self.results_dir / "separation_metrics.csv", index=False)
        cka_cols = [c for c in df.columns if "cka" in c or c == "epoch"]
        if cka_cols:
            df[cka_cols].to_csv(self.results_dir / "cka_log.csv", index=False)

    def _save_all_embeddings(self) -> None:
        """
        Save z_fused (and per-modality z_m) for train, val, and test splits
        as .npy files. Labels saved alongside.
        Saves to both 'features/' and 'embeddings/' for full spec compatibility.
        """
        loaders = {"train": self.train_loader, "val": self.val_loader}
        if self.test_loader is not None:
            loaders["test"] = self.test_loader

        # Load best checkpoint
        best_ckpt = self.results_dir / "checkpoints" / "best.pt"
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])

        for split, loader in loaders.items():
            z_fused, z_mod, labels = collect_embeddings(
                self.model, loader, self.modalities, self.device
            )
            # Save to features/ (PDF spec) and embeddings/
            for folder_name in ("features", "embeddings"):
                out_dir = self.results_dir / folder_name / split
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / "z_fused.npy", z_fused)
                np.save(out_dir / "labels.npy", labels)
                for mod, z_m in z_mod.items():
                    np.save(out_dir / f"z_{mod}.npy", z_m)
            print(f"[trainer] Saved {split} features to {self.results_dir / 'features' / split}")

    def _log_epoch(self, row: dict) -> None:
        ep = row["epoch"]
        t_loss = row.get("train_total", float("nan"))
        v_loss = row.get("val_total", float("nan"))
        knn = row.get("knn_acc", float("nan"))
        f1  = row.get("linear_probe_macro_f1", float("nan"))
        sil = row.get("silhouette", float("nan"))
        print(
            f"Epoch {ep:3d}/{self.num_epochs} | "
            f"train_loss={t_loss:.4f}  val_loss={v_loss:.4f} | "
            f"KNN={knn:.3f}  F1={f1:.3f}  Sil={sil:.3f} | "
            f"lr={row['lr']:.2e}  [{row['elapsed_s']:.1f}s]"
        )

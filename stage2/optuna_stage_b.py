"""
optuna_stage_b.py — Optuna Stage B hyperparameter search.

Stage B is CHEAP: it works on cached z_fused vectors (never re-runs the encoder).

Jointly tunes:
  - RCS params: eta, k, lambda
  - GMM params: reg_covar, n_init
  - Stage 3 classifier params: layers, hidden_dim, dropout, lr, weight_decay, epochs

Objective: validation Macro-F1 (with optional second objective: minority recall).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import optuna
    _OPTUNA_OK = True
except ImportError:
    optuna = None
    _OPTUNA_OK = False

from stage2.rcs import RCSBalancer


class OptunaStageB:
    """
    Optuna Stage B tuner.

    Parameters
    ----------
    z_train, y_train : np.ndarray  — train embeddings + labels from Stage 1
    z_val, y_val     : np.ndarray  — val embeddings + labels (NEVER augmented)
    stage_b_cfg      : ConfigNode  — cfg.optuna_stage_b sub-config
    results_dir      : str | Path  — where to save best_params.json and study.db
    """

    def __init__(
        self,
        z_train: np.ndarray,
        y_train: np.ndarray,
        z_val: np.ndarray,
        y_val: np.ndarray,
        stage_b_cfg,
        results_dir: str,
    ):
        if not _OPTUNA_OK:
            raise ImportError("optuna is required for Stage B tuning.")

        self.z_train = z_train
        self.y_train = y_train
        self.z_val = z_val
        self.y_val = y_val
        self.cfg = stage_b_cfg
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Ensure sqlite db directory exists
        db_uri = str(getattr(stage_b_cfg, "study_db", f"sqlite:///{self.results_dir}/stage_b.db"))
        if db_uri.startswith("sqlite:///"):
            db_path = Path(db_uri.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self.study_db = db_uri

    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Run Stage B study. Returns best params dict."""
        sampler = optuna.samplers.TPESampler()
        pruner  = optuna.pruners.MedianPruner()

        study = optuna.create_study(
            study_name=f"stage_b",
            storage=self.study_db,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        n_trials = int(getattr(self.cfg, "n_trials", 50))
        study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)

        print(f"\n[Stage B] Best trial: {study.best_trial.number}")
        print(f"[Stage B] Best value: {study.best_value:.4f}")
        print(f"[Stage B] Best params: {study.best_params}")

        # Save
        with open(self.results_dir / "stage_b_best_params.json", "w") as f:
            json.dump(study.best_params, f, indent=2)

        return study.best_params

    def _objective(self, trial: optuna.Trial) -> float:
        """Sample RCS + classifier params, balance, train MLP, return val macro-F1."""
        # ── RCS params ────────────────────────────────────────────────────────
        eta = trial.suggest_float("eta", 0.0, 1.0)
        k   = trial.suggest_int("k", 1, 15)
        lam = trial.suggest_float("lambda", 0.0, 1.0)

        # ── GMM params ────────────────────────────────────────────────────────
        gmm_reg_covar = trial.suggest_float("gmm_reg_covar", 1e-6, 1e-2, log=True)

        # ── Balance ───────────────────────────────────────────────────────────
        balancer = RCSBalancer(
            eta=eta, k=k, lam=lam,
            gmm_reg_covar=gmm_reg_covar,
            random_state=42,
        )
        try:
            z_bal, y_bal, _ = balancer.fit_transform(self.z_train, self.y_train)
        except Exception:
            return 0.0

        # ── Classifier params ─────────────────────────────────────────────────
        clf_lr         = trial.suggest_float("clf_lr", 1e-4, 1e-1, log=True)
        clf_hidden     = trial.suggest_categorical("clf_hidden", [64, 128, 256, 512])
        clf_layers     = trial.suggest_int("clf_layers", 1, 4)
        clf_dropout    = trial.suggest_float("clf_dropout", 0.0, 0.5)
        clf_epochs     = trial.suggest_int("clf_epochs", 10, 50)
        clf_weight_decay = trial.suggest_float("clf_weight_decay", 1e-6, 1e-2, log=True)

        # ── Train mini-MLP on balanced z_fused ────────────────────────────────
        macro_f1 = self._train_and_eval(
            z_bal, y_bal,
            lr=clf_lr, hidden=clf_hidden, n_layers=clf_layers,
            dropout=clf_dropout, epochs=clf_epochs, weight_decay=clf_weight_decay,
        )
        return macro_f1

    def _train_and_eval(
        self,
        z_train: np.ndarray,
        y_train: np.ndarray,
        lr: float,
        hidden: int,
        n_layers: int,
        dropout: float,
        epochs: int,
        weight_decay: float,
    ) -> float:
        """Train a small MLP and return val macro-F1."""
        import torch
        import torch.nn as nn
        from sklearn.metrics import f1_score
        from sklearn.preprocessing import LabelEncoder

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encode labels to 0-indexed
        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_train)
        y_val_enc = le.transform(self.y_val)
        num_classes = len(le.classes_)

        d = z_train.shape[1]

        # Build MLP
        layers_list = []
        in_dim = d
        for _ in range(n_layers):
            layers_list += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden
        layers_list.append(nn.Linear(in_dim, num_classes))
        model = nn.Sequential(*layers_list).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        # Convert to tensors
        X_t = torch.from_numpy(z_train.astype(np.float32)).to(device)
        Y_t = torch.from_numpy(y_tr_enc.astype(np.int64)).to(device)

        bs = min(256, len(z_train))
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(len(X_t))
            for i in range(0, len(X_t), bs):
                idx = perm[i:i+bs]
                optimizer.zero_grad()
                loss = criterion(model(X_t[idx]), Y_t[idx])
                loss.backward()
                optimizer.step()

        # Eval
        model.eval()
        X_v = torch.from_numpy(self.z_val.astype(np.float32)).to(device)
        with torch.no_grad():
            preds = model(X_v).argmax(1).cpu().numpy()

        return float(f1_score(y_val_enc, preds, average="macro", zero_division=0))

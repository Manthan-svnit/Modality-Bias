"""
evaluator.py — Multi-seed Trainer & Evaluator for Stage 3.

Per PDF Specification:
"Final: retrain with best params over several seeds; report test metrics (mean, std)
 from that run only."
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stage3.models import MLPClassifier
from stage3.metrics import compute_classification_metrics


def train_single_run(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    params: Dict[str, Any],
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Tuple[MLPClassifier, Dict[str, Any]]:
    """
    Train an MLP classifier on (z_train, y_train) and evaluate best epoch on (z_val, y_val).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = z_train.shape[1]
    hidden_dim = int(params.get("clf_hidden", 256))
    num_layers = int(params.get("clf_layers", 2))
    dropout = float(params.get("clf_dropout", 0.2))
    lr = float(params.get("clf_lr", 1e-3))
    weight_decay = float(params.get("clf_weight_decay", 1e-4))
    epochs = int(params.get("clf_epochs", 30))
    batch_size = int(params.get("clf_batch_size", 64))

    # Ensure labels are 0-indexed contiguous int64
    unique_labels = np.unique(np.concatenate([y_train, y_val]))
    if not (len(unique_labels) > 0 and unique_labels[0] == 0 and unique_labels[-1] == len(unique_labels) - 1):
        label_map = {orig: new for new, orig in enumerate(sorted(unique_labels))}
        y_train = np.array([label_map[l] for l in y_train], dtype=np.int64)
        y_val   = np.array([label_map[l] for l in y_val],   dtype=np.int64)
        num_classes = len(unique_labels)
    else:
        num_classes = max(num_classes, len(unique_labels))

    model = MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Tensor datasets
    train_ds = TensorDataset(torch.from_numpy(z_train).float(), torch.from_numpy(y_train).long())
    val_ds = TensorDataset(torch.from_numpy(z_val).float(), torch.from_numpy(y_val).long())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_macro_f1 = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        # Validation pass
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                logits = model(bx)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(by.numpy())

        val_preds_arr = np.array(val_preds)
        val_targets_arr = np.array(val_targets)
        val_metrics = compute_classification_metrics(val_targets_arr, val_preds_arr, num_classes=num_classes)
        val_f1 = val_metrics["macro_f1"]

        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_val_macro_f1": best_val_macro_f1}


def evaluate_model(
    model: MLPClassifier,
    z_eval: np.ndarray,
    y_eval: np.ndarray,
    num_classes: int,
    minority_classes: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Run evaluation on an arbitrary split and return complete metrics dict.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    ds = TensorDataset(torch.from_numpy(z_eval).float())
    loader = DataLoader(ds, batch_size=128, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device)
            logits = model(bx)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    preds_arr = np.array(all_preds)
    # y_eval is already 0-indexed by caller
    y_eval_mapped = np.asarray(y_eval, dtype=np.int64)

    num_classes_eff = num_classes if num_classes is not None else len(unique_eval)
    return compute_classification_metrics(
        y_eval_mapped, preds_arr, minority_classes=minority_classes, num_classes=num_classes_eff
    )


def run_multi_seed_evaluation(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    z_test: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    num_classes: int,
    params: Dict[str, Any],
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    minority_classes: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Train and evaluate over multiple random seeds.
    Reports mean and std for all test metrics as strictly required by PDF.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_seed_metrics = []
    best_model_overall = None
    best_seed_val_f1 = -1.0

    target_z = z_test if z_test is not None else z_val
    target_y = y_test if y_test is not None else y_val
    eval_split_name = "test" if z_test is not None else "val"

    # Create consistent 0-indexed label mapping across all splits
    all_known_labels = np.unique(np.concatenate([y_train, y_val, target_y]))
    label_map = {orig: new for new, orig in enumerate(sorted(all_known_labels))}
    y_train_mapped = np.array([label_map[l] for l in y_train], dtype=np.int64)
    y_val_mapped = np.array([label_map[l] for l in y_val], dtype=np.int64)
    target_y_mapped = np.array([label_map[l] for l in target_y], dtype=np.int64)
    num_classes_eff = len(label_map)
    mapped_minority = [label_map[c] for c in minority_classes if c in label_map] if minority_classes else None

    for s in seeds:
        model, train_info = train_single_run(
            z_train=z_train,
            y_train=y_train_mapped,
            z_val=z_val,
            y_val=y_val_mapped,
            num_classes=num_classes_eff,
            params=params,
            seed=s,
            device=device,
        )

        test_metrics = evaluate_model(
            model=model,
            z_eval=target_z,
            y_eval=target_y_mapped,
            num_classes=num_classes_eff,
            minority_classes=mapped_minority,
            device=device,
        )
        test_metrics["seed"] = s
        test_metrics["val_macro_f1"] = train_info["best_val_macro_f1"]
        all_seed_metrics.append(test_metrics)

        if train_info["best_val_macro_f1"] > best_seed_val_f1:
            best_seed_val_f1 = train_info["best_val_macro_f1"]
            best_model_overall = model

    # Aggregate scalar metrics
    scalar_keys = ["balanced_accuracy", "macro_f1", "mcc", "g_mean", "minority_recall", "accuracy"]
    summary: Dict[str, Any] = {
        "eval_split": eval_split_name,
        "seeds": list(seeds),
        "mean": {},
        "std": {},
        "per_class_f1_mean": {},
        "per_seed": all_seed_metrics,
    }

    for k in scalar_keys:
        vals = [m[k] for m in all_seed_metrics]
        summary["mean"][k] = float(np.mean(vals))
        summary["std"][k] = float(np.std(vals))

    # Aggregate per_class_f1
    classes = [str(c) for c in range(num_classes)]
    for c in classes:
        c_f1s = [m["per_class_f1"].get(c, 0.0) for m in all_seed_metrics]
        summary["per_class_f1_mean"][c] = {
            "mean": float(np.mean(c_f1s)),
            "std": float(np.std(c_f1s)),
        }

    return {
        "summary": summary,
        "best_model": best_model_overall,
    }

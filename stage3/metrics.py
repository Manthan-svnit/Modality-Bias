"""
metrics.py — Comprehensive Stage 3 Evaluation Metrics.

Per PDF Specification:
"Balanced Accuracy, Macro F1, MCC, G-Mean, minority-class recall, per-class F1
(not plain accuracy)."
"""

import math
from typing import Dict, List, Optional, Sequence, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    confusion_matrix,
)


def compute_g_mean(y_true: np.ndarray, y_pred: np.ndarray, num_classes: Optional[int] = None) -> float:
    """
    Compute Geometric Mean (G-Mean) of per-class sensitivities / recalls:
      G-Mean = (prod_{c=1}^C Recall_c) ** (1 / C)
    """
    if num_classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    else:
        classes = np.arange(num_classes)

    recalls = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    # Filter classes that have actual true instances
    present_mask = [np.sum(y_true == c) > 0 for c in classes]
    valid_recalls = recalls[present_mask]

    if len(valid_recalls) == 0:
        return 0.0

    # Geometric mean via log-sum-exp to avoid underflow
    log_recalls = np.log(np.maximum(valid_recalls, 1e-12))
    return float(np.exp(np.mean(log_recalls)))


def compute_minority_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    minority_classes: Sequence[int],
) -> float:
    """
    Compute average recall specifically for identified minority classes.
    """
    if not minority_classes:
        return 0.0

    recalls = recall_score(y_true, y_pred, labels=list(minority_classes), average=None, zero_division=0)
    return float(np.mean(recalls)) if len(recalls) > 0 else 0.0


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    minority_classes: Optional[Sequence[int]] = None,
    num_classes: Optional[int] = None,
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Compute all required classification metrics according to PDF spec.

    Returns dict containing:
      - balanced_accuracy
      - macro_f1
      - mcc (Matthews Correlation Coefficient)
      - g_mean
      - minority_recall (if minority_classes specified)
      - per_class_f1 (dict mapping str(class) -> float)
      - per_class_recall (dict mapping str(class) -> float)
      - accuracy
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if num_classes is None:
        classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    else:
        classes = list(range(num_classes))

    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    g_mean = compute_g_mean(y_true, y_pred, num_classes=len(classes))
    acc = float(accuracy_score(y_true, y_pred))

    per_f1 = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    per_rec = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)

    per_class_f1_dict = {str(c): float(score) for c, score in zip(classes, per_f1)}
    per_class_rec_dict = {str(c): float(score) for c, score in zip(classes, per_rec)}

    # Minority recall
    if minority_classes:
        min_recall = compute_minority_recall(y_true, y_pred, minority_classes)
    else:
        # Infer minority classes as those with sample counts below median count
        counts = np.bincount(y_true, minlength=len(classes))
        median_count = np.median(counts)
        inferred_min = [c for c, cnt in enumerate(counts) if cnt < median_count]
        min_recall = compute_minority_recall(y_true, y_pred, inferred_min)

    return {
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "g_mean": g_mean,
        "minority_recall": min_recall,
        "accuracy": acc,
        "per_class_f1": per_class_f1_dict,
        "per_class_recall": per_class_rec_dict,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
    }

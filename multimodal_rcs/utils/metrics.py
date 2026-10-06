from typing import Dict, List, Optional, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


def compute_g_mean(per_class_recalls: np.ndarray, eps: float = 1e-8) -> float:
    """Computes Geometric Mean (G-Mean) of per-class recalls.
    
    Formula:
        G-Mean = (prod_{c=1}^K Recall_c)^{1/K}
    """
    recalls = np.clip(per_class_recalls, eps, 1.0)
    log_recalls = np.log(recalls)
    g_mean = float(np.exp(np.mean(log_recalls)))
    return g_mean


def evaluate_classifier_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    minority_classes: Optional[List[int]] = None,
    num_classes: int = 10,
) -> Dict[str, Any]:
    """Computes all required evaluation metrics:
      - Balanced Accuracy
      - Macro F1
      - Matthews Correlation Coefficient (MCC)
      - G-Mean
      - Minority-class Recall
      - Per-class Recalls
      - Standard Accuracy
      - Confusion Matrix

    Args:
        y_true: Ground-truth class labels (N,)
        y_pred: Predicted class labels (N,)
        minority_classes: List of minority class indices for calculating minority recall.
        num_classes: Total number of classes (default: 10).

    Returns:
        Dictionary containing metric values.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    # Confusion matrix & Per-class recall
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    row_sums = cm.sum(axis=1)
    # Avoid div-by-zero if any class has 0 samples
    per_class_recall = np.zeros(num_classes, dtype=float)
    for c in range(num_classes):
        if row_sums[c] > 0:
            per_class_recall[c] = cm[c, c] / row_sums[c]

    g_mean = compute_g_mean(per_class_recall)

    # Minority-class recall
    if minority_classes is not None and len(minority_classes) > 0:
        minority_recalls = [per_class_recall[c] for c in minority_classes]
        minority_recall_mean = float(np.mean(minority_recalls))
    else:
        # Default fallback to bottom 3 classes (7, 8, 9)
        minority_recalls = [per_class_recall[c] for c in [7, 8, 9] if c < num_classes]
        minority_recall_mean = float(np.mean(minority_recalls)) if minority_recalls else 0.0

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "g_mean": g_mean,
        "minority_recall": minority_recall_mean,
        "per_class_recall": per_class_recall.tolist(),
        "confusion_matrix": cm.tolist(),
    }

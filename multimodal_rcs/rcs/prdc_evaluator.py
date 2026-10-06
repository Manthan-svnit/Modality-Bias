from typing import Dict, Any, Optional
import numpy as np
import logging
from prdc import compute_prdc

logger = logging.getLogger(__name__)


def evaluate_prdc(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    nearest_k: int = 5,
) -> Dict[str, float]:
    """Computes Precision, Recall, Density, Coverage (PRDC) between real and synthetic features.

    Args:
        real_features: Real feature vectors (N_real, D)
        fake_features: Synthetic feature vectors (N_fake, D)
        nearest_k: Number of nearest neighbors to construct hyperspheres (default: 5)

    Returns:
        Dict with keys 'precision', 'recall', 'density', 'coverage'.
    """
    real_features = np.asarray(real_features, dtype=np.float32)
    fake_features = np.asarray(fake_features, dtype=np.float32)

    if len(real_features) == 0 or len(fake_features) == 0:
        return {"precision": 0.0, "recall": 0.0, "density": 0.0, "coverage": 0.0}

    # If sample size is smaller than nearest_k, adjust k
    k = min(nearest_k, len(real_features) - 1, len(fake_features) - 1)
    if k < 1:
        k = 1

    metrics = compute_prdc(
        real_features=real_features,
        fake_features=fake_features,
        nearest_k=k,
    )
    return metrics


def evaluate_classwise_prdc(
    real_features: np.ndarray,
    real_labels: np.ndarray,
    synthetic_by_class: Dict[int, np.ndarray],
    nearest_k: int = 5,
) -> Dict[int, Dict[str, float]]:
    """Evaluates PRDC for each class that received synthetic samples."""
    classwise_metrics = {}
    for c, fake_feats in synthetic_by_class.items():
        real_c = real_features[real_labels == c]
        metrics = evaluate_prdc(real_c, fake_feats, nearest_k=nearest_k)
        classwise_metrics[c] = metrics
        logger.info(
            f"Class {c} PRDC: Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, Density={metrics['density']:.4f}, "
            f"Coverage={metrics['coverage']:.4f}"
        )
    return classwise_metrics

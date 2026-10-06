from typing import Tuple, Dict, Any
import numpy as np
import logging
from imblearn.over_sampling import RandomOverSampler, SMOTE

logger = logging.getLogger(__name__)


def apply_no_correction(
    features: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Baseline 1: No correction (original imbalanced representations)."""
    return features.copy(), labels.copy()


def apply_random_oversampling(
    features: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Baseline 2: Random Oversampling on fused vectors."""
    ros = RandomOverSampler(random_state=random_state)
    features_resampled, labels_resampled = ros.fit_resample(features, labels)
    return np.asarray(features_resampled, dtype=np.float32), np.asarray(labels_resampled, dtype=np.int64)


def apply_smote(
    features: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Baseline 3: SMOTE on fused vectors."""
    unique_classes, counts = np.unique(labels, return_counts=True)
    min_count = np.min(counts)
    # k_neighbors must be strictly less than the smallest class sample count
    k_neighbors = min(5, max(1, min_count - 1))

    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    features_resampled, labels_resampled = smote.fit_resample(features, labels)
    return np.asarray(features_resampled, dtype=np.float32), np.asarray(labels_resampled, dtype=np.int64)

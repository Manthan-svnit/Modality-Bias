import numpy as np
import torch
from multimodal_rcs.utils import (
    linear_cka,
    compute_g_mean,
    evaluate_classifier_predictions,
    apply_random_oversampling,
    apply_smote,
)


def test_linear_cka():
    torch.manual_seed(42)
    # Identical representations should have CKA close to 1.0
    x = torch.randn(100, 64)
    cka_identical = linear_cka(x, x)
    assert abs(cka_identical - 1.0) < 1e-4

    # Independent orthogonal representations should have low CKA
    y = torch.randn(100, 64)
    cka_indep = linear_cka(x, y)
    assert 0.0 <= cka_indep < 0.5


def test_metrics_evaluation():
    # 10 classes
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9])
    y_pred = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 0, 8, 8, 9, 0])

    metrics = evaluate_classifier_predictions(
        y_true, y_pred, minority_classes=[7, 8, 9], num_classes=10
    )

    assert "balanced_accuracy" in metrics
    assert "macro_f1" in metrics
    assert "mcc" in metrics
    assert "g_mean" in metrics
    assert "minority_recall" in metrics
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert 0.0 <= metrics["g_mean"] <= 1.0
    assert -1.0 <= metrics["mcc"] <= 1.0


def test_baseline_resamplers():
    feats = np.random.randn(80, 16).astype(np.float32)
    # Imbalanced labels: 50 for class 0, 30 for class 1
    labels = np.array([0] * 50 + [1] * 30, dtype=np.int64)

    feats_ros, labels_ros = apply_random_oversampling(feats, labels, random_state=42)
    assert len(labels_ros) == 100
    assert np.sum(labels_ros == 1) == 50

    feats_smote, labels_smote = apply_smote(feats, labels, random_state=42)
    assert len(labels_smote) == 100
    assert np.sum(labels_smote == 1) == 50

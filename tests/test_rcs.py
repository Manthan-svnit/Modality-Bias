import numpy as np
from multimodal_rcs.rcs import (
    sort_classes_by_imbalance,
    ClassGMMFitter,
    regularized_covariance_shrinkage,
    synthesize_minority_samples,
    MultimodalRCSBalancer,
    evaluate_prdc,
)


def test_class_sorting():
    # Counts: 500, 387, 300, 232, 180, 139, 108, 83, 65, 50
    class_counts = {
        0: 500, 1: 387, 2: 300, 3: 232, 4: 180,
        5: 139, 6: 108, 7: 83, 8: 65, 9: 50
    }
    # eta = 5 -> threshold = 500 / 5 = 100
    res = sort_classes_by_imbalance(class_counts, eta=5.0, majority_ratio=0.5)

    assert res["threshold"] == 100.0
    assert 9 in res["minority"]
    assert 8 in res["minority"]
    assert 7 in res["minority"]
    assert 0 in res["majority"]
    assert len(res["minority"]) == 3
    assert len(res["majority"]) + len(res["intermediate"]) + len(res["minority"]) == 10


def test_gmm_fitter():
    np.random.seed(42)
    # 2 classes: class 0 has 100 samples, class 1 has 50 samples
    x0 = np.random.randn(100, 16)
    x1 = np.random.randn(50, 16) + 3.0
    feats = np.vstack([x0, x1])
    labels = np.array([0] * 100 + [1] * 50)

    fitter = ClassGMMFitter(covariance_type="full", reg_covar=1e-3, random_state=42)
    fitter.fit(feats, labels)

    # Class 0 should have max(1, round(100/50)) = 2 components
    assert fitter.num_components_per_class[0] == 2
    # Class 1 should have max(1, round(50/50)) = 1 component
    assert fitter.num_components_per_class[1] == 1

    samples = fitter.sample_from_class(0, 10)
    assert samples.shape == (10, 16)


def test_rcs_balancer_and_prdc():
    np.random.seed(42)
    dim = 16
    # Synthetic imbalanced data across 5 classes
    counts = [100, 70, 40, 20, 10]
    all_feats = []
    all_labels = []
    for c, cnt in enumerate(counts):
        all_feats.append(np.random.randn(cnt, dim) + c * 2.0)
        all_labels.extend([c] * cnt)

    feats = np.vstack(all_feats)
    labels = np.array(all_labels)

    balancer = MultimodalRCSBalancer(eta=5.0, shrinkage_lambda=0.1, random_state=42)
    balanced_feats, balanced_labels, is_synthetic_mask, info = balancer.balance(feats, labels)

    # Balanced size should be majority size (100) per class = 500 samples
    assert len(balanced_feats) == 500
    assert len(balanced_labels) == 500
    unique, b_counts = np.unique(balanced_labels, return_counts=True)
    assert np.all(b_counts == 100)

    # Test PRDC calculation
    real_feats = balanced_feats[~is_synthetic_mask]
    fake_feats = balanced_feats[is_synthetic_mask]
    prdc = evaluate_prdc(real_feats, fake_feats, nearest_k=3)
    assert "precision" in prdc and "recall" in prdc and "density" in prdc and "coverage" in prdc
    assert 0.0 <= prdc["precision"] <= 1.0
    assert 0.0 <= prdc["recall"] <= 1.0

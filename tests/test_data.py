import pytest
import numpy as np
import torch
from multimodal_rcs.data import (
    AVMNISTDataset,
    create_imbalanced_counts,
    create_avmnist_splits,
)


def test_imbalanced_counts():
    counts_exp = create_imbalanced_counts(max_count=500, min_count=50, num_classes=10, mode="exponential")
    assert len(counts_exp) == 10
    assert counts_exp[0] == 500
    assert counts_exp[-1] == 50
    # Check monotonic decrease
    for i in range(len(counts_exp) - 1):
        assert counts_exp[i] >= counts_exp[i + 1]

    counts_lin = create_imbalanced_counts(max_count=500, min_count=50, num_classes=10, mode="linear")
    assert len(counts_lin) == 10
    assert counts_lin[0] == 500
    assert counts_lin[-1] == 50


def test_dataset_item_shapes():
    n_samples = 20
    images = np.random.randn(n_samples, 1, 28, 28).astype(np.float32)
    audios = np.random.randint(0, 256, (n_samples, 112, 112), dtype=np.uint8)
    labels = np.random.randint(0, 10, n_samples)

    ds = AVMNISTDataset(images=images, audios=audios, labels=labels)
    assert len(ds) == n_samples

    img, aud, lbl = ds[0]
    assert img.shape == (1, 28, 28)
    assert aud.shape == (1, 112, 112)
    assert isinstance(lbl.item(), int)
    assert img.dtype == torch.float32
    assert aud.dtype == torch.float32
    assert aud.max() <= 1.0 and aud.min() >= 0.0


def test_real_avmnist_splits():
    train_ds, val_ds, test_ds, counts = create_avmnist_splits(
        data_dir="data/avmnist",
        train_max_count=100,  # smaller for fast unit test
        train_min_count=10,
        val_samples_per_class=20,
        test_samples_per_class=50,
        random_seed=42,
    )
    assert len(val_ds) == 200  # 20 * 10
    assert len(test_ds) == 500 # 50 * 10
    # Verify strict class balance in validation set
    val_labels = val_ds.labels.numpy()
    unique, val_counts = np.unique(val_labels, return_counts=True)
    assert len(unique) == 10
    assert np.all(val_counts == 20)

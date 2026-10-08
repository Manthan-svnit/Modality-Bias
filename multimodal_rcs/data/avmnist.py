"""
AV-MNIST and generalized multimodal dataset loaders and splitters.
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def create_imbalanced_counts(
    max_count: int = 500,
    min_count: int = 50,
    num_classes: int = 10,
    mode: str = "exponential",
) -> List[int]:
    """Generates a class distribution with imbalance between max_count and min_count.

    Args:
        max_count: Number of samples in the most frequent class (class 0).
        min_count: Number of samples in the least frequent class (class num_classes - 1).
        num_classes: Total number of classes.
        mode: 'exponential' or 'linear'.

    Returns:
        List of integer counts per class, monotonically non-increasing.
    """
    if num_classes <= 1:
        return [max_count]

    if mode == "linear":
        counts = [
            int(round(max_count - i * (max_count - min_count) / (num_classes - 1)))
            for i in range(num_classes)
        ]
    elif mode == "exponential":
        ratio = min_count / max_count
        counts = [
            int(round(max_count * (ratio ** (i / (num_classes - 1)))))
            for i in range(num_classes)
        ]
    else:
        raise ValueError(f"Unknown imbalance mode: {mode}")

    # Ensure endpoints and monotonicity
    counts[0] = max_count
    counts[-1] = min_count
    for i in range(len(counts) - 1):
        if counts[i + 1] > counts[i]:
            counts[i + 1] = counts[i]

    return counts


class AVMNISTDataset(Dataset):
    """Dataset for Audio-Visual MNIST pairs.

    Stores:
        images: (N, C, 28, 28) float32 tensor
        audios: (N, 1, 112, 112) float32 tensor in [0, 1]
        labels: (N,) int64 tensor
    """

    def __init__(
        self,
        images: Union[np.ndarray, torch.Tensor],
        audios: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
    ):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if isinstance(audios, np.ndarray):
            audios = torch.from_numpy(audios)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        # Image processing: ensure (N, 1, 28, 28) and float32
        if images.dim() == 3:
            images = images.unsqueeze(1)
        self.images = images.float()

        # Audio processing: ensure (N, 1, 112, 112) float32 in [0, 1]
        if audios.dim() == 3:
            audios = audios.unsqueeze(1)
        audios = audios.float()
        if audios.max() > 1.0:
            audios = audios / 255.0
        self.audios = audios.clamp(0.0, 1.0)

        self.labels = labels.long()
        assert len(self.images) == len(self.audios) == len(self.labels), "Mismatched lengths"

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.images[idx], self.audios[idx], self.labels[idx]


class GenericMultimodalDataset(Dataset):
    """Arbitrary N-modality dataset supporting dictionary-based modalities."""

    def __init__(
        self,
        modalities: Dict[str, Union[np.ndarray, torch.Tensor]],
        labels: Union[np.ndarray, torch.Tensor],
    ):
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.modalities: Dict[str, torch.Tensor] = {}
        for mod, data in modalities.items():
            tensor = torch.as_tensor(data)
            if tensor.dtype == torch.float64:
                tensor = tensor.float()
            self.modalities[mod] = tensor
            assert len(tensor) == len(self.labels), f"Length mismatch for modality '{mod}'"

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        sample = {mod: tensor[idx] for mod, tensor in self.modalities.items()}
        return sample, self.labels[idx]


def _generate_synthetic_avmnist_data(
    num_samples_per_class: int,
    num_classes: int = 10,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_seed)
    total = num_samples_per_class * num_classes
    images = rng.randn(total, 1, 28, 28).astype(np.float32)
    audios = rng.randint(0, 256, (total, 1, 112, 112), dtype=np.uint8)
    labels = np.repeat(np.arange(num_classes), num_samples_per_class)
    perm = rng.permutation(total)
    return images[perm], audios[perm], labels[perm]


def create_avmnist_splits(
    data_dir: str = "data/avmnist",
    train_max_count: int = 500,
    train_min_count: int = 50,
    val_samples_per_class: int = 100,
    test_samples_per_class: int = 500,
    imbalance_mode: str = "exponential",
    num_classes: int = 10,
    random_seed: int = 42,
) -> Tuple[AVMNISTDataset, AVMNISTDataset, AVMNISTDataset, List[int]]:
    """Loads or generates AV-MNIST data splits according to class counts."""
    counts = create_imbalanced_counts(
        max_count=train_max_count,
        min_count=train_min_count,
        num_classes=num_classes,
        mode=imbalance_mode,
    )

    # Check if pre-extracted or raw numpy data exists in data_dir
    img_path = os.path.join(data_dir, "image.npy")
    aud_path = os.path.join(data_dir, "audio.npy")
    lbl_path = os.path.join(data_dir, "label.npy")

    if os.path.exists(img_path) and os.path.exists(aud_path) and os.path.exists(lbl_path):
        all_images = np.load(img_path)
        all_audios = np.load(aud_path)
        all_labels = np.load(lbl_path)
    else:
        # Fallback to realistic synthetic data generation for testing/mocking
        total_train = sum(counts)
        total_val = val_samples_per_class * num_classes
        total_test = test_samples_per_class * num_classes
        total_needed = max(total_train + total_val + total_test, 1000)
        per_class = int(np.ceil(total_needed / num_classes))
        all_images, all_audios, all_labels = _generate_synthetic_avmnist_data(
            num_samples_per_class=per_class,
            num_classes=num_classes,
            random_seed=random_seed,
        )

    rng = np.random.RandomState(random_seed)

    # Stratified split per class
    train_indices = []
    val_indices = []
    test_indices = []

    for c in range(num_classes):
        c_idx = np.where(all_labels == c)[0]
        rng.shuffle(c_idx)

        val_cnt = val_samples_per_class
        test_cnt = test_samples_per_class
        train_cnt = counts[c]

        v_idx = c_idx[:val_cnt]
        t_idx = c_idx[val_cnt : val_cnt + test_cnt]
        tr_idx = c_idx[val_cnt + test_cnt : val_cnt + test_cnt + train_cnt]

        # If data is short, wrap around
        if len(tr_idx) < train_cnt:
            extra = rng.choice(c_idx, size=(train_cnt - len(tr_idx)), replace=True)
            tr_idx = np.concatenate([tr_idx, extra])

        val_indices.extend(v_idx)
        test_indices.extend(t_idx)
        train_indices.extend(tr_idx)

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    train_ds = AVMNISTDataset(
        images=all_images[train_indices],
        audios=all_audios[train_indices],
        labels=all_labels[train_indices],
    )
    val_ds = AVMNISTDataset(
        images=all_images[val_indices],
        audios=all_audios[val_indices],
        labels=all_labels[val_indices],
    )
    test_ds = AVMNISTDataset(
        images=all_images[test_indices],
        audios=all_audios[test_indices],
        labels=all_labels[test_indices],
    )

    return train_ds, val_ds, test_ds, counts


def get_avmnist_dataloaders(
    data_dir: str = "data/avmnist",
    batch_size: int = 64,
    num_workers: int = 0,
    train_max_count: int = 500,
    train_min_count: int = 50,
    imbalance_mode: str = "exponential",
    val_samples_per_class: int = 100,
    test_samples_per_class: int = 500,
    num_classes: int = 10,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[int]]:
    """Returns PyTorch DataLoaders for train, val, test splits."""
    train_ds, val_ds, test_ds, counts = create_avmnist_splits(
        data_dir=data_dir,
        train_max_count=train_max_count,
        train_min_count=train_min_count,
        val_samples_per_class=val_samples_per_class,
        test_samples_per_class=test_samples_per_class,
        imbalance_mode=imbalance_mode,
        num_classes=num_classes,
        random_seed=random_seed,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, counts

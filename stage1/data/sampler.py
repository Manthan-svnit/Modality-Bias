"""
sampler.py — Class-balanced batch sampler for SupCon training.

Each batch is constructed by:
  1. Randomly selecting `num_classes_per_batch` classes.
  2. Sampling `samples_per_class` examples from each of those classes.

This guarantees that every batch has multiple examples per class,
which is the minimum requirement for SupCon loss computation.
"""

import math
import random
from typing import Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import Sampler


class ClassBalancedSampler(Sampler):
    """
    Yields batches such that each batch contains exactly
    ``num_classes_per_batch * samples_per_class`` samples.

    Parameters
    ----------
    labels : list of int
        Class label for every sample in the dataset (same order as the dataset).
    batch_size : int
        Total batch size.  Will be rounded down to
        ``num_classes_per_batch * samples_per_class``.
    num_classes_per_batch : int, optional
        How many distinct classes per batch.  Defaults to min(num_classes, 8).
    samples_per_class : int, optional
        Samples per class per batch.  Defaults to batch_size // num_classes_per_batch.
    drop_last : bool
        Whether to drop the last incomplete batch.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        labels: List[int],
        batch_size: int,
        num_classes_per_batch: Optional[int] = None,
        samples_per_class: Optional[int] = None,
        drop_last: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.labels = np.array(labels, dtype=np.int64)
        self.drop_last = drop_last
        self.seed = seed

        # Build per-class index lists
        unique_classes = np.unique(self.labels)
        self.num_classes = len(unique_classes)
        self.class_to_indices: dict = {
            int(c): np.where(self.labels == c)[0].tolist()
            for c in unique_classes
        }

        # Resolve num_classes_per_batch / samples_per_class
        if num_classes_per_batch is None:
            num_classes_per_batch = min(self.num_classes, max(2, batch_size // 8))
        if samples_per_class is None:
            samples_per_class = batch_size // num_classes_per_batch

        self.num_classes_per_batch = min(num_classes_per_batch, self.num_classes)
        self.samples_per_class = max(1, samples_per_class)
        self.effective_batch_size = self.num_classes_per_batch * self.samples_per_class

        # Number of batches per epoch
        # We aim to cover every sample at least once (approximately)
        total_samples = len(labels)
        if drop_last:
            self.num_batches = total_samples // self.effective_batch_size
        else:
            self.num_batches = math.ceil(total_samples / self.effective_batch_size)

        self.num_batches = max(1, self.num_batches)

    def __len__(self) -> int:
        return self.num_batches * self.effective_batch_size

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed)
        np_rng = np.random.default_rng(self.seed)

        # Shuffle per-class indices
        shuffled: dict = {
            c: np_rng.permutation(idxs).tolist()
            for c, idxs in self.class_to_indices.items()
        }
        pointers: dict = {c: 0 for c in shuffled}

        def _next_from_class(c: int) -> int:
            """Sample with replacement when exhausted."""
            idxs = shuffled[c]
            ptr = pointers[c]
            if ptr >= len(idxs):
                # Re-shuffle and restart
                shuffled[c] = np_rng.permutation(
                    self.class_to_indices[c]
                ).tolist()
                pointers[c] = 0
                ptr = 0
            idx = shuffled[c][ptr]
            pointers[c] += 1
            return idx

        all_classes = list(self.class_to_indices.keys())
        indices: List[int] = []

        for _ in range(self.num_batches):
            # Pick random subset of classes for this batch
            chosen_classes = rng.sample(all_classes, self.num_classes_per_batch)
            for c in chosen_classes:
                for _ in range(self.samples_per_class):
                    indices.append(_next_from_class(c))

        return iter(indices)


# ──────────────────────────────────────────────────────────────────────────────
# Factory helper
# ──────────────────────────────────────────────────────────────────────────────

def build_sampler(
    dataset,
    batch_size: int,
    use_balanced: bool,
    seed: int = 42,
) -> Optional[ClassBalancedSampler]:
    """
    Return a ``ClassBalancedSampler`` if ``use_balanced=True``, else ``None``.
    When ``None`` is returned, DataLoader will use its default sequential/random sampler.
    """
    if not use_balanced:
        return None

    labels = [int(dataset.df.iloc[i]["label"]) for i in range(len(dataset))]
    return ClassBalancedSampler(labels=labels, batch_size=batch_size, seed=seed)

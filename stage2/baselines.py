"""
baselines.py — Baseline oversampling methods that operate on z_fused.

Baselines used in Stage 3 comparison:
  1. NoCorrection     — use imbalanced real z_fused as-is
  2. RandomOversampler — duplicate minority samples until balanced
  3. SMOTEBalancer    — SMOTE-style linear interpolation on z_fused
"""

import numpy as np
from typing import Dict, Tuple


class NoCorrection:
    """Pass-through — real data only, no balancing."""

    def fit_transform(
        self, z: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        return z.copy(), y.copy(), {}


class RandomOversampler:
    """Duplicate minority samples with replacement until balanced."""

    def __init__(self, random_state: int = 42):
        self.rng = np.random.default_rng(random_state)

    def fit_transform(
        self, z: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        classes = np.unique(y)
        majority_size = max((y == c).sum() for c in classes)
        parts_z, parts_y = [z.copy()], [y.copy()]
        for c in classes:
            idx = np.where(y == c)[0]
            deficit = majority_size - len(idx)
            if deficit > 0:
                chosen = self.rng.choice(idx, deficit, replace=True)
                parts_z.append(z[chosen])
                parts_y.append(np.full(deficit, c, dtype=y.dtype))
        return np.concatenate(parts_z), np.concatenate(parts_y), {}


class SMOTEBalancer:
    """
    SMOTE-style interpolation on z_fused.

    For each minority point: pick a random k-nearest same-class neighbour,
    interpolate: z_new = z + rand(0,1) * (z_neighbour - z)
    """

    def __init__(self, k: int = 5, random_state: int = 42):
        self.k = k
        self.rng = np.random.default_rng(random_state)

    def fit_transform(
        self, z: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        from scipy.spatial.distance import cdist

        classes = np.unique(y)
        majority_size = max((y == c).sum() for c in classes)
        parts_z, parts_y = [z.copy()], [y.copy()]

        for c in classes:
            idx = np.where(y == c)[0]
            X_c = z[idx]
            deficit = majority_size - len(X_c)
            if deficit <= 0:
                continue

            k_use = min(self.k, len(X_c) - 1)
            if k_use < 1:
                # Too few points — just duplicate
                chosen = self.rng.choice(idx, deficit, replace=True)
                parts_z.append(z[chosen])
                parts_y.append(np.full(deficit, c, dtype=y.dtype))
                continue

            D = cdist(X_c, X_c, "euclidean")
            np.fill_diagonal(D, np.inf)
            nn = np.argsort(D, axis=1)[:, :k_use]

            synth = []
            while len(synth) < deficit:
                i = self.rng.integers(len(X_c))
                j = nn[i, self.rng.integers(k_use)]
                alpha = self.rng.random()
                synth.append(X_c[i] + alpha * (X_c[j] - X_c[i]))

            synth = np.array(synth[:deficit])
            parts_z.append(synth)
            parts_y.append(np.full(deficit, c, dtype=y.dtype))

        return np.concatenate(parts_z), np.concatenate(parts_y), {}

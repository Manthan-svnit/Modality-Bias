"""
rcs.py — RCS (Representation-space Class balancing via Synthesis)

Stage 2 algorithm — operates ONLY on the saved z_fused vectors from Stage 1.
No raw modality data is touched after Stage 1.

Algorithm:
----------
1. Threshold = majority_size + eta
   → sort each class into Majority / Intermediate / Minority buckets

2. Fit one GMM per class (sklearn or manual):
   n_components = max(1, round(class_size / smallest_class_size))

3. Intermediate classes:
   - Split the deficit evenly over GMM components
   - Sample from each component until class equals majority size

4. Minority classes:
   - For each real point in the minority class:
       a. Find k nearest GMM components from ALL Majority+Intermediate GMMs
       b. Weight each by (1 + component.sample_count)
       c. Blend: mean  = (1-λ)*real_point + λ*weighted_mean_of_k_components
                 cov   = (1-λ)*cov_minority + λ*weighted_cov_of_k_components
       d. Apply shrinkage: cov = (1-λ)*cov + λ*I
       e. Sample from this personalised Gaussian
   - Repeat until class equals majority size

5. PRDC quality check on synthetic vs real minority vectors

Output: real + synthetic z_fused  (balanced_z_fused.npy + balanced_labels.npy)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist


# ──────────────────────────────────────────────────────────────────────────────
# GMM component (manual, lightweight — no sklearn dependency required)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GMMComponent:
    mean: np.ndarray        # [d]
    cov: np.ndarray         # [d, d]
    sample_count: int       # number of real points assigned to this component
    weight: float           # mixture weight


# ──────────────────────────────────────────────────────────────────────────────
# GMM fitting (EM or k-means seeded)
# ──────────────────────────────────────────────────────────────────────────────

def fit_gmm(
    X: np.ndarray,              # [N, d]
    n_components: int,
    reg_covar: float = 1e-6,
    n_init: int = 3,
    random_state: int = 42,
) -> List[GMMComponent]:
    """
    Fit a GMM to the points X.

    Falls back to sklearn if available, otherwise uses k-means + covariance.

    Returns list of GMMComponent (one per component).
    """
    n = len(X)
    if n == 0:
        return []

    n_components = min(n_components, n)

    try:
        from sklearn.mixture import GaussianMixture
        gm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            reg_covar=reg_covar,
            n_init=n_init,
            random_state=random_state,
        )
        gm.fit(X)
        labels = gm.predict(X)

        components = []
        for k in range(n_components):
            mask = labels == k
            count = int(mask.sum())
            components.append(GMMComponent(
                mean=gm.means_[k],
                cov=gm.covariances_[k],
                sample_count=max(1, count),
                weight=float(gm.weights_[k]),
            ))
        return components

    except ImportError:
        # Fallback: k-means seeded empirical covariance
        return _fit_gmm_kmeans(X, n_components, reg_covar, random_state)


def _fit_gmm_kmeans(
    X: np.ndarray,
    n_components: int,
    reg_covar: float,
    random_state: int,
) -> List[GMMComponent]:
    """Lightweight GMM via k-means cluster assignments + empirical covariance."""
    rng = np.random.default_rng(random_state)
    n, d = X.shape

    # K-means++ initialisation
    centers = [X[rng.integers(n)]]
    for _ in range(1, n_components):
        dists = np.min(cdist(X, np.array(centers), "sqeuclidean"), axis=1)
        probs = dists / dists.sum()
        centers.append(X[rng.choice(n, p=probs)])

    centers = np.array(centers)

    for _ in range(100):  # k-means iterations
        labels = np.argmin(cdist(X, centers, "sqeuclidean"), axis=1)
        new_centers = np.array([
            X[labels == k].mean(0) if (labels == k).any() else centers[k]
            for k in range(n_components)
        ])
        if np.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers

    components = []
    for k in range(n_components):
        mask = labels == k
        pts = X[mask]
        count = len(pts)
        if count < 2:
            cov = np.eye(d) * reg_covar
        else:
            cov = np.cov(pts.T) + np.eye(d) * reg_covar
        components.append(GMMComponent(
            mean=centers[k],
            cov=cov,
            sample_count=max(1, count),
            weight=count / n,
        ))
    return components


# ──────────────────────────────────────────────────────────────────────────────
# Sample from a Gaussian
# ──────────────────────────────────────────────────────────────────────────────

def _sample_gaussian(
    mean: np.ndarray, cov: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample n points from N(mean, cov). Returns [n, d]."""
    d = len(mean)
    try:
        return rng.multivariate_normal(mean, cov, size=n)
    except np.linalg.LinAlgError:
        # Fallback: diagonal approximation
        return mean + rng.standard_normal((n, d)) * np.sqrt(np.diag(cov))


# ──────────────────────────────────────────────────────────────────────────────
# PRDC (Precision, Recall, Density, Coverage)
# ──────────────────────────────────────────────────────────────────────────────

def compute_prdc(
    real: np.ndarray,
    synthetic: np.ndarray,
    k: int = 5,
) -> Dict[str, float]:
    """
    Compute PRDC metrics between real and synthetic minority vectors.

    Parameters
    ----------
    real      : [N_r, d]
    synthetic : [N_s, d]
    k         : neighbourhood size

    Returns
    -------
    dict with keys: precision, recall, density, coverage
    """
    if len(real) == 0 or len(synthetic) == 0:
        return {"precision": 0.0, "recall": 0.0, "density": 0.0, "coverage": 0.0}

    def _kth_dist(X: np.ndarray, Y: np.ndarray, kk: int) -> np.ndarray:
        """For each point in X, return distance to its kk-th nearest in Y."""
        D = cdist(X, Y, "euclidean")
        D.sort(axis=1)
        kk = min(kk, D.shape[1] - 1)
        return D[:, kk]

    r_r = _kth_dist(real, real, k)          # [N_r]  k-th distance within real
    r_s = _kth_dist(synthetic, synthetic, k) # [N_s]  k-th distance within synthetic

    # Precision: fraction of synthetic in real manifold
    D_sr = cdist(synthetic, real, "euclidean")   # [N_s, N_r]
    precision = float((D_sr.min(1) < r_r[D_sr.argmin(1)]).mean())

    # Recall: fraction of real covered by the synthetic manifold
    # D_rs[i,j] = dist(real[i], synthetic[j]); argmin(1) -> nearest synthetic idx
    # r_s[j] = k-th radius of synthetic ball around synthetic[j]
    D_rs = cdist(real, synthetic, "euclidean")   # [N_r, N_s]
    recall = float((D_rs.min(1) < r_s[D_rs.argmin(1)]).mean())

    # Density: mean fraction of synthetic in each real ball
    density = float(
        np.mean([
            np.sum(D_sr[:, i] < r_r[i]) / k
            for i in range(len(real))
        ])
    )

    # Coverage: fraction of real manifold balls that contain at least one synthetic
    coverage = float(
        np.mean([
            np.any(D_sr[:, i] < r_r[i])
            for i in range(len(real))
        ])
    )

    return {
        "precision": precision,
        "recall": recall,
        "density": density,
        "coverage": coverage,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main RCS balancing engine
# ──────────────────────────────────────────────────────────────────────────────

class RCSBalancer:
    """
    RCS balancing for Stage 2.

    Parameters
    ----------
    eta : float
        Class-size threshold offset.
        Majority = classes with size > majority_size - eta
        (i.e., threshold = majority_size - eta)
    k : int
        Number of nearest GMM components to blend for minority synthesis.
    lam : float  (lambda)
        Shrinkage coefficient applied to covariance.
        final_cov = (1 - lam) * blended_cov + lam * I
    gmm_reg_covar : float
        Covariance regularisation term added to diagonal.
    gmm_n_init : int
        Number of GMM initialisations.
    prdc_k : int
        k for PRDC neighbour computation.
    random_state : int
    """

    def __init__(
        self,
        eta: float = 0.5,
        k: int = 5,
        lam: float = 0.5,
        gmm_reg_covar: float = 1e-6,
        gmm_n_init: int = 3,
        prdc_k: int = 5,
        random_state: int = 42,
    ):
        self.eta = eta
        self.k = k
        self.lam = lam
        self.gmm_reg_covar = gmm_reg_covar
        self.gmm_n_init = gmm_n_init
        self.prdc_k = prdc_k
        self.rng = np.random.default_rng(random_state)

        # Set after fit()
        self.class_labels_: Optional[List[int]] = None
        self.majority_size_: Optional[int] = None
        self.class_types_: Optional[Dict[int, str]] = None  # label → "majority/intermediate/minority"
        self.gmm_components_: Optional[Dict[int, List[GMMComponent]]] = None  # label → components
        self.prdc_scores_: Optional[Dict[int, Dict[str, float]]] = None

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, z: np.ndarray, y: np.ndarray) -> "RCSBalancer":
        """
        Fit GMMs on training z_fused and classify classes.

        Parameters
        ----------
        z : [N_train, d]
        y : [N_train]  integer class labels

        Returns
        -------
        self
        """
        classes = sorted(np.unique(y).tolist())
        class_sizes = {c: int((y == c).sum()) for c in classes}
        smallest = min(class_sizes.values())
        majority = max(class_sizes.values())
        self.majority_size_ = majority
        self.class_labels_ = classes

        # ── Sort into types ──────────────────────────────────────────────────
        threshold = majority - self.eta * (majority - smallest)
        self.class_types_ = {}
        for c, sz in class_sizes.items():
            if sz >= threshold:
                self.class_types_[c] = "majority"
            elif sz > smallest:
                self.class_types_[c] = "intermediate"
            else:
                self.class_types_[c] = "minority"

        # ── Fit GMM per class ─────────────────────────────────────────────────
        self.gmm_components_ = {}
        for c in classes:
            X_c = z[y == c]
            n_comp = max(1, round(class_sizes[c] / max(1, smallest)))
            self.gmm_components_[c] = fit_gmm(
                X_c, n_comp, self.gmm_reg_covar, self.gmm_n_init
            )

        return self

    # ── transform ─────────────────────────────────────────────────────────────

    def transform(
        self, z: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate synthetic samples and return balanced dataset.

        Parameters
        ----------
        z : [N_train, d]
        y : [N_train]  labels

        Returns
        -------
        z_bal  : [N_bal, d]   real + synthetic
        y_bal  : [N_bal]      labels
        report : dict         PRDC and synthesis counts per class
        """
        if self.gmm_components_ is None:
            raise RuntimeError("Call fit() before transform().")

        d = z.shape[1]
        majority_size = self.majority_size_
        synth_z_list, synth_y_list = [], []
        report: Dict = {"class_types": self.class_types_, "prdc": {}}

        # All Majority + Intermediate GMM components (for minority synthesis)
        ref_components: List[GMMComponent] = []
        for c, ctype in self.class_types_.items():
            if ctype in ("majority", "intermediate"):
                ref_components.extend(self.gmm_components_[c])

        for c in self.class_labels_:
            X_c = z[y == c]
            deficit = majority_size - len(X_c)
            if deficit <= 0:
                continue

            ctype = self.class_types_[c]
            components_c = self.gmm_components_[c]

            if ctype == "intermediate":
                synth = self._sample_intermediate(components_c, deficit, d)
            else:  # minority
                synth = self._sample_minority(X_c, components_c, ref_components, deficit, d)

            synth_z_list.append(synth)
            synth_y_list.append(np.full(len(synth), c, dtype=y.dtype))

            # PRDC check
            if len(synth) > 0 and len(X_c) > 0:
                report["prdc"][c] = compute_prdc(X_c, synth, k=self.prdc_k)

        # Concatenate
        z_bal = np.concatenate([z] + synth_z_list, axis=0) if synth_z_list else z.copy()
        y_bal = np.concatenate([y] + synth_y_list, axis=0) if synth_y_list else y.copy()

        self.prdc_scores_ = report["prdc"]
        return z_bal, y_bal, report

    def fit_transform(
        self, z: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        return self.fit(z, y).transform(z, y)

    # ── internal synthesis methods ─────────────────────────────────────────────

    def _sample_intermediate(
        self,
        components: List[GMMComponent],
        deficit: int,
        d: int,
    ) -> np.ndarray:
        """Sample deficit points from the class GMM, split across components."""
        n_comp = len(components)
        counts = self._split_evenly(deficit, n_comp)
        parts = []
        for comp, cnt in zip(components, counts):
            if cnt > 0:
                parts.append(_sample_gaussian(comp.mean, comp.cov, cnt, self.rng))
        return np.concatenate(parts, axis=0) if parts else np.zeros((0, d))

    def _sample_minority(
        self,
        X_c: np.ndarray,
        components_c: List[GMMComponent],
        ref_components: List[GMMComponent],
        deficit: int,
        d: int,
    ) -> np.ndarray:
        """
        Minority synthesis:
        For each real point → find k nearest ref components →
        blend with class cov → shrinkage → sample.
        """
        if len(ref_components) == 0 or len(X_c) == 0:
            # Fallback: sample from own GMM
            return self._sample_intermediate(components_c, deficit, d)

        # Class-level covariance (fallback if one component)
        if len(X_c) >= 2:
            cov_c = np.cov(X_c.T) + np.eye(d) * self.gmm_reg_covar
        else:
            cov_c = np.eye(d) * self.gmm_reg_covar

        # Stack ref component means for distance computation
        ref_means = np.array([comp.mean for comp in ref_components])  # [K_ref, d]
        ref_weights = np.array([1.0 + comp.sample_count for comp in ref_components])

        synth = []
        k_use = min(self.k, len(ref_components))

        for real_pt in X_c:
            if len(synth) >= deficit:
                break

            # k nearest ref components
            dists = np.linalg.norm(ref_means - real_pt, axis=1)   # [K_ref]
            nn_idx = np.argsort(dists)[:k_use]

            # Weighted mean of k components
            w = ref_weights[nn_idx]
            w = w / w.sum()

            blend_mean = np.sum([
                w[i] * ref_components[nn_idx[i]].mean
                for i in range(k_use)
            ], axis=0)

            blend_cov = np.sum([
                w[i] * ref_components[nn_idx[i]].cov
                for i in range(k_use)
            ], axis=0)

            # Interpolate with real point and class cov
            lam = self.lam
            final_mean = (1 - lam) * real_pt + lam * blend_mean
            final_cov  = (1 - lam) * cov_c  + lam * blend_cov
            # Shrinkage
            final_cov = (1 - lam) * final_cov + lam * np.eye(d)

            synth.append(_sample_gaussian(final_mean, final_cov, 1, self.rng))

        # If we still need more, loop
        synth_arr = np.concatenate(synth, axis=0) if synth else np.zeros((0, d))
        while len(synth_arr) < deficit:
            extra = self._sample_intermediate(components_c, deficit - len(synth_arr), d)
            synth_arr = np.concatenate([synth_arr, extra], axis=0)

        return synth_arr[:deficit]

    @staticmethod
    def _split_evenly(total: int, n: int) -> List[int]:
        """Split total into n parts as evenly as possible."""
        base, rem = divmod(total, n)
        return [base + (1 if i < rem else 0) for i in range(n)]

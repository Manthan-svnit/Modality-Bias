from typing import List, Dict, Tuple
import numpy as np
import logging
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


class ComponentInfo:
    """Holds information about a single Gaussian component from a fitted GMM."""

    def __init__(
        self,
        class_id: int,
        comp_idx: int,
        mean: np.ndarray,
        cov: np.ndarray,
        sample_count: float,
    ):
        self.class_id = class_id
        self.comp_idx = comp_idx
        self.mean = mean
        self.cov = cov
        self.sample_count = float(sample_count)


def collect_donor_components(
    fitted_gmms: Dict[int, GaussianMixture],
    donor_class_ids: List[int],
    class_counts: Dict[int, int],
) -> List[ComponentInfo]:
    """Gathers all GMM components from majority and intermediate classes."""
    components = []
    for c in donor_class_ids:
        if c not in fitted_gmms:
            continue
        gmm = fitted_gmms[c]
        c_count = class_counts[c]
        for k in range(gmm.n_components):
            # Effective sample count for component k
            comp_count = max(1.0, gmm.weights_[k] * c_count)
            cov = gmm.covariances_[k]
            # If covariance is diagonal or spherical, convert to full (D, D)
            if cov.ndim == 1:
                cov = np.diag(cov)
            comp_info = ComponentInfo(
                class_id=c,
                comp_idx=k,
                mean=gmm.means_[k],
                cov=cov,
                sample_count=comp_count,
            )
            components.append(comp_info)
    return components


def regularized_covariance_shrinkage(
    cov: np.ndarray,
    shrinkage_lambda: float = 0.1,
) -> np.ndarray:
    """Applies Ledoit-Wolf style shrinkage:
    Sigma_reg = (1 - lambda) * Cov + lambda * Identity
    """
    dim = cov.shape[0]
    identity = np.eye(dim, dtype=cov.dtype)
    cov_shrunk = (1.0 - shrinkage_lambda) * cov + shrinkage_lambda * identity

    # Ensure symmetric and positive semi-definite
    cov_shrunk = 0.5 * (cov_shrunk + cov_shrunk.T)
    # Add minimal jitter to guarantee strict positive definiteness
    min_eig = np.min(np.linalg.eigvalsh(cov_shrunk))
    if min_eig < 1e-5:
        cov_shrunk += (1e-5 - min_eig) * identity

    return cov_shrunk


def synthesize_minority_samples(
    minority_samples: np.ndarray,
    donor_components: List[ComponentInfo],
    n_samples_to_generate: int,
    top_k: int = 3,
    shrinkage_lambda: float = 0.1,
    center_on_sample: bool = True,
    blend_weight: float = 0.0,
    random_seed: int = 42,
) -> np.ndarray:
    """Generates synthetic samples for a minority class using Regularized Covariance Sharing.

    For each synthetic sample:
      1. Pick a minority anchor sample x_i
      2. Identify the top_k (3) nearest components from majority/intermediate GMMs
      3. Weight components inversely by sample count (1 / sample_count)
      4. Blend their mean and covariance
      5. Apply shrinkage: (1 - lambda) * Cov + lambda * Identity (lambda=0.1)
      6. Draw synthetic sample from N(anchor, Sigma_reg)

    Args:
        minority_samples: Array of shape (N_minority, D)
        donor_components: List of ComponentInfo from majority/intermediate classes
        n_samples_to_generate: Number of synthetic vectors to create
        top_k: Number of nearest components to blend (default: 3)
        shrinkage_lambda: Shrinkage parameter lambda (default: 0.1)
        center_on_sample: Anchor distribution at minority sample x_i (preserves class semantics)
        blend_weight: If > 0, blends sample x_i with blended component mean mu_blend
        random_seed: Seed for reproducibility

    Returns:
        Array of synthetic samples of shape (n_samples_to_generate, D)
    """
    rng = np.random.default_rng(random_seed)
    n_minority, dim = minority_samples.shape

    if len(donor_components) < top_k:
        top_k = len(donor_components)

    # Pre-extract donor means matrix (M, D)
    donor_means = np.array([c.mean for c in donor_components])  # (M, D)
    donor_counts = np.array([c.sample_count for c in donor_components])  # (M,)

    synthetic_samples = []

    for idx in range(n_samples_to_generate):
        # 1. Select anchor sample from minority class
        anchor_idx = idx % n_minority
        x_i = minority_samples[anchor_idx]

        # 2. Identify 3 nearest components by Euclidean distance
        dists = np.linalg.norm(donor_means - x_i, axis=1)
        nearest_indices = np.argsort(dists)[:top_k]

        # 3. Weight inversely by sample count (1 / sample_count)
        inv_counts = 1.0 / (donor_counts[nearest_indices] + 1e-8)
        weights = inv_counts / np.sum(inv_counts)

        # 4. Blend mean and covariance
        mu_blend = np.zeros(dim)
        cov_blend = np.zeros((dim, dim))
        for w, comp_idx in zip(weights, nearest_indices):
            comp = donor_components[comp_idx]
            mu_blend += w * comp.mean
            cov_blend += w * comp.cov

        # 5. Apply regularized covariance shrinkage ((1 - lambda) * Cov + lambda * Identity)
        cov_reg = regularized_covariance_shrinkage(
            cov_blend, shrinkage_lambda=shrinkage_lambda
        )

        # Distribution center
        if center_on_sample:
            target_mean = (1.0 - blend_weight) * x_i + blend_weight * mu_blend
        else:
            target_mean = mu_blend

        # 6. Draw synthetic sample
        sample = rng.multivariate_normal(target_mean, cov_reg)
        synthetic_samples.append(sample)

    return np.array(synthetic_samples, dtype=np.float32)

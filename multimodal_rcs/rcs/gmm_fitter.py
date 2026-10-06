from typing import Dict, Any
import numpy as np
import logging
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


class ClassGMMFitter:
    """Fits Gaussian Mixture Models (GMM) per class according to:
    num_components = max(1, round(class_size / smallest_class_size)).
    """

    def __init__(
        self,
        covariance_type: str = "full",
        reg_covar: float = 1e-3,
        random_state: int = 42,
    ):
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.fitted_gmms: Dict[int, GaussianMixture] = {}
        self.num_components_per_class: Dict[int, int] = {}

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "ClassGMMFitter":
        """Fits a GMM for every unique class in labels.

        Args:
            features: Array of shape (N, D)
            labels: Array of shape (N,)
        """
        unique_classes = np.unique(labels)
        class_sizes = {int(c): int(np.sum(labels == c)) for c in unique_classes}
        smallest_class_size = min(class_sizes.values())

        logger.info(f"Fitting per-class GMMs with smallest_class_size = {smallest_class_size}:")

        for c in sorted(unique_classes):
            c_int = int(c)
            c_size = class_sizes[c_int]
            k_comp = max(1, int(round(c_size / smallest_class_size)))
            # Cannot have more components than samples
            k_comp = min(k_comp, max(1, c_size // 2))

            self.num_components_per_class[c_int] = k_comp
            X_c = features[labels == c]

            gmm = GaussianMixture(
                n_components=k_comp,
                covariance_type=self.covariance_type,
                reg_covar=self.reg_covar,
                random_state=self.random_state,
                max_iter=150,
            )
            gmm.fit(X_c)
            self.fitted_gmms[c_int] = gmm

            logger.info(
                f"  Class {c_int}: {c_size} samples -> {k_comp} GMM component(s) "
                f"(converged={gmm.converged_})"
            )

        return self

    def sample_from_class(self, class_id: int, n_samples: int) -> np.ndarray:
        """Draws n_samples from the fitted GMM of class_id with robust PSD covariance."""
        if class_id not in self.fitted_gmms:
            raise KeyError(f"GMM for class {class_id} not fitted.")
        gmm = self.fitted_gmms[class_id]
        
        # Enforce strict symmetry and PSD on each component covariance before sampling
        dim = gmm.means_.shape[1]
        for k in range(gmm.n_components):
            cov = gmm.covariances_[k]
            if cov.ndim == 1:
                cov = np.diag(cov)
            cov = 0.5 * (cov + cov.T)
            min_eig = np.min(np.linalg.eigvalsh(cov))
            if min_eig < 1e-4:
                cov += (1e-4 - min_eig) * np.eye(dim)
            gmm.covariances_[k] = cov

        samples, _ = gmm.sample(n_samples)
        return samples.astype(np.float32)

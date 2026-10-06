from typing import Dict, Tuple, Optional, Any
import numpy as np
import logging
from .class_sorter import sort_classes_by_imbalance
from .gmm_fitter import ClassGMMFitter
from .covariance_sharing import collect_donor_components, synthesize_minority_samples

logger = logging.getLogger(__name__)


class MultimodalRCSBalancer:
    """Multimodal Regularized Covariance Sharing (RCS) Balancer.
    
    Orchestrates:
      1. Class sorting into Majority, Intermediate, Minority using threshold = majority_size / eta
      2. Per-class GMM fitting with num_components = max(1, round(class_size / smallest_class_size))
      3. GMM sampling for intermediate classes up to majority_size
      4. Covariance sharing and regularized shrinkage for minority classes up to majority_size
    """

    def __init__(
        self,
        eta: float = 5.0,
        shrinkage_lambda: float = 0.1,
        top_k: int = 3,
        random_state: int = 42,
    ):
        self.eta = eta
        self.shrinkage_lambda = shrinkage_lambda
        self.top_k = top_k
        self.random_state = random_state

        self.class_partitions: Optional[Dict[str, Any]] = None
        self.gmm_fitter: Optional[ClassGMMFitter] = None

    def balance(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Balances the dataset using Multimodal RCS.

        Args:
            features: Real features array of shape (N, D)
            labels: 1D labels array of shape (N,)

        Returns:
            Tuple of:
                - balanced_features: (N_balanced, D)
                - balanced_labels: (N_balanced,)
                - is_synthetic_mask: (N_balanced,) bool array (True for synthetic samples)
                - info_dict: Metadata regarding synthesis and partitions
        """
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)

        unique_classes = np.unique(labels)
        class_counts = {int(c): int(np.sum(labels == c)) for c in unique_classes}
        majority_size = max(class_counts.values())

        # 1. Partition classes into Majority, Intermediate, Minority
        self.class_partitions = sort_classes_by_imbalance(
            class_counts=class_counts, eta=self.eta
        )
        majority_classes = self.class_partitions["majority"]
        intermediate_classes = self.class_partitions["intermediate"]
        minority_classes = self.class_partitions["minority"]

        # 2. Fit per-class GMMs
        self.gmm_fitter = ClassGMMFitter(
            covariance_type="full",
            reg_covar=1e-3,
            random_state=self.random_state,
        )
        self.gmm_fitter.fit(features, labels)

        # 3. Gather donor components from Majority and Intermediate classes
        donor_classes = list(set(majority_classes + intermediate_classes))
        donor_components = collect_donor_components(
            fitted_gmms=self.gmm_fitter.fitted_gmms,
            donor_class_ids=donor_classes,
            class_counts=class_counts,
        )
        logger.info(
            f"Collected {len(donor_components)} donor components from classes {donor_classes} "
            f"for covariance sharing."
        )

        all_balanced_feats = []
        all_balanced_labels = []
        is_synthetic = []

        synthetic_by_class: Dict[int, np.ndarray] = {}

        # 4. Process each class
        for c in sorted(unique_classes):
            c_int = int(c)
            real_c_feats = features[labels == c]
            current_count = len(real_c_feats)
            needed_samples = majority_size - current_count

            # Keep all real samples
            all_balanced_feats.append(real_c_feats)
            all_balanced_labels.append(np.full(current_count, c_int, dtype=np.int64))
            is_synthetic.append(np.zeros(current_count, dtype=bool))

            if needed_samples <= 0:
                continue

            if c_int in intermediate_classes:
                # Intermediate class: Sample from fitted class GMM
                logger.info(
                    f"Class {c_int} (Intermediate): Drawing {needed_samples} samples from fitted class GMM."
                )
                synth_feats = self.gmm_fitter.sample_from_class(
                    class_id=c_int, n_samples=needed_samples
                )
            elif c_int in minority_classes:
                # Minority class: Regularized Covariance Sharing from nearest donor components
                logger.info(
                    f"Class {c_int} (Minority): Generating {needed_samples} samples via Regularized Covariance Sharing."
                )
                synth_feats = synthesize_minority_samples(
                    minority_samples=real_c_feats,
                    donor_components=donor_components,
                    n_samples_to_generate=needed_samples,
                    top_k=self.top_k,
                    shrinkage_lambda=self.shrinkage_lambda,
                    center_on_sample=True,
                    random_seed=self.random_state + c_int,
                )
            else:
                # Majority class (if slightly below max count due to slight variance)
                logger.info(
                    f"Class {c_int} (Majority): Sampling {needed_samples} from GMM to reach exact balance."
                )
                synth_feats = self.gmm_fitter.sample_from_class(
                    class_id=c_int, n_samples=needed_samples
                )

            all_balanced_feats.append(synth_feats)
            all_balanced_labels.append(np.full(needed_samples, c_int, dtype=np.int64))
            is_synthetic.append(np.ones(needed_samples, dtype=bool))
            synthetic_by_class[c_int] = synth_feats

        balanced_features = np.vstack(all_balanced_feats)
        balanced_labels = np.concatenate(all_balanced_labels)
        is_synthetic_mask = np.concatenate(is_synthetic)

        # Shuffle together
        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(len(balanced_features))
        balanced_features = balanced_features[perm]
        balanced_labels = balanced_labels[perm]
        is_synthetic_mask = is_synthetic_mask[perm]

        info_dict = {
            "partitions": self.class_partitions,
            "synthetic_by_class": synthetic_by_class,
            "original_counts": class_counts,
            "balanced_size_per_class": majority_size,
            "total_samples": len(balanced_features),
            "num_synthetic": int(np.sum(is_synthetic_mask)),
        }

        logger.info(
            f"RCS Balancing completed: {len(features)} original samples -> "
            f"{len(balanced_features)} balanced samples ({info_dict['num_synthetic']} synthetic)."
        )

        return balanced_features, balanced_labels, is_synthetic_mask, info_dict

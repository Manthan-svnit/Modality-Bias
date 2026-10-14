"""
stage3 — Final Classifier and Evaluation.

Trains the final MLP classifier on balanced representation space, compares
variants A/B/C/D and baselines across multiple random seeds, and outputs
comprehensive imbalanced metrics (Balanced Acc, Macro F1, MCC, G-Mean, minority recall).
"""

from stage3.models import MLPClassifier
from stage3.metrics import (
    compute_classification_metrics,
    compute_g_mean,
    compute_minority_recall,
)
from stage3.evaluator import (
    train_single_run,
    evaluate_model,
    run_multi_seed_evaluation,
)

__all__ = [
    "MLPClassifier",
    "compute_classification_metrics",
    "compute_g_mean",
    "compute_minority_recall",
    "train_single_run",
    "evaluate_model",
    "run_multi_seed_evaluation",
]

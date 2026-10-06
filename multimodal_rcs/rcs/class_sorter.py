from typing import Dict, List, Literal
import numpy as np
import logging

logger = logging.getLogger(__name__)


def sort_classes_by_imbalance(
    class_counts: Dict[int, int],
    eta: float = 5.0,
    majority_ratio: float = 0.5,
) -> Dict[str, List[int]]:
    """Sorts classes into Majority, Intermediate, and Minority.

    Threshold formula:
        threshold = (majority size) / eta
    
    Rule:
        - Minority: class_size < threshold
        - Majority: class_size >= majority_size * majority_ratio
        - Intermediate: threshold <= class_size < majority_size * majority_ratio

    Args:
        class_counts: Dictionary mapping class_id -> count of samples
        eta: Threshold scaling parameter (default: 5.0)
        majority_ratio: Ratio of majority_size to consider as majority (default: 0.5)

    Returns:
        Dict with keys 'majority', 'intermediate', 'minority' mapping to sorted lists of class IDs.
    """
    majority_size = max(class_counts.values())
    threshold = majority_size / eta

    majority_classes = []
    intermediate_classes = []
    minority_classes = []

    for c, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        if count < threshold:
            minority_classes.append(c)
        elif count >= majority_size * majority_ratio:
            majority_classes.append(c)
        else:
            intermediate_classes.append(c)

    # Ensure at least 1 majority class
    if not majority_classes:
        top_class = max(class_counts.items(), key=lambda x: x[1])[0]
        majority_classes.append(top_class)
        if top_class in intermediate_classes:
            intermediate_classes.remove(top_class)

    logger.info(
        f"Class partition with eta={eta} (Minority Threshold = {threshold:.1f}, "
        f"Majority Threshold = {majority_size * majority_ratio:.1f}):"
    )
    logger.info(f"  Majority classes:     {majority_classes} -> counts: {[class_counts[c] for c in majority_classes]}")
    logger.info(f"  Intermediate classes: {intermediate_classes} -> counts: {[class_counts[c] for c in intermediate_classes]}")
    logger.info(f"  Minority classes:     {minority_classes} -> counts: {[class_counts[c] for c in minority_classes]}")

    return {
        "majority": majority_classes,
        "intermediate": intermediate_classes,
        "minority": minority_classes,
        "threshold": threshold,
        "majority_size": majority_size,
    }

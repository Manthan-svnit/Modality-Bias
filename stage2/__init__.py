"""
stage2 — Representation-space Class balancing via Synthesis (RCS).

Operates ONLY on saved z_fused representations from Stage 1.
No raw multimodal data is touched in Stage 2.
"""

from stage2.rcs import RCSBalancer, compute_prdc, GMMComponent
from stage2.baselines import NoCorrection, RandomOversampler, SMOTEBalancer
from stage2.optuna_stage_b import OptunaStageB

__all__ = [
    "RCSBalancer",
    "compute_prdc",
    "GMMComponent",
    "NoCorrection",
    "RandomOversampler",
    "SMOTEBalancer",
    "OptunaStageB",
]

from dataclasses import dataclass, field
from typing import Literal, List
import argparse


@dataclass
class PipelineConfig:
    # Environment & Hardware
    device: str = "auto"
    random_seed: int = 42

    # Dataset & Imbalance Setup
    data_dir: str = "data/avmnist"
    batch_size: int = 64
    num_workers: int = 0
    train_max_count: int = 500
    train_min_count: int = 50
    imbalance_mode: str = "exponential"  # 'exponential' or 'linear'
    val_samples_per_class: int = 100
    test_samples_per_class: int = 500

    # Stage 1 Architecture & Training
    fusion_variant: Literal["early", "late"] = "early"
    embed_dim: int = 128
    num_classes: int = 10
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    stage1_lr: float = 1e-4
    stage1_weight_decay: float = 1e-4
    stage1_epochs: int = 100

    # Multi-Task Loss Weights & Temperatures
    weight_recon: float = 1.0
    weight_cls: float = 1.0
    weight_supcon: float = 0.1
    weight_cmm: float = 0.1
    temperature: float = 0.07

    # Stage 2 RCS Configuration
    eta: float = 5.0
    shrinkage_lambda: float = 0.1
    top_k_components: int = 3

    # Stage 3 Classifier Architecture & Training
    clf_hidden1: int = 256
    clf_hidden2: int = 128
    clf_dropout: float = 0.1
    clf_lr: float = 1e-3
    clf_weight_decay: float = 1e-4
    clf_epochs: int = 100
    clf_batch_size: int = 64

    # Output Paths
    checkpoint_dir: str = "checkpoints"
    features_dir: str = "extracted_features"
    results_dir: str = "results"
    figures_dir: str = "results/figures"


def add_pipeline_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds command-line arguments corresponding to PipelineConfig."""
    parser.add_argument("--data_dir", type=str, default="data/avmnist", help="Path to AV-MNIST dataset directory")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--fusion_variant", type=str, choices=["early", "late"], default="early", help="Fusion variant")
    parser.add_argument("--stage1_epochs", type=int, default=100, help="Number of training epochs for Stage 1")
    parser.add_argument("--stage1_lr", type=float, default=1e-4, help="Learning rate for Stage 1")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--eta", type=float, default=5.0, help="RCS minority threshold divisor (majority / eta)")
    parser.add_argument("--shrinkage_lambda", type=float, default=0.1, help="RCS covariance shrinkage parameter")
    parser.add_argument("--clf_epochs", type=int, default=100, help="Classifier training epochs for Stage 3")
    parser.add_argument("--clf_lr", type=float, default=1e-3, help="Classifier learning rate for Stage 3")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    return parser

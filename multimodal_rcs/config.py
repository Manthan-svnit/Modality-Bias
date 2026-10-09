"""
Generalized Pipeline Configuration.

Dataset-agnostic: all dataset-specific paths and parameters are configurable
via dataclass fields or CLI arguments. The 'dataset_name' field is used to
automatically namespace all output directories (checkpoints, features, results).

Active modalities are configurable so the same config can drive a 2-modality
(image+audio) or 3-modality (image+audio+text) run on any dataset.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, List
import os
import argparse


@dataclass
class PipelineConfig:
    # ------------------------------------------------------------------ #
    # Dataset Identity
    # ------------------------------------------------------------------ #
    dataset_name: str = "avmnist"
    """Short identifier used to namespace output directories, e.g. 'avmnist', 'hateful_memes'."""

    data_dir: str = "data/avmnist"
    """Root directory of the dataset."""

    # Active modalities: any non-empty subset of ["image", "audio", "text"]
    active_modalities: List[str] = field(default_factory=lambda: ["image", "audio"])
    """Which modalities to use. E.g. ["image","audio"], ["image","text"], or all three."""

    # ------------------------------------------------------------------ #
    # Hardware & Reproducibility
    # ------------------------------------------------------------------ #
    device: str = "auto"
    random_seed: int = 42

    # ------------------------------------------------------------------ #
    # DataLoader
    # ------------------------------------------------------------------ #
    batch_size: int = 64
    num_workers: int = 0

    # ------------------------------------------------------------------ #
    # Class Imbalance Setup
    # ------------------------------------------------------------------ #
    num_classes: int = 10
    train_max_count: int = 500
    train_min_count: int = 50
    imbalance_mode: str = "exponential"   # 'exponential' or 'linear'
    val_samples_per_class: int = 100
    test_samples_per_class: int = 500

    # ------------------------------------------------------------------ #
    # Modality-Specific Encoder Config
    # ------------------------------------------------------------------ #
    # Image
    image_in_channels: int = 1

    # Audio
    audio_in_channels: int = 1

    # Text
    text_vocab_size: int = 30000
    text_max_seq_len: int = 128
    text_ff_dim: int = 256
    text_padding_idx: int = 0

    # Video
    video_in_channels: int = 3
    video_recon_frames: int = 8
    video_recon_height: int = 28
    video_recon_width: int = 28

    # ------------------------------------------------------------------ #
    # Stage 1 Architecture & Training
    # ------------------------------------------------------------------ #
    fusion_variant: Literal["early", "late"] = "early"
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    stage1_lr: float = 1e-4
    stage1_weight_decay: float = 1e-4
    stage1_epochs: int = 100
    stage1_patience: int = 15
    use_lr_scheduler: bool = True

    # ------------------------------------------------------------------ #
    # Multi-Task Loss Weights & Temperatures
    # ------------------------------------------------------------------ #
    weight_recon: float = 1.0
    weight_cls: float = 1.0
    weight_supcon: float = 0.1
    weight_cmm: float = 0.1
    temperature: float = 0.07

    # ------------------------------------------------------------------ #
    # Stage 2: RCS Configuration
    # ------------------------------------------------------------------ #
    eta: float = 5.0
    shrinkage_lambda: float = 0.1
    top_k_components: int = 3

    # ------------------------------------------------------------------ #
    # Stage 3: Classifier Architecture & Training
    # ------------------------------------------------------------------ #
    clf_hidden1: int = 256
    clf_hidden2: int = 128
    clf_dropout: float = 0.1
    clf_lr: float = 1e-3
    clf_weight_decay: float = 1e-4
    clf_epochs: int = 100
    clf_batch_size: int = 64
    clf_patience: int = 20

    # ------------------------------------------------------------------ #
    # Output Paths (auto-namespaced by dataset_name, populated in __post_init__)
    # ------------------------------------------------------------------ #
    checkpoint_dir: str = ""
    features_dir: str = ""
    results_dir: str = ""
    figures_dir: str = ""

    def __post_init__(self):
        """Auto-populate output paths under results/{dataset_name}/ if not explicitly set."""
        base = os.path.join("results", self.dataset_name)
        if not self.checkpoint_dir:
            self.checkpoint_dir = os.path.join("checkpoints", self.dataset_name)
        if not self.features_dir:
            self.features_dir = os.path.join("extracted_features", self.dataset_name)
        if not self.results_dir:
            self.results_dir = base
        if not self.figures_dir:
            self.figures_dir = os.path.join(base, "figures")

    @property
    def modality_tag(self) -> str:
        """Short tag describing active modalities, e.g. 'img_aud_txt' or 'img_vid'."""
        abbrev = {"image": "img", "audio": "aud", "text": "txt", "video": "vid"}
        return "_".join(abbrev.get(m, m) for m in self.active_modalities)


def add_pipeline_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds command-line arguments corresponding to PipelineConfig."""
    parser.add_argument("--dataset_name", type=str, default="avmnist",
                        help="Short dataset identifier used to namespace outputs")
    parser.add_argument("--data_dir", type=str, default="data/avmnist",
                        help="Path to dataset root directory")
    parser.add_argument("--active_modalities", type=str, nargs="+",
                        default=["image", "audio"],
                        choices=["image", "audio", "text", "video"],
                        help="List of active modalities, e.g. --active_modalities image video audio")
    parser.add_argument("--device", type=str, default="auto",
                        help="Compute device: 'auto', 'cuda', or 'cpu'")
    parser.add_argument("--fusion_variant", type=str, choices=["early", "late"], default="early",
                        help="Fusion variant: 'early' or 'late'")
    parser.add_argument("--stage1_epochs", type=int, default=100,
                        help="Number of Stage 1 training epochs")
    parser.add_argument("--stage1_lr", type=float, default=1e-4,
                        help="Stage 1 learning rate")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--num_classes", type=int, default=10,
                        help="Number of output classes")
    parser.add_argument("--eta", type=float, default=5.0,
                        help="RCS minority threshold divisor (majority / eta)")
    parser.add_argument("--shrinkage_lambda", type=float, default=0.1,
                        help="RCS covariance shrinkage lambda")
    parser.add_argument("--clf_epochs", type=int, default=100,
                        help="Stage 3 classifier training epochs")
    parser.add_argument("--clf_lr", type=float, default=1e-3,
                        help="Stage 3 classifier learning rate")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--embed_dim", type=int, default=128,
                        help="Shared embedding dimensionality")
    # Image
    parser.add_argument("--image_in_channels", type=int, default=1,
                        help="Input channels for image encoder (1=grayscale, 3=RGB)")
    # Audio
    parser.add_argument("--audio_in_channels", type=int, default=1,
                        help="Input channels for audio encoder")
    # Text
    parser.add_argument("--text_vocab_size", type=int, default=30000,
                        help="Vocabulary size for text tokenizer")
    parser.add_argument("--text_max_seq_len", type=int, default=128,
                        help="Maximum token sequence length for text encoder")
    # Video
    parser.add_argument("--video_in_channels", type=int, default=3,
                        help="Input channels for video encoder (3=RGB, 1=grayscale)")
    parser.add_argument("--video_recon_frames", type=int, default=8,
                        help="Number of reconstructed video frames")
    # Early stopping & scheduler
    parser.add_argument("--stage1_patience", type=int, default=15,
                        help="Early stopping patience for Stage 1 training (epochs)")
    parser.add_argument("--clf_patience", type=int, default=20,
                        help="Early stopping patience for Stage 3 classifier training (epochs)")
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Constructs a PipelineConfig from parsed argparse arguments."""
    return PipelineConfig(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        active_modalities=args.active_modalities,
        device=args.device,
        fusion_variant=args.fusion_variant,
        stage1_epochs=args.stage1_epochs,
        stage1_lr=args.stage1_lr,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        eta=args.eta,
        shrinkage_lambda=args.shrinkage_lambda,
        clf_epochs=args.clf_epochs,
        clf_lr=args.clf_lr,
        random_seed=args.random_seed,
        embed_dim=args.embed_dim,
        image_in_channels=args.image_in_channels,
        audio_in_channels=args.audio_in_channels,
        text_vocab_size=args.text_vocab_size,
        text_max_seq_len=args.text_max_seq_len,
        video_in_channels=getattr(args, "video_in_channels", 3),
        video_recon_frames=getattr(args, "video_recon_frames", 8),
        stage1_patience=getattr(args, "stage1_patience", 15),
        clf_patience=getattr(args, "clf_patience", 20),
    )

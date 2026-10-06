import os
import json
import argparse
import logging
from typing import Dict, Any
import numpy as np
import torch
from tqdm import tqdm

from multimodal_rcs.config import PipelineConfig, add_pipeline_args
from multimodal_rcs.data import get_avmnist_dataloaders
from multimodal_rcs.models import Stage1Model
from multimodal_rcs.rcs import MultimodalRCSBalancer, evaluate_prdc, evaluate_classwise_prdc
from multimodal_rcs.utils import (
    get_device,
    plot_embedding_comparison,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrainStage2")


def extract_features(model: Stage1Model, dataloader, device: torch.device):
    """Extracts fused representations z_fused and corresponding labels."""
    model.eval()
    fused_list = []
    labels_list = []

    with torch.no_grad():
        for images, audios, targets in dataloader:
            images = images.to(device)
            audios = audios.to(device)
            z_fused = model.extract_fused(images, audios)
            fused_list.append(z_fused.cpu().numpy())
            labels_list.append(targets.numpy())

    features = np.vstack(fused_list)
    labels = np.concatenate(labels_list)
    return features, labels


def run_stage2(config: PipelineConfig):
    device = get_device(config.device)
    os.makedirs(config.features_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.figures_dir, exist_ok=True)

    ckpt_path = os.path.join(
        config.checkpoint_dir, f"stage1_{config.fusion_variant}.pt"
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found at '{ckpt_path}'. Run train_stage1.py first."
        )

    # 1. Load trained Stage 1 Model & Freeze parameters
    logger.info(f"Loading Stage 1 Model checkpoint from '{ckpt_path}'...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Stage1Model(
        fusion_variant=config.fusion_variant,
        embed_dim=config.embed_dim,
        num_classes=config.num_classes,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    logger.info("Stage 1 encoders and fusion module successfully frozen.")

    # 2. Extract features
    logger.info("Loading dataset splits...")
    train_loader, val_loader, test_loader, original_counts = get_avmnist_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_max_count=config.train_max_count,
        train_min_count=config.train_min_count,
        imbalance_mode=config.imbalance_mode,
        val_samples_per_class=config.val_samples_per_class,
        test_samples_per_class=config.test_samples_per_class,
        random_seed=config.random_seed,
    )

    logger.info("Extracting fused representations for train, val, and test splits...")
    train_feats, train_labels = extract_features(model, train_loader, device)
    val_feats, val_labels = extract_features(model, val_loader, device)
    test_feats, test_labels = extract_features(model, test_loader, device)

    # Save raw extracted features
    raw_train_save = os.path.join(config.features_dir, f"train_raw_{config.fusion_variant}.npz")
    val_save = os.path.join(config.features_dir, f"val_{config.fusion_variant}.npz")
    test_save = os.path.join(config.features_dir, f"test_{config.fusion_variant}.npz")

    np.savez_compressed(raw_train_save, features=train_feats, labels=train_labels)
    np.savez_compressed(val_save, features=val_feats, labels=val_labels)
    np.savez_compressed(test_save, features=test_feats, labels=test_labels)
    logger.info(f"Extracted train features: {train_feats.shape}, val: {val_feats.shape}, test: {test_feats.shape}")

    # 3. Regularized Covariance Sharing (RCS) Balancing
    logger.info(f"Applying Multimodal RCS with eta={config.eta}, lambda={config.shrinkage_lambda}...")
    balancer = MultimodalRCSBalancer(
        eta=config.eta,
        shrinkage_lambda=config.shrinkage_lambda,
        top_k=config.top_k_components,
        random_state=config.random_seed,
    )

    balanced_feats, balanced_labels, is_synthetic_mask, info_dict = balancer.balance(
        features=train_feats, labels=train_labels
    )

    # Save balanced features
    balanced_save = os.path.join(config.features_dir, f"train_balanced_rcs_{config.fusion_variant}.npz")
    np.savez_compressed(
        balanced_save,
        features=balanced_feats,
        labels=balanced_labels,
        is_synthetic=is_synthetic_mask,
    )
    logger.info(f"Saved RCS balanced features to '{balanced_save}'")

    # 4. Evaluate synthetic representations using PRDC
    logger.info("Evaluating synthetic representations using PRDC (Precision, Recall, Density, Coverage)...")
    real_feats = balanced_feats[~is_synthetic_mask]
    synth_feats = balanced_feats[is_synthetic_mask]

    overall_prdc = evaluate_prdc(real_feats, synth_feats, nearest_k=5)
    classwise_prdc = evaluate_classwise_prdc(
        real_features=train_feats,
        real_labels=train_labels,
        synthetic_by_class=info_dict["synthetic_by_class"],
        nearest_k=5,
    )

    prdc_results = {
        "fusion_variant": config.fusion_variant,
        "eta": config.eta,
        "shrinkage_lambda": config.shrinkage_lambda,
        "overall_prdc": overall_prdc,
        "classwise_prdc": {str(k): v for k, v in classwise_prdc.items()},
        "partitions": {
            "majority": info_dict["partitions"]["majority"],
            "intermediate": info_dict["partitions"]["intermediate"],
            "minority": info_dict["partitions"]["minority"],
            "threshold": float(info_dict["partitions"]["threshold"]),
        },
        "num_synthetic": int(info_dict["num_synthetic"]),
        "total_balanced": int(info_dict["total_samples"]),
    }

    prdc_json_path = os.path.join(
        config.results_dir, f"prdc_metrics_{config.fusion_variant}.json"
    )
    with open(prdc_json_path, "w") as f:
        json.dump(prdc_results, f, indent=2)

    logger.info("PRDC Evaluation Summary:")
    logger.info(
        f"  Precision: {overall_prdc['precision']:.4f} | Recall: {overall_prdc['recall']:.4f} | "
        f"Density: {overall_prdc['density']:.4f} | Coverage: {overall_prdc['coverage']:.4f}"
    )
    logger.info(f"Saved PRDC metrics to '{prdc_json_path}'")

    # 5. Output t-SNE and UMAP plots before and after RCS oversampling
    logger.info("Generating t-SNE comparison plot...")
    tsne_fig_path = os.path.join(
        config.figures_dir, f"tsne_rcs_comparison_{config.fusion_variant}.png"
    )
    plot_embedding_comparison(
        feats_before=train_feats,
        labels_before=train_labels,
        feats_after=balanced_feats,
        labels_after=balanced_labels,
        is_synthetic_mask=is_synthetic_mask,
        method="tsne",
        title_suffix=f"Fusion: {config.fusion_variant.upper()}",
        save_path=tsne_fig_path,
        random_state=config.random_seed,
    )
    logger.info(f"Saved t-SNE comparison plot to '{tsne_fig_path}'")

    logger.info("Generating UMAP comparison plot...")
    umap_fig_path = os.path.join(
        config.figures_dir, f"umap_rcs_comparison_{config.fusion_variant}.png"
    )
    plot_embedding_comparison(
        feats_before=train_feats,
        labels_before=train_labels,
        feats_after=balanced_feats,
        labels_after=balanced_labels,
        is_synthetic_mask=is_synthetic_mask,
        method="umap",
        title_suffix=f"Fusion: {config.fusion_variant.upper()}",
        save_path=umap_fig_path,
        random_state=config.random_seed,
    )
    logger.info(f"Saved UMAP comparison plot to '{umap_fig_path}'")

    return balanced_save, prdc_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Feature Extraction & RCS Balancing")
    parser = add_pipeline_args(parser)
    args = parser.parse_args()

    cfg = PipelineConfig(
        data_dir=args.data_dir,
        device=args.device,
        fusion_variant=args.fusion_variant,
        stage1_epochs=args.stage1_epochs,
        stage1_lr=args.stage1_lr,
        batch_size=args.batch_size,
        eta=args.eta,
        shrinkage_lambda=args.shrinkage_lambda,
        clf_epochs=args.clf_epochs,
        clf_lr=args.clf_lr,
        random_seed=args.random_seed,
    )
    run_stage2(cfg)

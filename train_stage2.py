"""
Stage 2: Feature Extraction & Regularized Covariance Sharing (RCS) Balancing.

Generalized to support any combination of active modalities on any dataset.
All outputs (features, metrics, plots) are namespaced by dataset_name.
"""

import os
import json
import argparse
import logging
from typing import Dict, Any
import numpy as np
import torch
from tqdm import tqdm

from multimodal_rcs.config import PipelineConfig, add_pipeline_args, config_from_args
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


def _batch_to_modality_dict(batch, active_modalities, device):
    """Unpacks a batch into a dict keyed by modality name."""
    modality_data: Dict[str, torch.Tensor] = {}
    if isinstance(batch, dict):
        targets = batch.get("label", batch.get("labels", batch.get("target", batch.get("targets", None))))
        for mod in active_modalities:
            if mod in batch:
                modality_data[mod] = batch[mod].to(device)
        return modality_data, targets.to(device) if targets is not None else None

    if isinstance(batch, (list, tuple)):
        if len(batch) == 2 and isinstance(batch[0], dict):
            m_dict, targets = batch
            for mod in active_modalities:
                if mod in m_dict:
                    modality_data[mod] = m_dict[mod].to(device)
            return modality_data, targets.to(device)

        *mod_tensors, targets = batch
        for i, mod in enumerate(active_modalities):
            if i < len(mod_tensors):
                modality_data[mod] = mod_tensors[i].to(device)
        return modality_data, targets.to(device)

    raise ValueError(f"Unsupported batch format: {type(batch)}")


def extract_features(model: Stage1Model, dataloader, active_modalities, device: torch.device):
    """Extracts fused representations z_fused and corresponding labels."""
    model.eval()
    fused_list = []
    labels_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features", leave=False):
            modality_data, targets = _batch_to_modality_dict(batch, active_modalities, device)
            z_fused = model.extract_fused(**modality_data)
            fused_list.append(z_fused.cpu().numpy())
            labels_list.append(targets.cpu().numpy())

    features = np.vstack(fused_list)
    labels = np.concatenate(labels_list)
    return features, labels


def run_stage2(config: PipelineConfig):
    device = get_device(config.device)
    os.makedirs(config.features_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.figures_dir, exist_ok=True)

    # Checkpoint namespaced identically to train_stage1
    ckpt_name = f"stage1_{config.modality_tag}_{config.fusion_variant}.pt"
    ckpt_path = os.path.join(config.checkpoint_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found at '{ckpt_path}'. Run train_stage1.py first."
        )

    # 1. Load & freeze Stage 1 Model
    logger.info(f"[{config.dataset_name}] Loading Stage 1 checkpoint '{ckpt_path}'...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Stage1Model(
        active_modalities=config.active_modalities,
        fusion_variant=config.fusion_variant,
        embed_dim=config.embed_dim,
        num_classes=config.num_classes,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        image_in_channels=config.image_in_channels,
        audio_in_channels=config.audio_in_channels,
        text_vocab_size=config.text_vocab_size,
        text_max_seq_len=config.text_max_seq_len,
        text_ff_dim=config.text_ff_dim,
        text_padding_idx=config.text_padding_idx,
        video_in_channels=getattr(config, "video_in_channels", 3),
        video_recon_frames=getattr(config, "video_recon_frames", 8),
        video_recon_height=getattr(config, "video_recon_height", 28),
        video_recon_width=getattr(config, "video_recon_width", 28),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    logger.info("Stage 1 model frozen.")

    # 2. Load data
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

    # 3. Extract features
    tag = f"{config.modality_tag}_{config.fusion_variant}"
    logger.info(f"Extracting fused representations (tag={tag})...")
    train_feats, train_labels = extract_features(model, train_loader, config.active_modalities, device)
    val_feats,   val_labels   = extract_features(model, val_loader,   config.active_modalities, device)
    test_feats,  test_labels  = extract_features(model, test_loader,  config.active_modalities, device)

    # Save raw features
    np.savez_compressed(
        os.path.join(config.features_dir, f"train_raw_{tag}.npz"),
        features=train_feats, labels=train_labels
    )
    np.savez_compressed(
        os.path.join(config.features_dir, f"val_{tag}.npz"),
        features=val_feats, labels=val_labels
    )
    np.savez_compressed(
        os.path.join(config.features_dir, f"test_{tag}.npz"),
        features=test_feats, labels=test_labels
    )
    logger.info(
        f"Extracted: train {train_feats.shape}, val {val_feats.shape}, test {test_feats.shape}"
    )

    # 4. RCS Balancing
    logger.info(f"Applying RCS (eta={config.eta}, lambda={config.shrinkage_lambda})...")
    balancer = MultimodalRCSBalancer(
        eta=config.eta,
        shrinkage_lambda=config.shrinkage_lambda,
        top_k=config.top_k_components,
        random_state=config.random_seed,
    )
    balanced_feats, balanced_labels, is_synthetic_mask, info_dict = balancer.balance(
        features=train_feats, labels=train_labels
    )

    balanced_save = os.path.join(config.features_dir, f"train_balanced_rcs_{tag}.npz")
    np.savez_compressed(
        balanced_save,
        features=balanced_feats,
        labels=balanced_labels,
        is_synthetic=is_synthetic_mask,
    )
    logger.info(f"Saved RCS balanced features to '{balanced_save}'")

    # 5. PRDC Evaluation
    logger.info("Evaluating synthetic representations with PRDC...")
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
        "dataset_name": config.dataset_name,
        "active_modalities": config.active_modalities,
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

    prdc_json_path = os.path.join(config.results_dir, f"prdc_metrics_{tag}.json")
    with open(prdc_json_path, "w") as f:
        json.dump(prdc_results, f, indent=2)

    logger.info(
        f"PRDC | P: {overall_prdc['precision']:.4f} | R: {overall_prdc['recall']:.4f} | "
        f"D: {overall_prdc['density']:.4f} | C: {overall_prdc['coverage']:.4f}"
    )
    logger.info(f"Saved PRDC metrics to '{prdc_json_path}'")

    # 6. Embedding Visualizations
    for method in ["tsne", "umap"]:
        fig_path = os.path.join(config.figures_dir, f"{method}_rcs_comparison_{tag}.png")
        try:
            plot_embedding_comparison(
                feats_before=train_feats,
                labels_before=train_labels,
                feats_after=balanced_feats,
                labels_after=balanced_labels,
                is_synthetic_mask=is_synthetic_mask,
                method=method,
                title_suffix=f"{config.dataset_name} | {tag}",
                save_path=fig_path,
                random_state=config.random_seed,
            )
            logger.info(f"Saved {method.upper()} plot to '{fig_path}'")
        except Exception as e:
            logger.warning(f"Could not generate {method.upper()} plot: {e}")

    return balanced_save, prdc_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Feature Extraction & RCS Balancing")
    parser = add_pipeline_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)
    run_stage2(cfg)

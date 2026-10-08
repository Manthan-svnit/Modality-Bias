"""
Stage 1 Training: Encoders, Modality Trust Gating, Cross-Attention Fusion.

Generalized to support any combination of active modalities
(image, audio, text) on any dataset.

Checkpoint and results are namespaced under the dataset_name from config.
"""

import os
import argparse
import logging
from typing import Dict, List, Any
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from multimodal_rcs.config import PipelineConfig, add_pipeline_args, config_from_args
from multimodal_rcs.data import get_avmnist_dataloaders
from multimodal_rcs.models import Stage1Model
from multimodal_rcs.losses import MultiTaskLoss
from multimodal_rcs.utils import get_device, linear_cka, plot_stage1_metrics

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrainStage1")


def _batch_to_modality_dict(batch, active_modalities: List[str], device: torch.device):
    """Unpacks a batch tuple/dict into a dict keyed by modality name.

    Current AV-MNIST dataloader returns (images, audios, targets).
    If text is active it expects a 4-tuple (images, audios, texts, targets).
    Extend this mapping when integrating new datasets / dataloaders.
    """
    modality_data: Dict[str, torch.Tensor] = {}
    if "image" in active_modalities and "audio" in active_modalities:
        if "text" in active_modalities:
            images, audios, texts, targets = batch
            modality_data["image"] = images.to(device)
            modality_data["audio"] = audios.to(device)
            modality_data["text"] = texts.to(device)
        else:
            images, audios, targets = batch
            modality_data["image"] = images.to(device)
            modality_data["audio"] = audios.to(device)
    elif "image" in active_modalities:
        images, targets = batch
        modality_data["image"] = images.to(device)
    elif "audio" in active_modalities:
        audios, targets = batch
        modality_data["audio"] = audios.to(device)
    elif "text" in active_modalities:
        texts, targets = batch
        modality_data["text"] = texts.to(device)

    targets = targets.to(device)
    return modality_data, targets


def train_stage1(config: PipelineConfig):
    # Set seeds
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)

    device = get_device(config.device)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.figures_dir, exist_ok=True)

    # 1. DataLoaders
    logger.info(
        f"[{config.dataset_name}] Loading data from '{config.data_dir}' | "
        f"Modalities: {config.active_modalities}"
    )
    # NOTE: Replace get_avmnist_dataloaders with your dataset-specific loader.
    # The loader must return (train_loader, val_loader, test_loader, class_counts).
    # Each batch should be a tuple ordered as per _batch_to_modality_dict above.
    train_loader, val_loader, test_loader, class_counts = get_avmnist_dataloaders(
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

    # 2. Stage 1 Model (generalized)
    logger.info(
        f"Instantiating Stage1Model | fusion={config.fusion_variant} | "
        f"modalities={config.active_modalities}"
    )
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
    ).to(device)

    # 3. Multi-Task Criterion (generalized)
    criterion = MultiTaskLoss(
        active_modalities=config.active_modalities,
        weight_recon=config.weight_recon,
        weight_cls=config.weight_cls,
        weight_supcon=config.weight_supcon,
        weight_cmm=config.weight_cmm,
        temperature=config.temperature,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.stage1_lr,
        weight_decay=config.stage1_weight_decay,
    )

    history: Dict[str, List[float]] = {
        "loss_total": [],
        "loss_cls": [],
        "loss_recon": [],
        "loss_supcon": [],
        "loss_cmm": [],
        "val_loss": [],
        "val_acc": [],
    }
    # Add per-modality CKA tracking (fused vs each modality)
    for mod in config.active_modalities:
        history[f"val_cka_fused_{mod}"] = []

    best_val_loss = float("inf")
    # Checkpoint namespaced by dataset + modality tag + fusion variant
    ckpt_name = f"stage1_{config.modality_tag}_{config.fusion_variant}.pt"
    ckpt_path = os.path.join(config.checkpoint_dir, ckpt_name)

    logger.info(
        f"Training for {config.stage1_epochs} epochs on {device}. "
        f"Checkpoint: '{ckpt_path}'"
    )

    for epoch in range(1, config.stage1_epochs + 1):
        model.train()
        epoch_losses: Dict[str, List[float]] = {
            "total": [], "cls": [], "recon": [], "supcon": [], "cmm": []
        }

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{config.stage1_epochs:03d}", leave=False)
        for batch in pbar:
            modality_data, targets = _batch_to_modality_dict(
                batch, config.active_modalities, device
            )

            optimizer.zero_grad()
            outputs = model(**modality_data)

            loss, loss_dict = criterion(
                outputs=outputs,
                targets=targets,
                orig_inputs=modality_data,
            )

            loss.backward()
            optimizer.step()

            epoch_losses["total"].append(loss_dict["loss_total"])
            epoch_losses["cls"].append(loss_dict["loss_cls"])
            epoch_losses["recon"].append(loss_dict["loss_recon"])
            epoch_losses["supcon"].append(loss_dict["loss_supcon"])
            epoch_losses["cmm"].append(loss_dict["loss_cmm"])

            pbar.set_postfix({
                "loss": f"{loss_dict['loss_total']:.4f}",
                "ce": f"{loss_dict['loss_cls']:.4f}"
            })

        avg_loss = float(np.mean(epoch_losses["total"]))
        history["loss_total"].append(avg_loss)
        history["loss_cls"].append(float(np.mean(epoch_losses["cls"])))
        history["loss_recon"].append(float(np.mean(epoch_losses["recon"])))
        history["loss_supcon"].append(float(np.mean(epoch_losses["supcon"])))
        history["loss_cmm"].append(float(np.mean(epoch_losses["cmm"])))

        # Validation step & CKA evaluation
        model.eval()
        val_losses = []
        correct = 0
        total_val = 0
        z_lists: Dict[str, List[torch.Tensor]] = {mod: [] for mod in config.active_modalities}
        z_fused_list = []

        with torch.no_grad():
            for batch in val_loader:
                modality_data, targets = _batch_to_modality_dict(
                    batch, config.active_modalities, device
                )
                outputs = model(**modality_data)
                v_loss, _ = criterion(outputs, targets, modality_data)
                val_losses.append(v_loss.item())

                preds = outputs["logits"].argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total_val += targets.size(0)

                z_fused_list.append(outputs["z_fused"].cpu())
                for mod in config.active_modalities:
                    z_lists[mod].append(outputs[f"z_{mod}"].cpu())

        val_loss = float(np.mean(val_losses))
        val_acc = correct / total_val

        z_fused_all = torch.cat(z_fused_list, dim=0)
        for mod in config.active_modalities:
            z_mod_all = torch.cat(z_lists[mod], dim=0)
            cka_val = linear_cka(z_fused_all, z_mod_all)
            history[f"val_cka_fused_{mod}"].append(cka_val)

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        cka_summary = " | ".join(
            f"CKA(fused,{m}): {history[f'val_cka_fused_{m}'][-1]:.4f}"
            for m in config.active_modalities
        )
        logger.info(
            f"Epoch {epoch:03d} | Train Loss: {avg_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | {cka_summary}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "config": vars(config),
                    "history": history,
                },
                ckpt_path,
            )

    logger.info(
        f"Stage 1 done. Best checkpoint: '{ckpt_path}' (Val Loss: {best_val_loss:.4f})"
    )

    fig_path = os.path.join(
        config.figures_dir,
        f"stage1_{config.modality_tag}_{config.fusion_variant}_history.png"
    )
    plot_stage1_metrics(history, save_path=fig_path)
    logger.info(f"Saved training curves to '{fig_path}'")

    return ckpt_path, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Multimodal Encoder & Fusion Training")
    parser = add_pipeline_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)
    train_stage1(cfg)

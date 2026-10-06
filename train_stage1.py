import os
import argparse
import logging
from typing import Dict, List, Any
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from multimodal_rcs.config import PipelineConfig, add_pipeline_args
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
    logger.info(f"Loading AV-MNIST data from '{config.data_dir}'...")
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

    # 2. Stage 1 Model
    logger.info(f"Instantiating Stage 1 Model with fusion variant: '{config.fusion_variant}'")
    model = Stage1Model(
        fusion_variant=config.fusion_variant,
        embed_dim=config.embed_dim,
        num_classes=config.num_classes,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    # 3. Multi-Task Criterion & Optimizer
    criterion = MultiTaskLoss(
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
        "val_cka_fused_img": [],
        "val_cka_fused_aud": [],
    }

    best_val_loss = float("inf")
    ckpt_path = os.path.join(
        config.checkpoint_dir, f"stage1_{config.fusion_variant}.pt"
    )

    logger.info(f"Starting Stage 1 training for {config.stage1_epochs} epochs on {device}...")

    for epoch in range(1, config.stage1_epochs + 1):
        model.train()
        epoch_losses: Dict[str, List[float]] = {
            "total": [],
            "cls": [],
            "recon": [],
            "supcon": [],
            "cmm": [],
        }

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{config.stage1_epochs:03d}", leave=False)
        for images, audios, targets in pbar:
            images = images.to(device)
            audios = audios.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images, audios)

            loss, loss_dict = criterion(
                outputs=outputs,
                targets=targets,
                orig_images=images,
                orig_audios=audios,
            )

            loss.backward()
            optimizer.step()

            epoch_losses["total"].append(loss_dict["loss_total"])
            epoch_losses["cls"].append(loss_dict["loss_cls"])
            epoch_losses["recon"].append(loss_dict["loss_recon"])
            epoch_losses["supcon"].append(loss_dict["loss_supcon"])
            epoch_losses["cmm"].append(loss_dict["loss_cmm"])

            pbar.set_postfix({"loss": f"{loss_dict['loss_total']:.4f}", "ce": f"{loss_dict['loss_cls']:.4f}"})

        # Epoch training statistics
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
        total = 0
        val_fused_list, val_img_list, val_aud_list = [], [], []

        with torch.no_grad():
            for images, audios, targets in val_loader:
                images = images.to(device)
                audios = audios.to(device)
                targets = targets.to(device)

                outputs = model(images, audios)
                v_loss, _ = criterion(outputs, targets, images, audios)
                val_losses.append(v_loss.item())

                preds = outputs["logits"].argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

                val_fused_list.append(outputs["z_fused"].cpu())
                val_img_list.append(outputs["z_img"].cpu())
                val_aud_list.append(outputs["z_aud"].cpu())

        val_loss = float(np.mean(val_losses))
        val_acc = correct / total

        # Compute CKA between fused representation and each modality representation
        val_fused_all = torch.cat(val_fused_list, dim=0)
        val_img_all = torch.cat(val_img_list, dim=0)
        val_aud_all = torch.cat(val_aud_list, dim=0)

        cka_fused_img = linear_cka(val_fused_all, val_img_all)
        cka_fused_aud = linear_cka(val_fused_all, val_aud_all)

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_cka_fused_img"].append(cka_fused_img)
        history["val_cka_fused_aud"].append(cka_fused_aud)

        logger.info(
            f"Epoch {epoch:03d} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | CKA(Fused, Img): {cka_fused_img:.4f} | CKA(Fused, Aud): {cka_fused_aud:.4f}"
        )

        # Save best model checkpoint
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

    logger.info(f"Stage 1 training finished. Best checkpoint saved to '{ckpt_path}' (Val Loss: {best_val_loss:.4f})")

    # Save training curves plot
    fig_path = os.path.join(
        config.figures_dir, f"stage1_{config.fusion_variant}_history.png"
    )
    plot_stage1_metrics(history, save_path=fig_path)
    logger.info(f"Saved Stage 1 training curves & CKA metrics plot to '{fig_path}'")

    return ckpt_path, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Encoders & Cross-Attention Fusion Training")
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
    train_stage1(cfg)

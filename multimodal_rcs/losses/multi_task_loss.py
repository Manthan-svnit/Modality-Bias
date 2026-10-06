from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .supcon_loss import SupervisedContrastiveLoss
from .cmm_loss import CrossModalityMatchingLoss


class MultiTaskLoss(nn.Module):
    """Multi-Task Loss for Stage 1:
      - Reconstruction Loss (MSE) with weight 1.0
      - Classification Cross-Entropy Loss with weight 1.0
      - Supervised Contrastive Loss (temperature = 0.07) with weight 0.1
      - Cross-Modality Matching Loss (temperature = 0.07) with weight 0.1
    """

    def __init__(
        self,
        weight_recon: float = 1.0,
        weight_cls: float = 1.0,
        weight_supcon: float = 0.1,
        weight_cmm: float = 0.1,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.weight_recon = weight_recon
        self.weight_cls = weight_cls
        self.weight_supcon = weight_supcon
        self.weight_cmm = weight_cmm

        self.supcon_criterion = SupervisedContrastiveLoss(temperature=temperature)
        self.cmm_criterion = CrossModalityMatchingLoss(temperature=temperature)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        orig_images: torch.Tensor,
        orig_audios: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            outputs: Dictionary from Stage1Model containing 'z_img', 'z_aud', 'z_fused', 'logits', 'recon_img', 'recon_aud'
            targets: Ground-truth class labels (B,)
            orig_images: Original input images (B, 1, 28, 28)
            orig_audios: Original input audios (B, 1, 112, 112)

        Returns:
            Tuple of:
                - total_loss (torch.Tensor with grad)
                - loss_dict (Dict[str, float] of detached scalar values)
        """
        # 1. Classification CE Loss
        loss_cls = F.cross_entropy(outputs["logits"], targets)

        # 2. Reconstruction MSE Loss (Image + Audio)
        loss_recon_img = F.mse_loss(outputs["recon_img"], orig_images)
        loss_recon_aud = F.mse_loss(outputs["recon_aud"], orig_audios)
        loss_recon = loss_recon_img + loss_recon_aud

        # 3. Supervised Contrastive Loss on Fused Representation
        loss_supcon = self.supcon_criterion(outputs["z_fused"], targets)

        # 4. Cross-Modality Matching Loss between Image and Audio Encoders
        loss_cmm = self.cmm_criterion(outputs["z_img"], outputs["z_aud"])

        # Combined Total Loss
        total_loss = (
            self.weight_cls * loss_cls
            + self.weight_recon * loss_recon
            + self.weight_supcon * loss_supcon
            + self.weight_cmm * loss_cmm
        )

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_cls": loss_cls.item(),
            "loss_recon": loss_recon.item(),
            "loss_recon_img": loss_recon_img.item(),
            "loss_recon_aud": loss_recon_aud.item(),
            "loss_supcon": loss_supcon.item(),
            "loss_cmm": loss_cmm.item(),
        }

        return total_loss, loss_dict

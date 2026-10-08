"""
Generalized Multi-Task Loss for Stage 1.

Handles any subset of active modalities {image, audio, text}.
Automatically computes only the loss components that apply to the
modalities present in the model outputs dict.

Loss components:
  1. Classification Cross-Entropy        (weight_cls,   always active)
  2. Reconstruction MSE                  (weight_recon, per-active modality)
  3. Supervised Contrastive (SupCon)     (weight_supcon, on z_fused)
  4. Cross-Modality Matching (CMM)       (weight_cmm,   all unique modality pairs)
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .supcon_loss import SupervisedContrastiveLoss
from .cmm_loss import CrossModalityMatchingLoss


# Modality output keys used by Stage1Model
_RECON_KEYS = {
    "image": ("recon_image", None),         # target = original image (passed in)
    "audio": ("recon_audio", None),         # target = original audio
    "text":  ("recon_text", "text_mean_emb"),  # target = mean token embedding (in outputs)
}


class MultiTaskLoss(nn.Module):
    """Dynamic multi-task loss for any combination of active modalities.

    Reconstruction targets for image and audio must be passed explicitly
    (as `orig_inputs` dict). The text reconstruction target is read directly
    from the model output dict (key "text_mean_emb").

    Args:
        active_modalities: List of modality names active in the model,
                           e.g. ["image", "audio"] or ["image", "audio", "text"].
        weight_recon     : Weight applied to total reconstruction loss.
        weight_cls       : Weight applied to classification CE loss.
        weight_supcon    : Weight applied to supervised contrastive loss.
        weight_cmm       : Weight applied to cross-modality matching loss
                           (summed over all unique modality pairs).
        temperature      : Temperature for SupCon and CMM losses.
    """

    def __init__(
        self,
        active_modalities: List[str],
        weight_recon: float = 1.0,
        weight_cls: float = 1.0,
        weight_supcon: float = 0.1,
        weight_cmm: float = 0.1,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.active_modalities = active_modalities
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
        orig_inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            outputs    : Dictionary from Stage1Model forward pass.
            targets    : Ground-truth class labels LongTensor (B,).
            orig_inputs: Dict mapping modality name -> original input tensor.
                         e.g. {"image": img_tensor, "audio": aud_tensor, "text": text_tensor}
                         Only required for modalities that have pixel/spectrogram reconstruction.
                         Text reconstruction target is sourced from outputs["text_mean_emb"].

        Returns:
            Tuple of:
                - total_loss (scalar torch.Tensor with grad)
                - loss_dict (Dict[str, float] with detached scalars)
        """
        loss_dict: Dict[str, float] = {}

        # ------------------------------------------------------------------ #
        # 1. Classification Loss (always)
        # ------------------------------------------------------------------ #
        loss_cls = F.cross_entropy(outputs["logits"], targets)
        loss_dict["loss_cls"] = loss_cls.item()

        # ------------------------------------------------------------------ #
        # 2. Reconstruction Loss (per active modality)
        # ------------------------------------------------------------------ #
        total_recon = torch.tensor(0.0, device=outputs["logits"].device)
        for mod in self.active_modalities:
            recon_key, target_key = _RECON_KEYS[mod]
            if recon_key not in outputs:
                continue

            recon_pred = outputs[recon_key]

            if mod == "text":
                # Target = mean token embedding (stop-gradient, from model output)
                recon_target = outputs[target_key].detach()
            else:
                # Target = original input tensor
                assert mod in orig_inputs, (
                    f"orig_inputs must contain '{mod}' for reconstruction loss"
                )
                recon_target = orig_inputs[mod]

            loss_mod = F.mse_loss(recon_pred, recon_target)
            loss_dict[f"loss_recon_{mod}"] = loss_mod.item()
            total_recon = total_recon + loss_mod

        loss_dict["loss_recon"] = total_recon.item()

        # ------------------------------------------------------------------ #
        # 3. Supervised Contrastive Loss on z_fused
        # ------------------------------------------------------------------ #
        loss_supcon = self.supcon_criterion(outputs["z_fused"], targets)
        loss_dict["loss_supcon"] = loss_supcon.item()

        # ------------------------------------------------------------------ #
        # 4. Cross-Modality Matching Loss (all unique pairs)
        # ------------------------------------------------------------------ #
        z_per_mod = [outputs[f"z_{m}"] for m in self.active_modalities]
        loss_cmm = torch.tensor(0.0, device=outputs["logits"].device)
        num_pairs = 0
        N = len(self.active_modalities)
        for i in range(N):
            for j in range(i + 1, N):
                loss_cmm = loss_cmm + self.cmm_criterion(z_per_mod[i], z_per_mod[j])
                num_pairs += 1
        if num_pairs > 0:
            loss_cmm = loss_cmm / num_pairs
        loss_dict["loss_cmm"] = loss_cmm.item()

        # ------------------------------------------------------------------ #
        # Total Combined Loss
        # ------------------------------------------------------------------ #
        total_loss = (
            self.weight_cls * loss_cls
            + self.weight_recon * total_recon
            + self.weight_supcon * loss_supcon
            + self.weight_cmm * loss_cmm
        )
        loss_dict["loss_total"] = total_loss.item()

        return total_loss, loss_dict

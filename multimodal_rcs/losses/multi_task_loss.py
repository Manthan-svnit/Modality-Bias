"""
Generalized Multi-Task Loss for Stage 1.

Handles any subset of active modalities {image, audio, text, video}.
Automatically computes only the loss components that apply to the
modalities present in the model outputs dict.

Loss components:
  1. Classification Cross-Entropy        (weight_cls,   always active)
  2. Reconstruction MSE                  (weight_recon, per-active modality)
  3. Supervised Contrastive (SupCon)     (weight_supcon, on z_fused)
  4. Cross-Modality Matching (CMM)       (weight_cmm,   all unique modality pairs)
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .supcon_loss import SupervisedContrastiveLoss
from .cmm_loss import CrossModalityMatchingLoss

# Canonical modality output keys used by Stage1Model
_RECON_KEYS = {
    "image": ("recon_image", None),         # target = original image
    "audio": ("recon_audio", None),         # target = original audio
    "text":  ("recon_text", "text_mean_emb"),  # target = mean token embedding (in outputs)
    "video": ("recon_video", None),         # target = original video
}


class MultiTaskLoss(nn.Module):
    """Dynamic multi-task loss for any combination of active modalities.

    Reconstruction targets for image, audio, and video can be passed explicitly
    as `orig_inputs` dict or as positional arguments. Text reconstruction target
    is read directly from the model output dict (key "text_mean_emb").

    Args:
        active_modalities: List of modality names active in the model,
                           e.g. ["image", "audio"], ["image", "video"], etc.
                           Defaults to ["image", "audio"] for backward compatibility.
        weight_recon     : Weight applied to total reconstruction loss.
        weight_cls       : Weight applied to classification CE loss.
        weight_supcon    : Weight applied to supervised contrastive loss.
        weight_cmm       : Weight applied to cross-modality matching loss
                           (summed over all unique modality pairs).
        temperature      : Temperature for SupCon and CMM losses.
    """

    def __init__(
        self,
        active_modalities: Optional[List[str]] = None,
        weight_recon: float = 1.0,
        weight_cls: float = 1.0,
        weight_supcon: float = 0.1,
        weight_cmm: float = 0.1,
        temperature: float = 0.07,
    ):
        super().__init__()
        if active_modalities is None:
            active_modalities = ["image", "audio"]
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
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            outputs    : Dictionary from Stage1Model forward pass.
            targets    : Ground-truth class labels LongTensor (B,).
            *args      : Optional positional tensors corresponding to active modalities,
                         e.g. criterion(outputs, targets, images, audios).
            **kwargs   : Can include orig_inputs=Dict[str, Tensor].

        Returns:
            Tuple of:
                - total_loss (scalar torch.Tensor with grad)
                - loss_dict (Dict[str, float] with detached scalars)
        """
        orig_inputs: Dict[str, torch.Tensor] = kwargs.get("orig_inputs", {})
        if not orig_inputs and len(args) > 0:
            for i, arg in enumerate(args):
                if i < len(self.active_modalities) and isinstance(arg, torch.Tensor):
                    orig_inputs[self.active_modalities[i]] = arg

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
            if mod not in _RECON_KEYS:
                continue
            recon_key, target_key = _RECON_KEYS[mod]
            if recon_key not in outputs:
                continue

            recon_pred = outputs[recon_key]

            if mod == "text":
                # Target = mean token embedding (stop-gradient, from model output)
                recon_target = outputs[target_key].detach()
            else:
                # Target = original input tensor
                if mod not in orig_inputs:
                    continue
                recon_target = orig_inputs[mod]

            # Adjust shape if slightly different resolution
            if recon_pred.shape != recon_target.shape:
                if recon_pred.dim() == 4:
                    recon_pred = F.interpolate(
                        recon_pred, size=recon_target.shape[2:], mode="bilinear", align_corners=False
                    )
                elif recon_pred.dim() == 5:
                    recon_pred = F.interpolate(
                        recon_pred, size=recon_target.shape[2:], mode="trilinear", align_corners=False
                    )

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
        z_per_mod = [
            outputs[f"z_{m}"] for m in self.active_modalities if f"z_{m}" in outputs
        ]
        loss_cmm = torch.tensor(0.0, device=outputs["logits"].device)
        num_pairs = 0
        N = len(z_per_mod)
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

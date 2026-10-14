"""
losses.py — All loss functions for Stage 1.

Implemented:
  1. SupConLoss     — Supervised Contrastive Loss (Khosla et al., 2020)
  2. InfoNCELoss    — Cross-modal InfoNCE alignment (optional)
  3. ReconLoss      — MSE reconstruction loss across all modalities
  4. CrossEntropyLoss — Standard CE with optional label smoothing

Composite loss is assembled by the trainer using weights from cfg.loss.
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 1. Supervised Contrastive Loss
# ──────────────────────────────────────────────────────────────────────────────

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Reference: https://arxiv.org/abs/2004.11362

    Parameters
    ----------
    temperature : float
        Logit scale τ (typically 0.03–0.3).
    base_temperature : float
        Normalisation constant (kept at 0.07 per original paper).
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features : [B, proj_dim]  — L2-normalised projections
        labels   : [B]            — integer class labels

        Returns
        -------
        loss : scalar
        """
        device = features.device
        B = features.shape[0]

        if B < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Cosine similarity matrix [B, B]
        sim = torch.mm(features, features.T) / self.temperature   # [B, B]

        # Mask: same-class pairs (excluding self)
        labels = labels.view(-1, 1)                               # [B, 1]
        mask = (labels == labels.T).float().to(device)             # [B, B]
        self_mask = torch.eye(B, device=device)
        mask = mask - self_mask                                    # remove diagonal
        # Remove rows where anchor has no positives
        valid = mask.sum(dim=1) > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Log-sum-exp denominator (exclude self)
        logits_mask = 1 - self_mask                                # [B, B]
        exp_sim = torch.exp(sim) * logits_mask                    # [B, B]
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

        # Mean over positives for each anchor
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-9)

        # Scale and average
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss[valid].mean()
        return loss


# ──────────────────────────────────────────────────────────────────────────────
# 2. Cross-modal InfoNCE (optional alignment)
# ──────────────────────────────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    """
    Pairwise cross-modal InfoNCE (used for cross-modal alignment).

    Given two modality projections p_a and p_b, maximises agreement
    between same-sample pairs.

    Parameters
    ----------
    temperature : float
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        p_a: torch.Tensor,
        p_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        p_a, p_b : [B, proj_dim]  — L2-normalised projections from two modalities

        Returns
        -------
        loss : scalar
        """
        B = p_a.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=p_a.device, requires_grad=True)

        # [B, B] similarity matrix
        logits = torch.mm(p_a, p_b.T) / self.temperature
        labels = torch.arange(B, device=p_a.device)

        # Symmetric: a→b and b→a
        loss = (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)
        ) / 2
        return loss


# ──────────────────────────────────────────────────────────────────────────────
# 3. Reconstruction (MSE) Loss
# ──────────────────────────────────────────────────────────────────────────────

class ReconLoss(nn.Module):
    """
    MSE reconstruction loss averaged over all active modalities.

    target : z_m     (stop-gradient detached from encoder)
    pred   : decoder(z_fused)
    """

    def forward(
        self,
        z_rec: Dict[str, torch.Tensor],
        z_target: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z_rec    : dict modality → [B, d]  (decoder outputs)
        z_target : dict modality → [B, d]  (encoder outputs, detached)

        Returns
        -------
        loss : scalar
        """
        losses = []
        for mod in z_rec:
            if mod in z_target:
                losses.append(
                    F.mse_loss(z_rec[mod], z_target[mod].detach())
                )
        if not losses:
            return torch.tensor(0.0, requires_grad=True)
        return torch.stack(losses).mean()


# ──────────────────────────────────────────────────────────────────────────────
# 4. Cross-Entropy Loss
# ──────────────────────────────────────────────────────────────────────────────

class ClassificationLoss(nn.Module):
    """Standard cross-entropy loss with optional label smoothing."""

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, labels)


# ──────────────────────────────────────────────────────────────────────────────
# Composite loss assembler
# ──────────────────────────────────────────────────────────────────────────────

class Stage1Loss(nn.Module):
    """
    Assembles all Stage 1 loss terms according to cfg.loss weights and
    the contrastive_placement setting.

    Supports epoch-based warmup for contrastive terms.

    Parameters
    ----------
    loss_cfg : ConfigNode
        ``cfg.loss``
    contrastive_placement : str
        ``"modality_only"`` | ``"fused_only"`` | ``"both"``
    modalities : list of str
    """

    def __init__(self, loss_cfg, contrastive_placement: str, modalities: List[str]):
        super().__init__()
        self.cfg = loss_cfg
        self.placement = contrastive_placement
        self.modalities = modalities
        self.warmup_epochs: int = getattr(loss_cfg, "warmup_epochs", 5)

        self.supcon = SupConLoss(
            temperature=loss_cfg.supcon_temperature,
            base_temperature=loss_cfg.supcon_temperature,
        )
        self.infonce = InfoNCELoss(temperature=loss_cfg.supcon_temperature)
        self.recon = ReconLoss()
        self.ce = ClassificationLoss()

    def _warmup_scale(self, epoch: int) -> float:
        if self.warmup_epochs <= 0:
            return 1.0
        return min(1.0, epoch / self.warmup_epochs)

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        out    : forward() output dict from MultimodalModel
        labels : [B] integer class labels
        epoch  : current training epoch (for warmup)

        Returns
        -------
        dict with keys:
          ``total``  : weighted sum of all active terms
          ``con_fused``, ``con_mod``, ``cls``, ``rec``, ``align``  : individual terms
        """
        ws = self._warmup_scale(epoch)
        cfg = self.cfg

        # ── Contrastive: fused ───────────────────────────────────────────────
        if self.placement in ("fused_only", "both"):
            l_con_fused = self.supcon(out["p_fused"], labels)
        else:
            l_con_fused = torch.tensor(0.0, device=labels.device)

        # ── Contrastive: per-modality ─────────────────────────────────────────
        if self.placement in ("modality_only", "both"):
            l_con_mods = [
                self.supcon(out["p_mod"][mod], labels)
                for mod in self.modalities
            ]
            l_con_mod = torch.stack(l_con_mods).mean()
        else:
            l_con_mod = torch.tensor(0.0, device=labels.device)

        # ── Cross-entropy ─────────────────────────────────────────────────────
        l_cls = self.ce(out["logits"], labels)

        # ── Reconstruction ────────────────────────────────────────────────────
        l_rec = self.recon(out["z_rec"], out["z"])

        # ── Cross-modal alignment (InfoNCE) ───────────────────────────────────
        w_align = float(cfg.w_align)
        if w_align > 0 and len(self.modalities) >= 2:
            mods = self.modalities
            align_losses = []
            for i in range(len(mods)):
                for j in range(i + 1, len(mods)):
                    align_losses.append(
                        self.infonce(out["p_mod"][mods[i]], out["p_mod"][mods[j]])
                    )
            l_align = torch.stack(align_losses).mean()
        else:
            l_align = torch.tensor(0.0, device=labels.device)

        # ── Weighted sum ──────────────────────────────────────────────────────
        total = (
            ws * float(cfg.w_con_fused) * l_con_fused
            + ws * float(cfg.w_con_mod)  * l_con_mod
            + float(cfg.w_cls)           * l_cls
            + float(cfg.w_rec)           * l_rec
            + ws * w_align               * l_align
        )

        return {
            "total": total,
            "con_fused": l_con_fused.detach(),
            "con_mod":   l_con_mod.detach(),
            "cls":       l_cls.detach(),
            "rec":       l_rec.detach(),
            "align":     l_align.detach(),
        }

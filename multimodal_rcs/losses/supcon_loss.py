import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
    
    Pulls representations of the same class together while pushing apart different classes.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape (B, embed_dim), unnormalized or normalized.
            labels: 1D Tensor of shape (B,) containing class labels.
        Returns:
            Scalar loss tensor.
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # L2 normalize features
        norm_features = F.normalize(features, p=2, dim=-1)

        # Compute cosine similarity matrix scaled by temperature
        sim_matrix = torch.matmul(norm_features, norm_features.T) / self.temperature  # (B, B)

        # Mask for diagonal (self-contrast)
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)

        # Mask for positive pairs (same class, not self)
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T) & (~diag_mask)  # (B, B)

        # Numerical stability: subtract max per row (excluding diagonal)
        sim_matrix_for_max = sim_matrix.masked_fill(diag_mask, -1e9)
        max_sim, _ = torch.max(sim_matrix_for_max, dim=1, keepdim=True)
        logits = sim_matrix - max_sim.detach()

        # Denominator: log-sum-exp over all a != i
        exp_logits = torch.exp(logits) * (~diag_mask).float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Mean of log-likelihood over positive pairs for each anchor
        pos_counts = pos_mask.sum(dim=1)
        valid_anchors = pos_counts > 0

        if not valid_anchors.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1)[valid_anchors] / pos_counts[valid_anchors]
        loss = -mean_log_prob_pos.mean()
        return loss

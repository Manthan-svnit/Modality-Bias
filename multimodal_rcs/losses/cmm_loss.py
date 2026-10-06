import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalityMatchingLoss(nn.Module):
    """Bidirectional InfoNCE Cross-Modality Matching Loss.
    
    Aligns paired visual and audio embeddings in the shared embedding space.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_img: torch.Tensor, z_aud: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_img: Visual embedding of shape (B, embed_dim)
            z_aud: Audio embedding of shape (B, embed_dim)
        Returns:
            Scalar loss tensor.
        """
        batch_size = z_img.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=z_img.device, requires_grad=True)

        # L2 normalize
        u = F.normalize(z_img, p=2, dim=-1)
        v = F.normalize(z_aud, p=2, dim=-1)

        # Pairwise cosine similarity matrix scaled by temperature
        logits_img2aud = torch.matmul(u, v.T) / self.temperature  # (B, B)
        logits_aud2img = logits_img2aud.T                         # (B, B)

        targets = torch.arange(batch_size, device=z_img.device, dtype=torch.long)

        loss_i2a = F.cross_entropy(logits_img2aud, targets)
        loss_a2i = F.cross_entropy(logits_aud2img, targets)

        loss = 0.5 * (loss_i2a + loss_a2i)
        return loss

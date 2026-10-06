import torch
from multimodal_rcs.losses import (
    SupervisedContrastiveLoss,
    CrossModalityMatchingLoss,
    MultiTaskLoss,
)


def test_supcon_loss():
    batch_size = 16
    feats = torch.randn(batch_size, 128)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3, 4, 4, 5, 5])

    loss_fn = SupervisedContrastiveLoss(temperature=0.07)
    loss = loss_fn(feats, labels)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0


def test_cmm_loss():
    batch_size = 8
    z_img = torch.randn(batch_size, 128)
    z_aud = torch.randn(batch_size, 128)

    loss_fn = CrossModalityMatchingLoss(temperature=0.07)
    loss = loss_fn(z_img, z_aud)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0


def test_multitask_loss_backward():
    batch_size = 4
    images = torch.randn(batch_size, 1, 28, 28, requires_grad=True)
    audios = torch.randn(batch_size, 1, 112, 112, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3])

    from multimodal_rcs.models import Stage1Model
    model = Stage1Model(fusion_variant="early")

    outputs = model(images, audios)
    criterion = MultiTaskLoss()

    total_loss, loss_dict = criterion(outputs, targets, images, audios)
    assert total_loss.requires_grad
    total_loss.backward()

    # Verify gradients computed on model parameters
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient"

"""
Tests for VideoEncoder, VideoDecoder, and Arbitrary N-Modality Dynamic Generalization.
"""

import pytest
import torch
import torch.nn as nn
from multimodal_rcs.models import (
    VideoEncoder,
    VideoDecoder,
    ImageEncoder,
    AudioEncoder,
    TextEncoder,
    Stage1Model,
    ModalityTrustGating,
    EarlyFusionModule,
    LateFusionModule,
)
from multimodal_rcs.losses import MultiTaskLoss
from multimodal_rcs.data import GenericMultimodalDataset
from torch.utils.data import DataLoader


def test_video_encoder_shapes_and_invariance():
    batch_size = 2
    embed_dim = 128

    encoder = VideoEncoder(embed_dim=embed_dim, in_channels=3)

    # 1. Standard video batch (B, C, T, H, W)
    video_16f = torch.randn(batch_size, 3, 16, 28, 28)
    z1 = encoder(video_16f)
    assert z1.shape == (batch_size, embed_dim)

    # 2. Invariance to different frame counts (T=8) and spatial resolutions (H=40, W=40)
    video_8f_large = torch.randn(batch_size, 3, 8, 40, 40)
    z2 = encoder(video_8f_large)
    assert z2.shape == (batch_size, embed_dim)

    # 3. Gradients flow
    loss = z1.sum()
    loss.backward()
    for name, p in encoder.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"VideoEncoder parameter {name} has no gradient"


def test_video_decoder():
    batch_size = 3
    embed_dim = 128
    out_channels = 3
    out_frames = 8
    out_height = 28
    out_width = 28

    decoder = VideoDecoder(
        embed_dim=embed_dim,
        out_channels=out_channels,
        out_frames=out_frames,
        out_height=out_height,
        out_width=out_width,
    )

    z = torch.randn(batch_size, embed_dim)
    recon = decoder(z)
    assert recon.shape == (batch_size, out_channels, out_frames, out_height, out_width)

    # Dynamic target shape
    recon_custom = decoder(z, target_shape=(12, 32, 32))
    assert recon_custom.shape == (batch_size, out_channels, 12, 32, 32)


def test_dynamic_modality_combinations():
    batch_size = 2
    embed_dim = 128
    num_classes = 5

    x_img = torch.randn(batch_size, 1, 28, 28)
    x_aud = torch.randn(batch_size, 1, 112, 112)
    x_vid = torch.randn(batch_size, 3, 8, 28, 28)
    x_txt = torch.randint(1, 1000, (batch_size, 16))

    combinations = [
        # (active_modalities, input_kwargs)
        (["image", "video"], {"image": x_img, "video": x_vid}),
        (["video", "text"], {"video": x_vid, "text": x_txt}),
        (["image", "video", "audio"], {"image": x_img, "video": x_vid, "audio": x_aud}),
        (["image", "video", "audio", "text"], {"image": x_img, "video": x_vid, "audio": x_aud, "text": x_txt}),
    ]

    for active_mods, inputs in combinations:
        for fusion in ["early", "late"]:
            model = Stage1Model(
                active_modalities=active_mods,
                fusion_variant=fusion,
                embed_dim=embed_dim,
                num_classes=num_classes,
                text_vocab_size=1000,
                text_max_seq_len=32,
            )

            outputs = model(**inputs)

            assert outputs["z_fused"].shape == (batch_size, embed_dim)
            assert outputs["logits"].shape == (batch_size, num_classes)
            assert outputs["trust_scores"].shape == (batch_size, len(active_mods))
            assert torch.allclose(
                outputs["trust_scores"].sum(dim=-1), torch.ones(batch_size), atol=1e-5
            )

            # Check per-modality and reconstruction keys
            for mod in active_mods:
                assert f"z_{mod}" in outputs
                assert outputs[f"z_{mod}"].shape == (batch_size, embed_dim)
                assert f"recon_{mod}" in outputs

            # Check feature extractor
            z_ext = model.extract_fused(**inputs)
            assert z_ext.shape == (batch_size, embed_dim)


def test_nmodality_multitask_loss():
    batch_size = 4
    num_classes = 4
    embed_dim = 128

    active_modalities = ["image", "video", "audio"]
    model = Stage1Model(
        active_modalities=active_modalities,
        embed_dim=embed_dim,
        num_classes=num_classes,
        fusion_variant="early",
    )

    criterion = MultiTaskLoss(
        active_modalities=active_modalities,
        weight_recon=1.0,
        weight_cls=1.0,
        weight_supcon=0.1,
        weight_cmm=0.1,
    )

    inputs = {
        "image": torch.randn(batch_size, 1, 28, 28, requires_grad=True),
        "video": torch.randn(batch_size, 3, 8, 28, 28, requires_grad=True),
        "audio": torch.randn(batch_size, 1, 112, 112, requires_grad=True),
    }
    targets = torch.tensor([0, 1, 2, 3])

    outputs = model(**inputs)
    total_loss, loss_dict = criterion(outputs=outputs, targets=targets, orig_inputs=inputs)

    assert total_loss.requires_grad
    assert "loss_recon_video" in loss_dict
    assert "loss_recon_image" in loss_dict
    assert "loss_recon_audio" in loss_dict
    assert "loss_supcon" in loss_dict
    assert "loss_cmm" in loss_dict
    assert "loss_cls" in loss_dict

    total_loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient"


def test_generic_multimodal_dataset_dataloader():
    num_samples = 20
    modalities = {
        "image": torch.randn(num_samples, 1, 28, 28),
        "video": torch.randn(num_samples, 3, 4, 14, 14),
    }
    labels = torch.randint(0, 5, (num_samples,))

    ds = GenericMultimodalDataset(modalities=modalities, labels=labels)
    assert len(ds) == num_samples

    item_mods, label = ds[0]
    assert "image" in item_mods and "video" in item_mods
    assert item_mods["image"].shape == (1, 28, 28)
    assert item_mods["video"].shape == (3, 4, 14, 14)

    loader = DataLoader(ds, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    b_mods, b_labels = batch
    assert b_mods["image"].shape == (4, 1, 28, 28)
    assert b_mods["video"].shape == (4, 3, 4, 14, 14)
    assert b_labels.shape == (4,)

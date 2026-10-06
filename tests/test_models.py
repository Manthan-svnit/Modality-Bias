import torch
from multimodal_rcs.models import (
    ImageEncoder,
    AudioEncoder,
    ModalityTrustGating,
    CrossAttentionFusion,
    EarlyFusionModule,
    LateFusionModule,
    ImageDecoder,
    AudioDecoder,
    Stage1Model,
    Stage3Classifier,
)


def test_encoders_and_gating():
    batch_size = 4
    images = torch.randn(batch_size, 1, 28, 28)
    audios = torch.randn(batch_size, 1, 112, 112)

    img_enc = ImageEncoder(embed_dim=128)
    aud_enc = AudioEncoder(embed_dim=128)

    z_img = img_enc(images)
    z_aud = aud_enc(audios)

    assert z_img.shape == (batch_size, 128)
    assert z_aud.shape == (batch_size, 128)

    gating = ModalityTrustGating(embed_dim=128)
    g_img, g_aud, trust = gating(z_img, z_aud)

    assert g_img.shape == (batch_size, 128)
    assert g_aud.shape == (batch_size, 128)
    assert trust.shape == (batch_size, 2)
    # Check softmax property: sum equals 1
    assert torch.allclose(trust.sum(dim=-1), torch.ones(batch_size), atol=1e-5)


def test_fusion_variants():
    batch_size = 4
    z_img = torch.randn(batch_size, 128)
    z_aud = torch.randn(batch_size, 128)

    early = EarlyFusionModule(dim=128, num_heads=4, num_layers=2, dropout=0.1)
    late = LateFusionModule(dim=128, num_heads=4, num_layers=2, dropout=0.1)

    fused_early = early(z_img, z_aud)
    fused_late = late(z_img, z_aud)

    assert fused_early.shape == (batch_size, 128)
    assert fused_late.shape == (batch_size, 128)


def test_decoders():
    batch_size = 4
    z_fused = torch.randn(batch_size, 128)

    img_dec = ImageDecoder(embed_dim=128)
    aud_dec = AudioDecoder(embed_dim=128)

    recon_img = img_dec(z_fused)
    recon_aud = aud_dec(z_fused)

    assert recon_img.shape == (batch_size, 1, 28, 28)
    assert recon_aud.shape == (batch_size, 1, 112, 112)


def test_stage1_model_forward():
    batch_size = 4
    images = torch.randn(batch_size, 1, 28, 28)
    audios = torch.randn(batch_size, 1, 112, 112)

    model = Stage1Model(fusion_variant="early", embed_dim=128, num_classes=10)
    out = model(images, audios)

    assert out["z_img"].shape == (batch_size, 128)
    assert out["z_aud"].shape == (batch_size, 128)
    assert out["z_fused"].shape == (batch_size, 128)
    assert out["logits"].shape == (batch_size, 10)
    assert out["recon_img"].shape == (batch_size, 1, 28, 28)
    assert out["recon_aud"].shape == (batch_size, 1, 112, 112)

    extracted = model.extract_fused(images, audios)
    assert extracted.shape == (batch_size, 128)


def test_stage3_classifier():
    batch_size = 8
    x = torch.randn(batch_size, 128)
    clf = Stage3Classifier(in_dim=128, hidden1=256, hidden2=128, num_classes=10)
    logits = clf(x)
    assert logits.shape == (batch_size, 10)

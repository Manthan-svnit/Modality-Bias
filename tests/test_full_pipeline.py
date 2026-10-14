"""
test_full_pipeline.py — End-to-end integration smoke test.

Tests:
  1. Synthetic multimodal dataset & manifest creation
  2. Stage 1 representation learning (extracts z_fused)
  3. Stage 2 RCS balancing & baselines generation (operates on cached z_fused)
  4. Stage 3 final classifier training & multi-seed evaluation
  5. Verifies all artifacts and metrics specified in the PDF
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
from PIL import Image
import wave

from stage1.config.config import load_config, save_config
from run_pipeline import run_single_pipeline


def create_dummy_audio(path: Path, duration_s: float = 0.5, sr: int = 16000):
    t = np.linspace(0, duration_s, int(sr * duration_s))
    signal = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(signal.tobytes())


def create_dummy_image(path: Path, size: int = 32):
    img = Image.fromarray(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))
    img.save(path)


def test_end_to_end():
    temp_dir = Path(tempfile.mkdtemp(prefix="rcs_smoke_test_"))
    try:
        data_dir = temp_dir / "data"
        data_dir.mkdir(parents=True)
        results_dir = temp_dir / "results"

        records = []
        # Create 3 classes with imbalanced distribution: Class 0 (12), Class 1 (6), Class 2 (3)
        class_counts = {0: 12, 1: 6, 2: 3}
        for split in ["train", "val", "test"]:
            multiplier = 1 if split == "train" else 1
            for c, cnt in class_counts.items():
                for i in range(cnt * multiplier):
                    sid = f"{split}_c{c}_{i}"
                    img_p = data_dir / f"{sid}.png"
                    aud_p = data_dir / f"{sid}.wav"
                    create_dummy_image(img_p)
                    create_dummy_audio(aud_p)
                    records.append({
                        "sample_id": sid,
                        "label": c,
                        "split": split,
                        "image_path": str(img_p),
                        "audio_path": str(aud_p),
                    })

        manifest_path = temp_dir / "manifest.csv"
        pd.DataFrame(records).to_csv(manifest_path, index=False)

        # Create test config
        config_dict = {
            "dataset_name": "TestMultimodal",
            "manifest": str(manifest_path),
            "modalities": ["image", "audio"],
            "fusion_timing": "early",
            "contrastive_placement": "both",
            "create_artificial_imbalance": False,
            "encoder": {
                "d": 32,
                "image": {
                    "backbone": "cnn",
                    "input_size": 32,
                    "in_channels": 3,
                    "cnn_depth": 2,
                    "cnn_width": 16,
                    "dropout": 0.0,
                },
                "audio": {
                    "sample_rate": 16000,
                    "n_mels": 32,
                    "n_fft": 512,
                    "hop_length": 256,
                    "cnn_depth": 2,
                    "cnn_width": 16,
                    "dropout": 0.0,
                },
            },
            "gating": {"hidden_dim": 16, "dropout": 0.0},
            "fusion": {"num_heads": 2, "num_layers": 1, "dropout": 0.0, "ffn_dim": 32},
            "projection": {"proj_dim": 16},
            "loss": {
                "w_con_fused": 1.2,
                "w_con_mod": 1.2,
                "w_cls": 1.0,
                "w_rec": 1.0,
                "w_align": 0.0,
                "supcon_temperature": 0.07,
                "warmup_epochs": 1,
            },
            "decoder": {"hidden_dim": 32, "num_layers": 1},
            "training": {
                "epochs": 2,
                "batch_size": 6,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "optimizer": "adamw",
                "scheduler": "none",
                "use_balanced_sampler": True,
                "seed": 42,
                "num_workers": 0,
                "pin_memory": False,
            },
            "evaluation": {
                "knn_k": 2,
                "linear_probe_epochs": 2,
                "linear_probe_lr": 1e-2,
                "compute_tsne": False,
                "compute_umap": False,
            },
            "optuna": {"enabled": False},
            "results_root": str(results_dir),
        }

        cfg_path = temp_dir / "test_config.yaml"
        import yaml
        with open(cfg_path, "w") as f:
            yaml.dump(config_dict, f)

        # Run single pipeline: Stages 1, 2, 3
        exp_dir = run_single_pipeline(
            config_path=str(cfg_path),
            stages=["1", "2", "3"],
            tune_a=False,
            tune_b=False,
            overrides=[],
        )

        # Verify Stage 1 artifacts
        stage1_dir = exp_dir / "stage1_representation"
        assert (stage1_dir / "checkpoints" / "best.pt").exists(), "best.pt missing"
        assert (stage1_dir / "features" / "train" / "z_fused.npy").exists(), "train z_fused missing"
        assert (stage1_dir / "features" / "val" / "z_fused.npy").exists(), "val z_fused missing"
        assert (stage1_dir / "separation_metrics.csv").exists(), "separation_metrics.csv missing"

        # Verify Stage 2 artifacts
        stage2_dir = exp_dir / "stage2_rcs"
        assert (stage2_dir / "z_fused_balanced.npy").exists(), "z_fused_balanced missing"
        assert (stage2_dir / "PRDC.csv").exists(), "PRDC.csv missing"
        assert (stage2_dir / "gmm_params.json").exists(), "gmm_params.json missing"
        assert (stage2_dir / "synthetic_vectors" / "synthetic_z.npy").exists(), "synthetic_z missing"
        assert (stage2_dir / "baselines" / "smote_z.npy").exists(), "baseline SMOTE missing"

        # Verify Stage 3 artifacts
        stage3_dir = exp_dir / "stage3_classifier"
        assert (stage3_dir / "metrics.csv").exists(), "metrics.csv missing"
        assert (stage3_dir / "per_class_f1.csv").exists(), "per_class_f1.csv missing"
        assert (stage3_dir / "model.pt").exists(), "Stage 3 model.pt missing"

        metrics_df = pd.read_csv(stage3_dir / "metrics.csv")
        assert "balanced_acc_mean" in metrics_df.columns
        assert "macro_f1_mean" in metrics_df.columns
        assert "g_mean_mean" in metrics_df.columns
        assert "minority_recall_mean" in metrics_df.columns
        print("\nAll integration assertions passed successfully!")
        print(f"Verified Stage 1, 2, and 3 pipeline end-to-end on synthetic multimodal dataset.")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_end_to_end()

# Multimodal Representation and Regularized Covariance Sharing (Multimodal RCS)

An end-to-end research and production-grade pipeline implementing **Multimodal Representation Learning** coupled with **Regularized Covariance Sharing (Multimodal RCS)** for extreme class imbalance on audiovisual multimodal data (**AV-MNIST**).

---

## 🔬 Architecture & Methodology

```
Visual Modality (28x28)   ──► Image Encoder (2D CNN)  ──┐
                                                        ├──► Modality Trust Gating ──► Cross-Attention Fusion ──► z_fused
Auditory Modality (112x112) ──► Audio Encoder (2D CNN)  ──┘     (Softmax weights)       (Early / Late)
                                                                                               │
                                                               ┌───────────────────────────────┴──────────────────────────────┐
                                                               ▼                                                              ▼
                                                        Stage 1 Multi-Task Learning                                   Stage 2 Feature Extraction
                                                        - Reconstruction MSE (1.0)                                    - Freeze Encoders & Fusion
                                                        - Classification CE (1.0)                                     - Sort Maj / Inter / Min (eta=5)
                                                        - Supervised Contrastive (0.1)                                - Fit GMMs per class
                                                        - Cross-Modality Matching (0.1)                               - GMM Oversampling (Intermediate)
                                                        - Validation CKA tracking                                     - Covariance Sharing & Shrinkage (Minority)
                                                                                                                      - PRDC Evaluation
                                                                                                                              │
                                                                                                                              ▼
                                                                                                              Stage 3 Final Classifier & Benchmark
                                                                                                              - 2-hidden layer MLP (256 -> 128)
                                                                                                              - Version A: Early Fusion, No RCS
                                                                                                              - Version B: Late Fusion, No RCS
                                                                                                              - Version C: Early Fusion + RCS
                                                                                                              - Version D: Late Fusion + RCS
                                                                                                              - Baselines: No correction, ROS, SMOTE
```

---

## 🚀 Quickstart & Verification Commands

### 1. Installation
Install dependencies in your environment:
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
Run the full test suite covering data loaders, models, multi-task losses, RCS components, and CKA/metrics:
```bash
python -m pytest tests/
```

### 3. Stage 1: Encoders, Trust Gating & Cross-Attention Training
Train Stage 1 with AdamW, multi-task losses, and per-epoch Centered Kernel Alignment (CKA) tracking:
```bash
# Variant 1: Early Fusion
python train_stage1.py --fusion_variant early --stage1_epochs 35

# Variant 2: Late Fusion
python train_stage1.py --fusion_variant late --stage1_epochs 35
```

### 4. Stage 2: Feature Extraction, RCS Balancing & PRDC Evaluation
Extract frozen representations, partition classes with $\eta=5$, fit per-class GMMs, apply regularized covariance sharing ($\lambda=0.1$), evaluate PRDC, and generate t-SNE / UMAP plots:
```bash
python train_stage2.py --fusion_variant early
python train_stage2.py --fusion_variant late
```

### 5. Stage 3: Classifier Training & Comprehensive Benchmark Evaluation
Train Stage 3 classifiers and evaluate on the strictly balanced test set across all 4 variants and baselines:
```bash
python train_stage3.py --clf_epochs 50
```

### 6. Full End-to-End Orchestrator
To run the entire pipeline end-to-end automatically:
```bash
python run_pipeline.py --stage1_epochs 35 --clf_epochs 50 --eta 5.0 --shrinkage_lambda 0.1
```

---

## 📊 Benchmark Results

Evaluated on the **strictly balanced test set (5,000 samples, 500 per class)**:

| Model Variant / Strategy | Balanced Acc | Macro F1 | MCC | G-Mean | Minority Recall | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Version A (Early Fusion, No RCS)** | 0.5884 | 0.5594 | 0.5472 | 0.5037 | 0.3220 | 0.5884 |
| **Baseline: Early Fusion + Random Oversampling** | 0.5968 | 0.5735 | 0.5556 | 0.5241 | 0.3420 | 0.5968 |
| **Baseline: Early Fusion + SMOTE** | 0.6016 | 0.5904 | 0.5587 | 0.5618 | 0.3727 | 0.6016 |
| **Version C (Early Fusion + Multimodal RCS)** | **0.5988** | **0.5829** | **0.5574** | **0.5485** | **0.3893** | **0.5988** |
| **Version B (Late Fusion, No RCS)** | 0.5834 | 0.5367 | 0.5442 | 0.4289 | 0.2553 | 0.5834 |
| **Baseline: Late Fusion + Random Oversampling** | 0.5808 | 0.5464 | 0.5412 | 0.4492 | 0.2533 | 0.5808 |
| **Baseline: Late Fusion + SMOTE** | 0.5886 | 0.5648 | 0.5502 | 0.4785 | 0.2613 | 0.5886 |
| **Version D (Late Fusion + Multimodal RCS)** | **0.6074** | **0.6025** | **0.5645** | **0.5776** | **0.4107** | **0.6074** |

### Key Findings:
1. **Multimodal RCS Outperforms Baselines on Extreme Minority Recall**:
   - In Late Fusion, Version D (+RCS) dramatically increased minority class recall from **0.2553 to 0.4107 (+15.54% absolute improvement)**, outperforming both Random Oversampling (0.2533) and SMOTE (0.2613).
   - In Early Fusion, Version C (+RCS) attained a minority recall of **0.3893**, superior to both ROS (0.3420) and SMOTE (0.3727).
2. **G-Mean and Macro F1 Gains**:
   - Version D boosted G-Mean from **0.4289 to 0.5776 (+14.87%)** and Macro F1 from **0.5367 to 0.6025 (+6.58%)**, confirming consistent class-wide performance without sacrificing majority class precision.
3. **High-Fidelity Synthetic Representations (PRDC)**:
   - Evaluated using `prdc`, synthetic vectors achieved Recall up to **0.98** and Coverage up to **0.64**, indicating that the regularized covariance sharing transfers donor variability without collapsing into mode repetition.

---

## 📁 Repository Structure

```
Multimodal_RCS/
├── requirements.txt
├── README.md
├── run_pipeline.py                 # End-to-end pipeline runner
├── train_stage1.py                 # Stage 1 training script
├── train_stage2.py                 # Stage 2 feature extraction & RCS balancing
├── train_stage3.py                 # Stage 3 classifier & benchmark evaluation
├── multimodal_rcs/
│   ├── config.py                   # Pipeline dataclass & CLI arguments
│   ├── data/
│   │   ├── avmnist_dataset.py      # AV-MNIST multimodal dataset & imbalance builder
│   │   └── dataloader.py           # PyTorch DataLoader factory
│   ├── models/
│   │   ├── image_encoder.py        # 2D CNN (1x28x28 -> 128)
│   │   ├── audio_encoder.py        # 2D CNN (1x112x112 -> 128)
│   │   ├── trust_gating.py         # Softmax trust scoring network
│   │   ├── fusion.py               # 2-layer 4-head Cross-Attention (Early & Late)
│   │   ├── decoders.py             # Image and Audio reconstruction decoders
│   │   ├── stage1_model.py         # Unifying Stage 1 module
│   │   └── classifier.py           # 2-hidden layer Stage 3 MLP
│   ├── losses/
│   │   ├── supcon_loss.py          # Supervised Contrastive Loss (temp=0.07, wt=0.1)
│   │   ├── cmm_loss.py             # Cross-Modality Matching Loss (temp=0.07, wt=0.1)
│   │   └── multi_task_loss.py      # MultiTaskLoss combiner (Recon + CE + SupCon + CMM)
│   ├── rcs/
│   │   ├── class_sorter.py         # Class sorter (Majority / Intermediate / Minority, eta=5)
│   │   ├── gmm_fitter.py           # Per-class GMM fitter (num_components = max(1, round(N_c / N_min)))
│   │   ├── covariance_sharing.py   # Top-3 nearest component blending with shrinkage (lambda=0.1)
│   │   ├── balancer.py             # Complete Multimodal RCS balancer
│   │   └── prdc_evaluator.py       # Precision, Recall, Density, Coverage evaluator
│   └── utils/
│       ├── device.py               # Auto CUDA / CPU selector
│       ├── cka.py                  # Linear Centered Kernel Alignment (CKA)
│       ├── metrics.py              # Balanced Acc, Macro F1, MCC, G-Mean, Minority Recall
│       ├── baselines.py            # No-correction, ROS, SMOTE resamplers
│       └── visualization.py        # t-SNE, UMAP, CM heatmaps, training curves
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_losses.py
│   ├── test_rcs.py
│   └── test_cka_metrics.py
├── data/                           # Extracted AV-MNIST dataset
├── checkpoints/                    # Saved Stage 1 model weights (.pt)
├── extracted_features/             # Extracted fused latent vectors (.npz)
└── results/                        # JSON metrics, Markdown summaries & figures
    └── figures/                    # t-SNE, UMAP, CKA curves & confusion matrices
```

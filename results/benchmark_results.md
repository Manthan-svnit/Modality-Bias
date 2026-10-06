# Multimodal RCS Benchmark Evaluation Results

| Model Variant / Strategy | Balanced Acc | Macro F1 | MCC | G-Mean | Minority Recall | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Version A (Early Fusion, No RCS)** | 0.5922 | 0.5647 | 0.5513 | 0.5124 | 0.3300 | 0.5922 |
| **Baseline: Early Fusion + Random Oversampling** | 0.5996 | 0.5789 | 0.5583 | 0.5349 | 0.3500 | 0.5996 |
| **Baseline: Early Fusion + SMOTE** | 0.6022 | 0.5902 | 0.5595 | 0.5621 | 0.3620 | 0.6022 |
| **Version C (Early Fusion + Multimodal RCS)** | 0.6048 | 0.5956 | 0.5623 | 0.5709 | 0.4147 | 0.6048 |
| **Version B (Late Fusion, No RCS)** | 0.5860 | 0.5402 | 0.5471 | 0.4385 | 0.2607 | 0.5860 |
| **Baseline: Late Fusion + Random Oversampling** | 0.5832 | 0.5476 | 0.5437 | 0.4481 | 0.2540 | 0.5832 |
| **Baseline: Late Fusion + SMOTE** | 0.5824 | 0.5588 | 0.5426 | 0.4877 | 0.2333 | 0.5824 |
| **Version D (Late Fusion + Multimodal RCS)** | 0.5986 | 0.5757 | 0.5580 | 0.5282 | 0.3433 | 0.5986 |
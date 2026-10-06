import os
import re
import json
import argparse
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW

from multimodal_rcs.config import PipelineConfig, add_pipeline_args
from multimodal_rcs.models import Stage3Classifier
from multimodal_rcs.utils import (
    get_device,
    evaluate_classifier_predictions,
    apply_no_correction,
    apply_random_oversampling,
    apply_smote,
    plot_confusion_matrix_heatmap,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrainStage3")


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: PipelineConfig,
    device: torch.device,
) -> Stage3Classifier:
    """Trains the 2-hidden layer Stage 3 classifier with AdamW."""
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.clf_batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.clf_batch_size, shuffle=False
    )

    classifier = Stage3Classifier(
        in_dim=X_train.shape[1],
        hidden1=config.clf_hidden1,
        hidden2=config.clf_hidden2,
        num_classes=config.num_classes,
        dropout=config.clf_dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        classifier.parameters(),
        lr=config.clf_lr,
        weight_decay=config.clf_weight_decay,
    )

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, config.clf_epochs + 1):
        classifier.train()
        for x_b, y_b in train_loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            logits = classifier(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

        # Validation
        classifier.eval()
        val_losses = []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b = x_b.to(device)
                y_b = y_b.to(device)
                logits = classifier(x_b)
                loss = criterion(logits, y_b)
                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}

    if best_state is not None:
        classifier.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return classifier


def evaluate_classifier_on_test(
    classifier: Stage3Classifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    minority_classes: Optional[list] = None,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Evaluates classifier on test representations and returns metrics & predictions."""
    classifier.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(X_test).float().to(device)
        logits = classifier(x_tensor)
        preds = logits.argmax(dim=-1).cpu().numpy()

    metrics = evaluate_classifier_predictions(
        y_true=y_test,
        y_pred=preds,
        minority_classes=minority_classes,
        num_classes=10,
    )
    return metrics, preds


def run_stage3(config: PipelineConfig):
    device = get_device(config.device)
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.figures_dir, exist_ok=True)

    # Dictionary to collect all evaluated models
    all_results = {}

    for variant in ["early", "late"]:
        raw_train_path = os.path.join(config.features_dir, f"train_raw_{variant}.npz")
        val_path = os.path.join(config.features_dir, f"val_{variant}.npz")
        test_path = os.path.join(config.features_dir, f"test_{variant}.npz")
        rcs_train_path = os.path.join(config.features_dir, f"train_balanced_rcs_{variant}.npz")
        prdc_path = os.path.join(config.results_dir, f"prdc_metrics_{variant}.json")

        if not os.path.exists(raw_train_path):
            logger.warning(f"Raw features for {variant} fusion not found at '{raw_train_path}'. Skipping {variant}.")
            continue

        raw_train = np.load(raw_train_path)
        val_data = np.load(val_path)
        test_data = np.load(test_path)

        X_train_raw, y_train_raw = raw_train["features"], raw_train["labels"]
        X_val, y_val = val_data["features"], val_data["labels"]
        X_test, y_test = test_data["features"], test_data["labels"]

        # Identify minority classes from PRDC partition or default [7, 8, 9]
        minority_classes = [7, 8, 9]
        if os.path.exists(prdc_path):
            with open(prdc_path, "r") as f:
                prdc_meta = json.load(f)
                minority_classes = prdc_meta.get("partitions", {}).get("minority", [7, 8, 9])

        # Define configurations to train for this fusion variant:
        # 1. No RCS (Version A or B)
        # 2. Random Oversampling (Baseline)
        # 3. SMOTE (Baseline)
        # 4. Multimodal RCS (Version C or D)
        exp_configs = {}

        version_letter_no_rcs = "Version A" if variant == "early" else "Version B"
        version_letter_rcs = "Version C" if variant == "early" else "Version D"

        # No correction
        X_no_rcs, y_no_rcs = apply_no_correction(X_train_raw, y_train_raw)
        exp_configs[f"{version_letter_no_rcs} ({variant.capitalize()} Fusion, No RCS)"] = (
            X_no_rcs,
            y_no_rcs,
        )

        # Random Oversampling baseline
        X_ros, y_ros = apply_random_oversampling(X_train_raw, y_train_raw, random_state=config.random_seed)
        exp_configs[f"Baseline: {variant.capitalize()} Fusion + Random Oversampling"] = (
            X_ros,
            y_ros,
        )

        # SMOTE baseline
        X_smote, y_smote = apply_smote(X_train_raw, y_train_raw, random_state=config.random_seed)
        exp_configs[f"Baseline: {variant.capitalize()} Fusion + SMOTE"] = (
            X_smote,
            y_smote,
        )

        # Multimodal RCS
        if os.path.exists(rcs_train_path):
            rcs_train = np.load(rcs_train_path)
            X_rcs, y_rcs = rcs_train["features"], rcs_train["labels"]
            exp_configs[f"{version_letter_rcs} ({variant.capitalize()} Fusion + Multimodal RCS)"] = (
                X_rcs,
                y_rcs,
            )
        else:
            logger.warning(f"RCS balanced features not found at '{rcs_train_path}'. Skipping RCS model.")

        for name, (X_tr, y_tr) in exp_configs.items():
            logger.info(f"--- Training Stage 3 Classifier for: {name} ---")
            clf = train_classifier(X_tr, y_tr, X_val, y_val, config, device)
            metrics, preds = evaluate_classifier_on_test(clf, X_test, y_test, device, minority_classes)

            all_results[name] = {
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "mcc": metrics["mcc"],
                "g_mean": metrics["g_mean"],
                "minority_recall": metrics["minority_recall"],
                "accuracy": metrics["accuracy"],
                "per_class_recall": metrics["per_class_recall"],
            }

            logger.info(
                f"[{name}] Balanced Acc: {metrics['balanced_accuracy']:.4f} | "
                f"Macro F1: {metrics['macro_f1']:.4f} | MCC: {metrics['mcc']:.4f} | "
                f"G-Mean: {metrics['g_mean']:.4f} | Minority Recall: {metrics['minority_recall']:.4f}"
            )

            # Plot confusion matrix
            clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name).strip('_')
            cm_save_path = os.path.join(
                config.figures_dir, f"cm_{clean_name}.png"
            )
            plot_confusion_matrix_heatmap(
                np.array(metrics["confusion_matrix"]),
                save_path=cm_save_path,
                title=f"Confusion Matrix: {name}",
            )

    # Save comprehensive results summary
    json_path = os.path.join(config.results_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Output formatted markdown table
    md_lines = [
        "# Multimodal RCS Benchmark Evaluation Results",
        "",
        "| Model Variant / Strategy | Balanced Acc | Macro F1 | MCC | G-Mean | Minority Recall | Accuracy |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m_name, res in all_results.items():
        md_lines.append(
            f"| **{m_name}** | {res['balanced_accuracy']:.4f} | {res['macro_f1']:.4f} | "
            f"{res['mcc']:.4f} | {res['g_mean']:.4f} | {res['minority_recall']:.4f} | {res['accuracy']:.4f} |"
        )
    md_lines.append("")

    md_path = os.path.join(config.results_dir, "benchmark_results.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    logger.info(f"Stage 3 evaluation finished. Saved results to '{json_path}' and '{md_path}'.")
    print("\n" + "\n".join(md_lines))

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Classifier Training and Benchmark Evaluation")
    parser = add_pipeline_args(parser)
    args = parser.parse_args()

    cfg = PipelineConfig(
        data_dir=args.data_dir,
        device=args.device,
        fusion_variant=args.fusion_variant,
        stage1_epochs=args.stage1_epochs,
        stage1_lr=args.stage1_lr,
        batch_size=args.batch_size,
        eta=args.eta,
        shrinkage_lambda=args.shrinkage_lambda,
        clf_epochs=args.clf_epochs,
        clf_lr=args.clf_lr,
        random_seed=args.random_seed,
    )
    run_stage3(cfg)

"""
run_stage3.py — Stage 3 Final Classifier & Comprehensive Evaluation.

Per PDF Specification:
"STAGE 3 — Final Classifier and Evaluation
 Goal: train the classifier on the balanced set and measure honestly whether the method helps the rare classes.
 Classifier: MLP on real + synthetic vectors, cross-entropy, AdamW.
 Final: retrain with best params over several seeds; report test metrics (mean, std) from that run only.
 Variants and baselines:
   Version A: Early cross-attention, RCS No
   Version B: Late cross-attention, RCS No
   Version C: Early cross-attention, RCS Yes (main method)
   Version D: Late cross-attention, RCS Yes
 Baselines: no correction, random oversampling, SMOTE-style interpolation on z_fused.
 Metrics: Balanced Accuracy, Macro F1, MCC, G-Mean, minority-class recall, per-class F1."
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

from stage3.evaluator import run_multi_seed_evaluation, evaluate_model, train_single_run
from stage3.metrics import compute_classification_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: Classifier & Evaluation")
    parser.add_argument(
        "--stage2_dir", required=True,
        help="Path to stage2_rcs directory or the parent experiment directory."
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
        help="Seeds for multi-seed retraining. Default: [42, 43, 44, 45, 46]"
    )
    parser.add_argument(
        "--eval_baselines", action="store_true", default=True,
        help="Also train and evaluate on baselines (NoCorrection, RandomOversample, SMOTE)."
    )
    return parser.parse_args()


def resolve_directories(input_dir: Path) -> Tuple[Path, Path, Path, Path]:
    """
    Resolve experiment directory, stage1 dir, stage2 dir, and stage3 dir.
    Returns (exp_dir, stage1_dir, stage2_dir, stage3_dir).
    """
    input_dir = input_dir.resolve()
    if input_dir.name == "stage2_rcs":
        stage2_dir = input_dir
        exp_dir = input_dir.parent
    elif (input_dir / "stage2_rcs").exists():
        stage2_dir = input_dir / "stage2_rcs"
        exp_dir = input_dir
    else:
        stage2_dir = input_dir
        exp_dir = input_dir

    stage1_dir = exp_dir / "stage1_representation"
    stage3_dir = exp_dir / "stage3_classifier"
    stage3_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir, stage1_dir, stage2_dir, stage3_dir


def load_array_safely(paths: List[Path]) -> Optional[np.ndarray]:
    for p in paths:
        if p.exists():
            return np.load(p)
    return None


def main():
    args = parse_args()
    input_path = Path(args.stage2_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    exp_dir, stage1_dir, stage2_dir, stage3_dir = resolve_directories(input_path)
    print(f"\n{'='*60}")
    print(f"[Stage 3] Final Classifier & Evaluation")
    print(f"  Exp Dir    : {exp_dir}")
    print(f"  Stage 2 Dir: {stage2_dir}")
    print(f"  Stage 3 Dir: {stage3_dir}")
    print(f"{'='*60}\n")

    # ── Load datasets ──────────────────────────────────────────────────────────
    z_bal = load_array_safely([stage2_dir / "z_fused_balanced.npy", stage2_dir / "features" / "z_fused_balanced.npy"])
    y_bal = load_array_safely([stage2_dir / "labels_balanced.npy", stage2_dir / "features" / "labels_balanced.npy"])

    z_real = load_array_safely([
        stage2_dir / "z_fused_real_train.npy",
        stage2_dir / "features" / "z_fused_real_train.npy",
        stage1_dir / "features" / "train" / "z_fused.npy",
        stage1_dir / "embeddings" / "train" / "z_fused.npy",
    ])
    y_real = load_array_safely([
        stage2_dir / "labels_real_train.npy",
        stage2_dir / "features" / "labels_real_train.npy",
        stage1_dir / "features" / "train" / "labels.npy",
        stage1_dir / "embeddings" / "train" / "labels.npy",
    ])

    z_val = load_array_safely([
        stage2_dir / "z_fused_val.npy",
        stage2_dir / "features" / "z_fused_val.npy",
        stage1_dir / "features" / "val" / "z_fused.npy",
        stage1_dir / "embeddings" / "val" / "z_fused.npy",
    ])
    y_val = load_array_safely([
        stage2_dir / "labels_val.npy",
        stage2_dir / "features" / "labels_val.npy",
        stage1_dir / "features" / "val" / "labels.npy",
        stage1_dir / "embeddings" / "val" / "labels.npy",
    ])

    z_test = load_array_safely([
        stage2_dir / "z_fused_test.npy",
        stage2_dir / "features" / "z_fused_test.npy",
        stage1_dir / "features" / "test" / "z_fused.npy",
        stage1_dir / "embeddings" / "test" / "z_fused.npy",
    ])
    y_test = load_array_safely([
        stage2_dir / "labels_test.npy",
        stage2_dir / "features" / "labels_test.npy",
        stage1_dir / "features" / "test" / "labels.npy",
        stage1_dir / "embeddings" / "test" / "labels.npy",
    ])

    if z_bal is None or y_bal is None:
        raise FileNotFoundError(f"Could not find balanced z_fused in {stage2_dir}")
    if z_val is None or y_val is None:
        raise FileNotFoundError(f"Could not find validation z_fused in {stage2_dir} or {stage1_dir}")

    # Determine num_classes and minority classes from metadata if available
    all_labels = np.concatenate([y_real if y_real is not None else y_bal, y_val])
    classes = sorted(np.unique(all_labels))
    num_classes = len(classes)

    meta_file = stage2_dir / "stage2_meta.json"
    minority_classes = []
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
            class_types = meta.get("class_types", {})
            minority_classes = [int(c) for c, t in class_types.items() if t == "minority"]
    if not minority_classes:
        # Fallback: classes with count below median
        counts = np.bincount(y_real if y_real is not None else y_bal, minlength=num_classes)
        med = np.median(counts)
        minority_classes = [c for c, cnt in enumerate(counts) if cnt < med]

    print(f"  Classes: {num_classes} | Minority classes: {minority_classes}")
    print(f"  Balanced train size: {len(z_bal)} (real: {len(z_real) if z_real is not None else 'N/A'})")
    print(f"  Val size: {len(z_val)} | Test size: {len(z_test) if z_test is not None else 'None (evaluating on val)'}")

    # ── Load hyperparameters from Optuna Stage B if available ─────────────────
    params: Dict[str, Any] = {
        "clf_hidden": 256,
        "clf_layers": 2,
        "clf_dropout": 0.2,
        "clf_lr": 1e-3,
        "clf_weight_decay": 1e-4,
        "clf_epochs": 35,
        "clf_batch_size": 64,
    }

    optuna_best_path = stage2_dir / "optuna" / "stage_b_best_params.json"
    if not optuna_best_path.exists():
        optuna_best_path = stage2_dir / "stage_b_best_params.json"
    if optuna_best_path.exists():
        print(f"\n[Stage 3] Loading tuned hyperparameters from: {optuna_best_path}")
        with open(optuna_best_path) as f:
            best_tuned = json.load(f)
            for k, v in best_tuned.items():
                if k.startswith("clf_"):
                    params[k] = v

    print(f"[Stage 3] Classifier training parameters: {params}")

    # ── Datasets to compare (Main Method + Variants + Baselines) ───────────────
    # Identify variant name per PDF:
    # A = Early, No RCS; B = Late, No RCS; C = Early + RCS (main); D = Late + RCS
    is_late = "late" in str(exp_dir).lower()
    main_variant_name = "Version D (Late + RCS)" if is_late else "Version C (Early + RCS, Main)"
    no_rcs_name = "Version B (Late, No RCS)" if is_late else "Version A (Early, No RCS)"

    train_variants: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        main_variant_name: (z_bal, y_bal),
    }

    if z_real is not None and y_real is not None:
        train_variants[no_rcs_name] = (z_real, y_real)

    if args.eval_baselines:
        base_dir = stage2_dir / "baselines"
        for b_name in ["random_oversample", "smote"]:
            bz_path = base_dir / f"{b_name}_z.npy"
            by_path = base_dir / f"{b_name}_y.npy"
            if bz_path.exists() and by_path.exists():
                train_variants[f"Baseline: {b_name}"] = (np.load(bz_path), np.load(by_path))

    # ── Multi-seed evaluation for each variant ─────────────────────────────────
    results_records = []
    per_class_records = []
    all_multi_seed_results = {}
    best_model_to_save = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for v_name, (v_z, v_y) in train_variants.items():
        print(f"\n[Stage 3] Running multi-seed evaluation for: {v_name} (train size: {len(v_z)})...")
        res = run_multi_seed_evaluation(
            z_train=v_z,
            y_train=v_y,
            z_val=z_val,
            y_val=y_val,
            z_test=z_test,
            y_test=y_test,
            num_classes=num_classes,
            params=params,
            seeds=args.seeds,
            minority_classes=minority_classes,
            device=device,
        )
        summary = res["summary"]
        all_multi_seed_results[v_name] = summary

        if v_name == main_variant_name:
            best_model_to_save = res["best_model"]

        # Build row for metrics.csv
        row = {
            "variant": v_name,
            "eval_split": summary["eval_split"],
            "balanced_acc_mean": summary["mean"]["balanced_accuracy"],
            "balanced_acc_std": summary["std"]["balanced_accuracy"],
            "macro_f1_mean": summary["mean"]["macro_f1"],
            "macro_f1_std": summary["std"]["macro_f1"],
            "mcc_mean": summary["mean"]["mcc"],
            "mcc_std": summary["std"]["mcc"],
            "g_mean_mean": summary["mean"]["g_mean"],
            "g_mean_std": summary["std"]["g_mean"],
            "minority_recall_mean": summary["mean"]["minority_recall"],
            "minority_recall_std": summary["std"]["minority_recall"],
            "accuracy_mean": summary["mean"]["accuracy"],
            "accuracy_std": summary["std"]["accuracy"],
        }
        results_records.append(row)

        # Per-class row
        pc_row = {"variant": v_name}
        for c, f1_stat in summary["per_class_f1_mean"].items():
            pc_row[f"class_{c}_f1_mean"] = f1_stat["mean"]
            pc_row[f"class_{c}_f1_std"] = f1_stat["std"]
        per_class_records.append(pc_row)

    # ── Save Results & Metrics CSV ─────────────────────────────────────────────
    metrics_df = pd.DataFrame(results_records)
    metrics_path = stage3_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n[Stage 3] Saved evaluation metrics to: {metrics_path}")
    print(metrics_df[["variant", "balanced_acc_mean", "macro_f1_mean", "g_mean_mean", "minority_recall_mean"]].to_string(index=False))

    per_class_df = pd.DataFrame(per_class_records)
    per_class_path = stage3_dir / "per_class_f1.csv"
    per_class_df.to_csv(per_class_path, index=False)

    with open(stage3_dir / "multi_seed_results.json", "w") as f:
        json.dump(all_multi_seed_results, f, indent=2)

    # ── Save Best Classifier Checkpoint ────────────────────────────────────────
    if best_model_to_save is not None:
        model_save_path = stage3_dir / "model.pt"
        torch.save(
            {
                "model_state": best_model_to_save.state_dict(),
                "params": params,
                "input_dim": z_bal.shape[1],
                "num_classes": num_classes,
            },
            model_save_path,
        )
        print(f"[Stage 3] Saved best classifier model to: {model_save_path}")

    # ── Save Baseline Comparison in results/<dataset>/baselines ─────────────────
    try:
        dataset_name = exp_dir.parent.parent.parent.name
        global_baselines_dir = exp_dir.parent.parent.parent / "baselines"
        global_baselines_dir.mkdir(parents=True, exist_ok=True)
        baseline_summary_path = global_baselines_dir / "comparison_summary.csv"

        metrics_copy = metrics_df.copy()
        metrics_copy["experiment"] = exp_dir.name
        metrics_copy["fusion_timing"] = exp_dir.parent.parent.name
        metrics_copy["contrastive_placement"] = exp_dir.parent.name

        if baseline_summary_path.exists():
            existing = pd.read_csv(baseline_summary_path)
            combined = pd.concat([existing, metrics_copy]).drop_duplicates(subset=["experiment", "variant"], keep="last")
            combined.to_csv(baseline_summary_path, index=False)
        else:
            metrics_copy.to_csv(baseline_summary_path, index=False)
        print(f"[Stage 3] Updated global baselines comparison at: {baseline_summary_path}")
    except Exception as e:
        print(f"[Stage 3] Global baselines logging skipped: {e}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots_dir = stage3_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Bar chart comparing variants
        fig, ax = plt.subplots(figsize=(10, 6))
        variants = metrics_df["variant"].tolist()
        x = np.arange(len(variants))
        width = 0.2

        ax.bar(x - 1.5*width, metrics_df["balanced_acc_mean"], width, yerr=metrics_df["balanced_acc_std"], capsize=4, label="Balanced Acc")
        ax.bar(x - 0.5*width, metrics_df["macro_f1_mean"], width, yerr=metrics_df["macro_f1_std"], capsize=4, label="Macro F1")
        ax.bar(x + 0.5*width, metrics_df["g_mean_mean"], width, yerr=metrics_df["g_mean_std"], capsize=4, label="G-Mean")
        ax.bar(x + 1.5*width, metrics_df["minority_recall_mean"], width, yerr=metrics_df["minority_recall_std"], capsize=4, label="Minority Recall")

        ax.set_ylabel("Score")
        ax.set_title("Stage 3 Performance Comparison across Variants & Baselines")
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=15, ha="right")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(plots_dir / "variant_comparison.png", dpi=150)
        plt.close()
        print(f"  Saved comparison plot to: {plots_dir / 'variant_comparison.png'}")
    except Exception as e:
        print(f"  [plots] Plot generation skipped ({e})")

    print(f"\n[Stage 3] Complete. All outputs saved to: {stage3_dir}\n")


if __name__ == "__main__":
    main()

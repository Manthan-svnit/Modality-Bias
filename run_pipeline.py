"""
run_pipeline.py — Master End-to-End Orchestrator for Multimodal RCS.

Integrates Stage 1 (Representation Learning), Stage 2 (RCS Balancing), and
Stage 3 (Final Classifier & Evaluation) into a unified, dataset-agnostic pipeline.

Usage
-----
# Full pipeline on any dataset:
python run_pipeline.py --config stage1/config/generic_dataset.yaml

# Run specific stages (e.g. fast iteration on Stage 2 and 3 using cached embeddings):
python run_pipeline.py --config stage1/config/generic_dataset.yaml --stages 2,3

# With hyperparameter tuning (Stage A and/or Stage B):
python run_pipeline.py --config stage1/config/generic_dataset.yaml --tune_stage_a --tune_stage_b

# Run full ablation matrix (2 fusion timings x 3 placements = 6 runs):
python run_pipeline.py --config stage1/config/generic_dataset.yaml --ablation

# CLI overrides:
python run_pipeline.py --config stage1/config/generic_dataset.yaml fusion_timing=late contrastive_placement=both
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from stage1.config.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal RCS Full Pipeline Orchestrator")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    parser.add_argument(
        "--stages", type=str, default="all",
        help="Comma-separated stages to run: 'all', '1', '2', '3', '1,2', '2,3'. Default: 'all'"
    )
    parser.add_argument(
        "--tune_stage_a", action="store_true",
        help="Run Optuna Stage A (encoder/fusion tuning) during Stage 1."
    )
    parser.add_argument(
        "--tune_stage_b", action="store_true",
        help="Run Optuna Stage B (RCS and classifier tuning) during Stage 2/3."
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run full ablation matrix: 2 fusion timings (early, late) x 3 contrastive placements."
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="Dot-notation overrides (e.g. training.lr=5e-4)."
    )
    return parser.parse_args()


def parse_overrides(override_list: List[str]):
    result = {}
    for item in override_list:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        if v == "true":
            v = True
        elif v == "false":
            v = False
        result[k] = v
    return result


def get_experiment_dir(cfg) -> Path:
    modality_tag = "_".join(sorted(cfg.modalities))
    return Path(cfg.results_root) / cfg.dataset_name / str(cfg.fusion_timing) / str(cfg.contrastive_placement) / modality_tag


def run_stage1_cmd(config_path: str, tune: bool, overrides: List[str]):
    cmd = [sys.executable, "-m", "stage1.run_stage1", "--config", str(config_path)]
    if tune:
        cmd.append("--tune")
    cmd.extend(overrides)
    print(f"\n>>> Running Stage 1: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def run_stage2_cmd(exp_dir: Path, tune: bool, rcs_eta: Optional[float] = None, rcs_k: Optional[int] = None, rcs_lam: Optional[float] = None):
    cmd = [sys.executable, "-m", "stage2.run_stage2", "--stage1_dir", str(exp_dir)]
    if tune:
        cmd.append("--tune")
    if rcs_eta is not None:
        cmd.extend(["--eta", str(rcs_eta)])
    if rcs_k is not None:
        cmd.extend(["--k", str(rcs_k)])
    if rcs_lam is not None:
        cmd.extend(["--lam", str(rcs_lam)])
    print(f"\n>>> Running Stage 2: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def run_stage3_cmd(exp_dir: Path, seeds: List[int] = (42, 43, 44, 45, 46)):
    cmd = [
        sys.executable, "-m", "stage3.run_stage3",
        "--stage2_dir", str(exp_dir),
        "--seeds", *[str(s) for s in seeds],
    ]
    print(f"\n>>> Running Stage 3: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def run_single_pipeline(config_path: str, stages: List[str], tune_a: bool, tune_b: bool, overrides: List[str]):
    parsed_ovr = parse_overrides(overrides)
    cfg = load_config(config_path, parsed_ovr)
    exp_dir = get_experiment_dir(cfg)

    print(f"\n{'='*70}")
    print(f"Executing Multimodal RCS Pipeline")
    print(f"Dataset      : {cfg.dataset_name}")
    print(f"Modalities   : {cfg.modalities}")
    print(f"Fusion Timing: {cfg.fusion_timing} | Placement: {cfg.contrastive_placement}")
    print(f"Experiment   : {exp_dir}")
    print(f"Stages       : {stages}")
    print(f"{'='*70}\n")

    # Stage 1: Representation Learning
    if "1" in stages:
        run_stage1_cmd(config_path, tune=tune_a, overrides=overrides)

    # Stage 2: RCS Balancing in representation space
    if "2" in stages:
        run_stage2_cmd(exp_dir, tune=tune_b)

    # Stage 3: Final Classifier and Multi-seed Evaluation
    if "3" in stages:
        run_stage3_cmd(exp_dir)

    print(f"\n[run_pipeline] Successfully completed pipeline for {exp_dir}")
    return exp_dir


def run_ablation_matrix(config_path: str, stages: List[str], tune_a: bool, tune_b: bool, base_overrides: List[str]):
    """
    Run 2 fusion timings (early, late) x 3 contrastive placements (modality_only, fused_only, both).
    Collect all Stage 3 results into a unified summary table.
    """
    timings = ["early", "late"]
    # Contrastive placement is strictly locked to 'both' (per-modality AND fused)
    placement = "both"
    all_exp_dirs = []

    print(f"\n{'='*70}")
    print(f"STARTING ABLATION MATRIX (Early vs Late Fusion Timing with locked Dual Contrastive)")
    print(f"{'='*70}\n")

    for t in timings:
        ovr = list(base_overrides) + [f"fusion_timing={t}", f"contrastive_placement={placement}"]
        exp_dir = run_single_pipeline(config_path, stages, tune_a, tune_b, ovr)
        all_exp_dirs.append(exp_dir)

    # Compile ablation summary
    summary_rows = []
    for exp_dir in all_exp_dirs:
        metrics_csv = exp_dir / "stage3_classifier" / "metrics.csv"
        if metrics_csv.exists():
            df = pd.read_csv(metrics_csv)
            df["fusion_timing"] = exp_dir.parent.parent.name
            df["contrastive_placement"] = exp_dir.parent.name
            summary_rows.append(df)

    if summary_rows:
        combined = pd.concat(summary_rows, ignore_index=True)
        dataset_dir = all_exp_dirs[0].parent.parent.parent
        summary_out = dataset_dir / "full_ablation_summary.csv"
        combined.to_csv(summary_out, index=False)
        print(f"\n{'='*70}")
        print(f"FULL ABLATION MATRIX COMPLETE!")
        print(f"Summary saved to: {summary_out}")
        print(combined[["fusion_timing", "contrastive_placement", "variant", "balanced_acc_mean", "macro_f1_mean", "g_mean_mean", "minority_recall_mean"]].to_string())
        print(f"{'='*70}\n")


def main():
    args = parse_args()
    stages_str = args.stages.lower().strip()
    if stages_str == "all":
        stages = ["1", "2", "3"]
    else:
        stages = [s.strip() for s in stages_str.split(",") if s.strip() in ("1", "2", "3")]

    if args.ablation:
        run_ablation_matrix(args.config, stages, args.tune_stage_a, args.tune_stage_b, args.overrides)
    else:
        run_single_pipeline(args.config, stages, args.tune_stage_a, args.tune_stage_b, args.overrides)


if __name__ == "__main__":
    main()

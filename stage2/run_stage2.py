"""
run_stage2.py — Stage 2 entry point.

Loads z_fused saved by Stage 1, runs RCS balancing, saves balanced set.
Can optionally run Optuna Stage B first to tune RCS + classifier params.

Usage
-----
# Basic (uses params from config):
python -m stage2.run_stage2 --stage1_dir results/<dataset>/<run_name>

# With Optuna Stage B first:
python -m stage2.run_stage2 --stage1_dir results/<dataset>/<run_name> --tune

# With CLI overrides:
python -m stage2.run_stage2 --stage1_dir results/<dataset>/<run_name> \
    rcs.eta=0.4 rcs.k=7 rcs.lambda=0.3
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from stage2.rcs import RCSBalancer, compute_prdc
from stage2.baselines import NoCorrection, RandomOversampler, SMOTEBalancer


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: RCS Balancing")
    parser.add_argument(
        "--stage1_dir", required=True,
        help="Path to a Stage 1 output dir containing embeddings/train/, embeddings/val/, etc."
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run Optuna Stage B to jointly tune RCS + classifier params before balancing."
    )
    parser.add_argument(
        "--eta", type=float, default=0.5,
        help="RCS class-size threshold fraction (0–1). Default: 0.5"
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="RCS k nearest GMM components for minority synthesis. Default: 5"
    )
    parser.add_argument(
        "--lam", type=float, default=0.5,
        help="RCS shrinkage lambda (0–1). Default: 0.5"
    )
    parser.add_argument(
        "--gmm_reg_covar", type=float, default=1e-6,
        help="GMM covariance regularisation. Default: 1e-6"
    )
    parser.add_argument(
        "--optuna_n_trials", type=int, default=50,
        help="Number of Optuna Stage B trials. Default: 50"
    )
    parser.add_argument(
        "--random_state", type=int, default=42
    )
    return parser.parse_args()


def resolve_stage1_dir(stage1_dir: Path) -> Tuple[Path, Path]:
    """
    Resolve stage1 directory and corresponding stage2 output directory.
    Returns (stage1_repr_dir, stage2_rcs_dir).
    """
    if (stage1_dir / "stage1_representation").exists():
        stage1_repr = stage1_dir / "stage1_representation"
        stage2_out = stage1_dir / "stage2_rcs"
    elif stage1_dir.name == "stage1_representation":
        stage1_repr = stage1_dir
        stage2_out = stage1_dir.parent / "stage2_rcs"
    else:
        stage1_repr = stage1_dir
        stage2_out = stage1_dir / "stage2_rcs"
    return stage1_repr, stage2_out


def load_split(stage1_repr_dir: Path, split: str):
    """Load z_fused.npy and labels.npy for a split (checking features/ or embeddings/)."""
    for folder in ("features", "embeddings"):
        candidate = stage1_repr_dir / folder / split
        if (candidate / "z_fused.npy").exists():
            z = np.load(candidate / "z_fused.npy")
            y = np.load(candidate / "labels.npy")
            return z, y
    raise FileNotFoundError(
        f"Could not find z_fused.npy in {stage1_repr_dir}/features/{split} or {stage1_repr_dir}/embeddings/{split}"
    )


def main():
    args = parse_args()
    raw_path = Path(args.stage1_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Stage 1 output dir not found: {raw_path}")

    stage1_dir, out_dir = resolve_stage1_dir(raw_path)

    # ── Load Stage 1 embeddings ────────────────────────────────────────────────
    print(f"\n[Stage 2] Loading Stage 1 embeddings from: {stage1_dir}")
    z_train, y_train = load_split(stage1_dir, "train")
    z_val,   y_val   = load_split(stage1_dir, "val")

    test_exists = any(
        (stage1_dir / f / "test" / "z_fused.npy").exists()
        for f in ("features", "embeddings")
    )
    z_test, y_test = (load_split(stage1_dir, "test") if test_exists else (None, None))

    print(f"  Train: {z_train.shape}  classes: {np.unique(y_train).tolist()}")
    print(f"  Val  : {z_val.shape}")
    if z_test is not None:
        print(f"  Test : {z_test.shape}")

    # Print class distribution
    classes, counts = np.unique(y_train, return_counts=True)
    print("\n  Class distribution (train):")
    for c, n in zip(classes, counts):
        print(f"    class {c}: {n} samples")

    # ── Optuna Stage B (optional) ──────────────────────────────────────────────
    rcs_params = {"eta": args.eta, "k": args.k, "lam": args.lam, "gmm_reg_covar": args.gmm_reg_covar}

    if args.tune:
        print("\n[Stage 2] Running Optuna Stage B...")
        from stage2.optuna_stage_b import OptunaStageB

        class _MockCfg:
            n_trials = args.optuna_n_trials
            study_db = f"sqlite:///{stage1_dir}/stage2_rcs/optuna/stage_b.db"

        tuner = OptunaStageB(
            z_train=z_train, y_train=y_train,
            z_val=z_val, y_val=y_val,
            stage_b_cfg=_MockCfg(),
            results_dir=str(stage1_dir / "stage2_rcs" / "optuna"),
        )
        best = tuner.run()

        # Apply best RCS params
        rcs_params["eta"]           = best.get("eta",           rcs_params["eta"])
        rcs_params["k"]             = best.get("k",             rcs_params["k"])
        rcs_params["lam"]           = best.get("lambda",        rcs_params["lam"])
        rcs_params["gmm_reg_covar"] = best.get("gmm_reg_covar", rcs_params["gmm_reg_covar"])
        print(f"\n[Stage 2] Using Optuna-tuned RCS params: {rcs_params}")
    
    # ── Stage 2 output directory ───────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── RCS Balancing ─────────────────────────────────────────────────────────
    print(f"\n[Stage 2] Running RCS with params: {rcs_params}")
    balancer = RCSBalancer(
        eta=rcs_params["eta"],
        k=rcs_params["k"],
        lam=rcs_params["lam"],
        gmm_reg_covar=rcs_params["gmm_reg_covar"],
        random_state=args.random_state,
    )
    z_bal, y_bal, report = balancer.fit_transform(z_train, y_train)

    # ── Print balanced distribution ────────────────────────────────────────────
    bal_classes, bal_counts = np.unique(y_bal, return_counts=True)
    print("\n  Balanced class distribution (train):")
    for c, n in zip(bal_classes, bal_counts):
        ct = report.get("class_types", {}).get(c, "?")
        print(f"    class {c}: {n} samples  ({ct})")

    print(f"\n  Total: {len(z_train)} -> {len(z_bal)} samples (added {len(z_bal)-len(z_train)})")

    # ── PRDC summary & CSV ────────────────────────────────────────────────────
    import pandas as pd
    if report.get("prdc"):
        print("\n  PRDC scores (synthetic quality):")
        for c, scores in report["prdc"].items():
            print(f"    class {c}: P={scores['precision']:.3f}  R={scores['recall']:.3f}  "
                  f"D={scores['density']:.3f}  C={scores['coverage']:.3f}")
        prdc_df = pd.DataFrame(report["prdc"]).T
        prdc_df.index.name = "class_id"
        prdc_df.to_csv(out_dir / "PRDC.csv")

    # ── Separate synthetic vectors ────────────────────────────────────────────
    # Real train vectors are the first len(z_train) elements
    synth_dir = out_dir / "synthetic_vectors"
    synth_dir.mkdir(exist_ok=True)
    if len(z_bal) > len(z_train):
        z_synth = z_bal[len(z_train):]
        y_synth = y_bal[len(z_train):]
    else:
        z_synth = np.zeros((0, z_train.shape[1]), dtype=z_train.dtype)
        y_synth = np.zeros((0,), dtype=y_train.dtype)
    np.save(synth_dir / "synthetic_z.npy", z_synth)
    np.save(synth_dir / "synthetic_y.npy", y_synth)

    # ── Save GMM parameters ───────────────────────────────────────────────────
    gmm_meta = {}
    for c, comps in balancer.gmm_components_.items():
        gmm_meta[str(c)] = [
            {
                "weight": float(comp.weight),
                "sample_count": int(comp.sample_count),
                "mean_norm": float(np.linalg.norm(comp.mean)),
            }
            for comp in comps
        ]
    with open(out_dir / "gmm_params.json", "w") as f:
        json.dump(gmm_meta, f, indent=2)

    # ── Save balanced training set ─────────────────────────────────────────────
    # Save both at root of stage2_rcs and in stage2_rcs/features for consistency
    for save_dir in [out_dir, out_dir / "features"]:
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "z_fused_balanced.npy",   z_bal)
        np.save(save_dir / "labels_balanced.npy",    y_bal)
        np.save(save_dir / "z_fused_real_train.npy", z_train)
        np.save(save_dir / "labels_real_train.npy",  y_train)
        np.save(save_dir / "z_fused_val.npy",        z_val)
        np.save(save_dir / "labels_val.npy",         y_val)
        if z_test is not None:
            np.save(save_dir / "z_fused_test.npy",   z_test)
            np.save(save_dir / "labels_test.npy",    y_test)

    # ── Save class type metadata ───────────────────────────────────────────────
    meta = {
        "rcs_params": rcs_params,
        "class_types": {str(k): v for k, v in report.get("class_types", {}).items()},
        "majority_size": int(balancer.majority_size_),
        "n_train_before": int(len(z_train)),
        "n_train_after":  int(len(z_bal)),
        "prdc": {
            str(c): {k: float(v) for k, v in sc.items()}
            for c, sc in report.get("prdc", {}).items()
        },
    }
    with open(out_dir / "stage2_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── 2D Plots (before vs after RCS) ────────────────────────────────────────
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots_dir = out_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Plot balanced train set (real vs synthetic)
        tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(z_bal)//4)), random_state=args.random_state)
        z_2d = tsne.fit_transform(z_bal)

        plt.figure(figsize=(8, 6))
        n_real = len(z_train)
        plt.scatter(z_2d[:n_real, 0], z_2d[:n_real, 1], c=y_bal[:n_real], cmap="tab10", alpha=0.7, label="Real", marker="o")
        if len(z_bal) > n_real:
            plt.scatter(z_2d[n_real:, 0], z_2d[n_real:, 1], c=y_bal[n_real:], cmap="tab10", alpha=0.9, label="RCS Synthetic", marker="^", edgecolors="k", linewidths=0.5)
        plt.title("Stage 2 RCS: Balanced Fused Space (t-SNE)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "tsne_balanced.png", dpi=150)
        plt.close()
        print(f"  Saved t-SNE plot to: {plots_dir / 'tsne_balanced.png'}")
    except Exception as e:
        print(f"  [plots] Plot generation skipped ({e})")

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("\n[Stage 2] Generating baseline balanced sets...")
    baselines_dir = out_dir / "baselines"
    baselines_dir.mkdir(exist_ok=True)

    for name, balancer_cls in [
        ("no_correction", NoCorrection()),
        ("random_oversample", RandomOversampler(random_state=args.random_state)),
        ("smote", SMOTEBalancer(k=rcs_params["k"], random_state=args.random_state)),
    ]:
        z_b, y_b, _ = balancer_cls.fit_transform(z_train, y_train)
        np.save(baselines_dir / f"{name}_z.npy", z_b)
        np.save(baselines_dir / f"{name}_y.npy", y_b)
        print(f"  {name}: {len(z_b)} samples")

    print(f"\n[Stage 2] Complete. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()

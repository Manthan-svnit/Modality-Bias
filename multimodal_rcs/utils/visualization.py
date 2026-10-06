import os
from typing import Optional, List, Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/terminal execution
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap


def plot_embedding_comparison(
    feats_before: np.ndarray,
    labels_before: np.ndarray,
    feats_after: np.ndarray,
    labels_after: np.ndarray,
    is_synthetic_mask: Optional[np.ndarray] = None,
    method: str = "tsne",
    title_suffix: str = "",
    save_path: str = "results/figures/tsne_comparison.png",
    random_state: int = 42,
    subsample_max: int = 2500,
):
    """Generates a side-by-side comparison plot of representations before and after oversampling.

    Args:
        feats_before: Features before balancing (N1, D)
        labels_before: Labels before balancing (N1,)
        feats_after: Features after balancing (N2, D)
        labels_after: Labels after balancing (N2,)
        is_synthetic_mask: (N2,) boolean array marking synthetic samples
        method: 'tsne' or 'umap'
        title_suffix: Extra description for title
        save_path: Output file path
        random_state: Random seed for projection
        subsample_max: Max samples to plot per pane for speed and readability
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Subsample if large
    if len(feats_before) > subsample_max:
        idx_b = np.random.RandomState(random_state).choice(len(feats_before), subsample_max, replace=False)
        feats_b, labels_b = feats_before[idx_b], labels_before[idx_b]
    else:
        feats_b, labels_b = feats_before, labels_before

    if len(feats_after) > subsample_max:
        idx_a = np.random.RandomState(random_state).choice(len(feats_after), subsample_max, replace=False)
        feats_a, labels_a = feats_after[idx_a], labels_after[idx_a]
        synth_a = is_synthetic_mask[idx_a] if is_synthetic_mask is not None else None
    else:
        feats_a, labels_a = feats_after, labels_after
        synth_a = is_synthetic_mask

    # Project with t-SNE or UMAP
    if method.lower() == "tsne":
        reducer_b = TSNE(n_components=2, random_state=random_state, perplexity=30)
        proj_b = reducer_b.fit_transform(feats_b)
        reducer_a = TSNE(n_components=2, random_state=random_state, perplexity=30)
        proj_a = reducer_a.fit_transform(feats_a)
        method_name = "t-SNE"
    else:
        reducer_b = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1)
        proj_b = reducer_b.fit_transform(feats_b)
        reducer_a = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1)
        proj_a = reducer_a.fit_transform(feats_a)
        method_name = "UMAP"

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    cmap = plt.get_cmap("tab10")

    # Pane 1: Before Oversampling
    scatter1 = axes[0].scatter(
        proj_b[:, 0],
        proj_b[:, 1],
        c=labels_b,
        cmap=cmap,
        alpha=0.75,
        s=18,
        edgecolors="none",
    )
    axes[0].set_title(f"{method_name}: Before Balancing (Imbalanced)\n{title_suffix}", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Component 1")
    axes[0].set_ylabel("Component 2")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    # Pane 2: After Oversampling
    if synth_a is not None:
        # Plot real samples
        real_mask = ~synth_a
        axes[1].scatter(
            proj_a[real_mask, 0],
            proj_a[real_mask, 1],
            c=labels_a[real_mask],
            cmap=cmap,
            alpha=0.75,
            s=18,
            label="Real",
            edgecolors="none",
        )
        # Plot synthetic samples with different marker
        axes[1].scatter(
            proj_a[synth_a, 0],
            proj_a[synth_a, 1],
            c=labels_a[synth_a],
            cmap=cmap,
            marker="^",
            alpha=0.85,
            s=28,
            edgecolors="black",
            linewidths=0.5,
            label="Synthetic (RCS)",
        )
        axes[1].legend(loc="upper right", frameon=True)
    else:
        axes[1].scatter(
            proj_a[:, 0],
            proj_a[:, 1],
            c=labels_a,
            cmap=cmap,
            alpha=0.75,
            s=18,
            edgecolors="none",
        )

    axes[1].set_title(f"{method_name}: After Balancing (Multimodal RCS)\n{title_suffix}", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Component 1")
    axes[1].set_ylabel("Component 2")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    # Colorbar
    cbar = fig.colorbar(scatter1, ax=axes, orientation="horizontal", fraction=0.046, pad=0.12)
    cbar.set_ticks(range(10))
    cbar.set_ticklabels([f"Digit {c}" for c in range(10)])
    # Adjust layout to accommodate bottom colorbar cleanly
    fig.subplots_adjust(bottom=0.18, wspace=0.15)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    save_path: str = "results/figures/confusion_matrix.png",
    title: str = "Confusion Matrix",
):
    """Plots and saves confusion matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=[f"D{i}" for i in range(cm.shape[1])],
        yticklabels=[f"D{i}" for i in range(cm.shape[0])],
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_stage1_metrics(
    history: Dict[str, List[float]],
    save_path: str = "results/figures/stage1_training_history.png",
):
    """Plots Stage 1 training loss curves and validation CKA progression."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["loss_total"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # 1. Losses
    axes[0].plot(epochs, history["loss_total"], label="Total Loss", linewidth=2, color="black")
    if "loss_cls" in history:
        axes[0].plot(epochs, history["loss_cls"], label="CE Loss (x1.0)", linestyle="--")
    if "loss_recon" in history:
        axes[0].plot(epochs, history["loss_recon"], label="Recon MSE (x1.0)", linestyle="--")
    if "loss_supcon" in history:
        axes[0].plot(epochs, history["loss_supcon"], label="SupCon (x0.1)", linestyle=":")
    if "loss_cmm" in history:
        axes[0].plot(epochs, history["loss_cmm"], label="CMM (x0.1)", linestyle=":")
    axes[0].set_title("Stage 1 Multi-Task Training Losses", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    # 2. CKA Alignment
    if "val_cka_fused_img" in history and "val_cka_fused_aud" in history:
        axes[1].plot(epochs, history["val_cka_fused_img"], label="CKA(Fused, Image)", color="#1f77b4", linewidth=2)
        axes[1].plot(epochs, history["val_cka_fused_aud"], label="CKA(Fused, Audio)", color="#ff7f0e", linewidth=2)
        axes[1].set_title("Validation Centered Kernel Alignment (CKA)", fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("CKA Score [0, 1]")
        axes[1].set_ylim([0.0, 1.05])
        axes[1].grid(True, linestyle="--", alpha=0.4)
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path

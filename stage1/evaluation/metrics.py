"""
metrics.py — Separation / quality metrics for z_fused embeddings.

Computed after every epoch (or on saved embeddings):
  1. KNN accuracy             — k-nearest-neighbour classification accuracy
  2. Linear probe accuracy    — logistic regression on frozen z_fused
  3. Silhouette score         — cluster separation quality
  4. CKA                      — Centred Kernel Alignment between z_fused and z_m
  5. t-SNE / UMAP projection  — 2D coordinates saved to disk
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import LabelEncoder
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

try:
    from sklearn.manifold import TSNE
    _TSNE_OK = True
except ImportError:
    _TSNE_OK = False

try:
    import umap
    _UMAP_OK = True
except ImportError:
    _UMAP_OK = False


# ──────────────────────────────────────────────────────────────────────────────
# KNN accuracy
# ──────────────────────────────────────────────────────────────────────────────

def knn_accuracy(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    k: int = 5,
) -> float:
    """
    K-nearest-neighbour accuracy on the validation set.

    Parameters
    ----------
    z_train, z_val : [N, d]  L2-normalised embeddings
    y_train, y_val : [N]     integer labels
    k              : number of neighbours

    Returns
    -------
    accuracy : float  (0–1)
    """
    if not _SKLEARN_OK:
        raise ImportError("scikit-learn is required for KNN evaluation.")
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    clf.fit(z_train, y_train)
    return float(clf.score(z_val, y_val))


# ──────────────────────────────────────────────────────────────────────────────
# Linear probe accuracy
# ──────────────────────────────────────────────────────────────────────────────

def linear_probe_accuracy(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    max_iter: int = 1000,
) -> float:
    """
    Logistic regression linear probe on frozen embeddings.

    Returns
    -------
    macro_f1 : float  (0–1)
    """
    if not _SKLEARN_OK:
        raise ImportError("scikit-learn is required for linear probe evaluation.")
    from sklearn.metrics import f1_score
    clf = LogisticRegression(max_iter=max_iter, C=1.0)
    clf.fit(z_train, y_train)
    preds = clf.predict(z_val)
    return float(f1_score(y_val, preds, average="macro", zero_division=0))


# ──────────────────────────────────────────────────────────────────────────────
# Silhouette score
# ──────────────────────────────────────────────────────────────────────────────

def compute_silhouette(z: np.ndarray, y: np.ndarray, sample_size: int = 5000) -> float:
    """
    Silhouette score (cosine metric).

    Sub-samples to ``sample_size`` if the dataset is large.

    Returns
    -------
    silhouette : float  (-1 to 1, higher is better)
    """
    if not _SKLEARN_OK:
        raise ImportError("scikit-learn is required for silhouette score.")
    if len(np.unique(y)) < 2:
        return float("nan")
    if len(z) > sample_size:
        idx = np.random.choice(len(z), sample_size, replace=False)
        z, y = z[idx], y[idx]
    return float(silhouette_score(z, y, metric="cosine"))


# ──────────────────────────────────────────────────────────────────────────────
# CKA (Centred Kernel Alignment)
# ──────────────────────────────────────────────────────────────────────────────

def _hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Unbiased estimate of HSIC (Gretton et al.)."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    KH = K @ H
    LH = L @ H
    return float(np.trace(KH @ LH)) / (n - 1) ** 2


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between representation matrices X [N, d1] and Y [N, d2].

    Returns
    -------
    cka : float  (0–1, 1 = identical structure)
    """
    K = X @ X.T
    L = Y @ Y.T
    hsic_kl = _hsic(K, L)
    hsic_kk = _hsic(K, K)
    hsic_ll = _hsic(L, L)
    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-12:
        return float("nan")
    return float(hsic_kl / denom)


def compute_cka(
    z_fused: np.ndarray,
    z_modalities: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Compute CKA between z_fused and each per-modality z_m.

    Returns
    -------
    dict: modality_name → CKA value
    """
    results = {}
    for mod, z_m in z_modalities.items():
        results[f"cka_{mod}"] = linear_cka(z_fused, z_m)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 2D projections: t-SNE and UMAP
# ──────────────────────────────────────────────────────────────────────────────

def compute_tsne(
    z: np.ndarray,
    perplexity: float = 30.0,
    n_iter: int = 1000,
) -> np.ndarray:
    """Returns [N, 2] t-SNE coordinates."""
    if not _TSNE_OK:
        raise ImportError("scikit-learn is required for t-SNE.")
    perp = min(perplexity, max(1.0, float(len(z) - 1)))
    try:
        tsne = TSNE(n_components=2, perplexity=perp, max_iter=n_iter, random_state=42)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=perp, n_iter=n_iter, random_state=42)
    return tsne.fit_transform(z)


def compute_umap(
    z: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """Returns [N, 2] UMAP coordinates."""
    if not _UMAP_OK:
        raise ImportError("umap-learn is required for UMAP.")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    )
    return reducer.fit_transform(z)


# ──────────────────────────────────────────────────────────────────────────────
# Embedding collector (run over a DataLoader)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_embeddings(
    model: torch.nn.Module,
    loader,
    modalities: List[str],
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Run the model in eval mode and collect:
      - z_fused [N, d]
      - z_m     {modality: [N, d]}
      - labels  [N]

    Returns
    -------
    z_fused  : np.ndarray [N, d]
    z_mod    : dict str → np.ndarray [N, d]
    labels   : np.ndarray [N]
    """
    model.eval()
    all_fused, all_labels = [], []
    all_mod: Dict[str, List[np.ndarray]] = {m: [] for m in modalities}

    for batch in loader:
        batch_gpu = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        out = model(batch_gpu)
        all_fused.append(out["z_fused"].cpu().numpy())
        all_labels.append(
            batch["label"].numpy() if isinstance(batch["label"], torch.Tensor)
            else np.array(batch["label"])
        )
        for m in modalities:
            all_mod[m].append(out["z"][m].cpu().numpy())

    z_fused = np.concatenate(all_fused, axis=0)
    labels  = np.concatenate(all_labels, axis=0)
    z_mod   = {m: np.concatenate(all_mod[m], axis=0) for m in modalities}

    return z_fused, z_mod, labels


# ──────────────────────────────────────────────────────────────────────────────
# Full evaluation suite
# ──────────────────────────────────────────────────────────────────────────────

def run_separation_metrics(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    z_mod_train: Dict[str, np.ndarray],
    z_mod_val: Dict[str, np.ndarray],
    eval_cfg,
) -> Dict[str, float]:
    """
    Compute all separation metrics and return as a flat dict.
    """
    results: Dict[str, float] = {}

    # KNN
    try:
        results["knn_acc"] = knn_accuracy(z_train, y_train, z_val, y_val, k=eval_cfg.knn_k)
    except Exception as e:
        results["knn_acc"] = float("nan")
        print(f"[metrics] KNN failed: {e}")

    # Linear probe
    try:
        results["linear_probe_macro_f1"] = linear_probe_accuracy(z_train, y_train, z_val, y_val)
    except Exception as e:
        results["linear_probe_macro_f1"] = float("nan")
        print(f"[metrics] Linear probe failed: {e}")

    # Silhouette on val
    try:
        results["silhouette"] = compute_silhouette(z_val, y_val)
    except Exception as e:
        results["silhouette"] = float("nan")
        print(f"[metrics] Silhouette failed: {e}")

    # CKA val (fused vs each modality)
    try:
        cka_vals = compute_cka(z_val, z_mod_val)
        results.update(cka_vals)
    except Exception as e:
        print(f"[metrics] CKA failed: {e}")

    return results


def save_2d_projections(
    z: np.ndarray,
    y: np.ndarray,
    save_dir: str,
    epoch: int,
    eval_cfg,
) -> None:
    """
    Compute t-SNE and/or UMAP and save coordinates as .npy files.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # t-SNE
    try:
        coords = compute_tsne(z, perplexity=eval_cfg.tsne_perplexity)
        np.save(save_dir / f"tsne_epoch{epoch:03d}.npy", coords)
        np.save(save_dir / f"labels_epoch{epoch:03d}.npy", y)
    except Exception as e:
        print(f"[metrics] t-SNE failed: {e}")

    # UMAP
    try:
        coords = compute_umap(
            z,
            n_neighbors=eval_cfg.umap_n_neighbors,
            min_dist=eval_cfg.umap_min_dist,
        )
        np.save(save_dir / f"umap_epoch{epoch:03d}.npy", coords)
    except Exception as e:
        print(f"[metrics] UMAP failed: {e}")

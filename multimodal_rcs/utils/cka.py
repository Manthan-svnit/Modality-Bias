import torch
import numpy as np
from typing import Union


def linear_cka(
    X: Union[torch.Tensor, np.ndarray],
    Y: Union[torch.Tensor, np.ndarray],
) -> float:
    """Computes Linear Centered Kernel Alignment (CKA) between representations X and Y.
    
    Formula:
        CKA(X, Y) = ||Y_c^T X_c||_F^2 / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F)
    where X_c, Y_c are centered representations.

    Args:
        X: Feature tensor/array of shape (N, D1)
        Y: Feature tensor/array of shape (N, D2)

    Returns:
        Scalar CKA similarity in [0, 1].
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    if isinstance(Y, np.ndarray):
        Y = torch.from_numpy(Y).float()

    X = X.float()
    Y = Y.float()

    assert X.shape[0] == Y.shape[0], f"Sample count mismatch: {X.shape[0]} vs {Y.shape[0]}"
    N = X.shape[0]
    if N <= 1:
        return 1.0

    # Center representations along sample dimension
    X_c = X - X.mean(dim=0, keepdim=True)
    Y_c = Y - Y.mean(dim=0, keepdim=True)

    # Compute cross-covariance and auto-covariance Gram matrices
    # Y_c^T X_c is (D2, D1)
    YtX = torch.matmul(Y_c.T, X_c)
    XtX = torch.matmul(X_c.T, X_c)
    YtY = torch.matmul(Y_c.T, Y_c)

    hsic_xy = torch.sum(YtX ** 2)
    hsic_xx = torch.sqrt(torch.sum(XtX ** 2))
    hsic_yy = torch.sqrt(torch.sum(YtY ** 2))

    denominator = hsic_xx * hsic_yy
    if denominator < 1e-12:
        return 0.0

    cka = (hsic_xy / denominator).item()
    return float(np.clip(cka, 0.0, 1.0))

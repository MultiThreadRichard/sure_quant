"""Stiefel-manifold optimizer utilities.

This module provides a light-weight projected-gradient optimizer that keeps
matrix variables on the Stiefel manifold:

    St(n, p) = {R in R^{n x p} | R^T R = I_p}
"""

from __future__ import annotations

import torch


class StiefelOptimizer:
    """Projected-gradient update on the Stiefel manifold.

    Args:
        lr: Learning rate for the gradient step before projection.
    """

    def __init__(self, lr: float):
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        self.lr = float(lr)

    def step(self, R: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """Apply one Stiefel-constrained update.

        The update is:
            Y = R - lr * grad
            R_new = qf(Y)   (Q from QR decomposition)

        where qf(.) is the orthonormal factor projection onto the manifold.

        Args:
            R: Current matrix (or batched matrices) with shape ``[..., n, p]``.
            grad: Gradient tensor with the same shape as ``R``.

        Returns:
            Projected matrix ``R_new`` on the Stiefel manifold.
        """
        if R.shape != grad.shape:
            raise ValueError(f"R and grad must have same shape, got {R.shape} vs {grad.shape}")
        if R.dim() < 2:
            raise ValueError(f"R must be at least 2D [..., n, p], got shape {R.shape}")

        Y = R - self.lr * grad

        # Reduced QR gives Q with orthonormal columns.
        Q, R_upper = torch.linalg.qr(Y, mode="reduced")

        # Sign correction for deterministic orientation.
        diag = torch.diagonal(R_upper, dim1=-2, dim2=-1)
        sign = torch.where(diag >= 0, torch.ones_like(diag), -torch.ones_like(diag))
        Q = Q * sign.unsqueeze(-2)
        return Q

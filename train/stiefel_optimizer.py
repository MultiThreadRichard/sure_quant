"""Stiefel-manifold utilities via Householder parameterization.

This module avoids per-step QR retraction by parameterizing orthogonal blocks
with k Householder reflectors.
"""

from __future__ import annotations

import torch


class StiefelOptimizer:
    """Gradient update for Householder reflectors.

    Args:
        lr: Learning rate for reflector parameter update.
    """

    def __init__(self, lr: float):
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        self.lr = float(lr)

    def step(self, V: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """Apply one update to reflector parameters.

        Args:
            V: Reflector parameters with shape ``[M, k, g]``.
            grad: Gradient tensor with same shape as ``V``.

        Returns:
            Updated reflector parameters ``V_new``.
        """
        if V.shape != grad.shape:
            raise ValueError(f"V and grad must have same shape, got {V.shape} vs {grad.shape}")
        if V.dim() != 3:
            raise ValueError(f"V must be 3D [M, k, g], got shape {V.shape}")
        return V - self.lr * grad


def apply_householder_batch(x: torch.Tensor, V: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Apply k Householder reflectors to batched block vectors.

    Args:
        x: Tensor with shape ``[B, M, g]``.
        V: Reflector parameters with shape ``[M, k, g]``.
        eps: Numerical stability epsilon for normalization.

    Returns:
        Rotated tensor with shape ``[B, M, g]``.
    """
    if x.dim() != 3:
        raise ValueError(f"x must be 3D [B, M, g], got {x.shape}")
    if V.dim() != 3:
        raise ValueError(f"V must be 3D [M, k, g], got {V.shape}")
    if x.shape[1] != V.shape[0] or x.shape[2] != V.shape[2]:
        raise ValueError(f"x and V shape mismatch, got x={x.shape}, V={V.shape}")

    y = x
    k = V.shape[1]
    for i in range(k):
        v = V[:, i, :]
        v = v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)
        dot = (y * v.unsqueeze(0)).sum(dim=-1, keepdim=True)
        y = y - 2.0 * dot * v.unsqueeze(0)
    return y


def reflectors_to_rotation_matrix(V: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert Householder reflectors to explicit block rotation matrices.

    Args:
        V: Reflector parameters with shape ``[M, k, g]``.
        eps: Numerical stability epsilon for normalization.

    Returns:
        Rotation matrices ``R`` with shape ``[M, g, g]``.
    """
    if V.dim() != 3:
        raise ValueError(f"V must be 3D [M, k, g], got {V.shape}")

    m, k, g = V.shape
    eye = torch.eye(g, device=V.device, dtype=V.dtype)
    R = eye.unsqueeze(0).repeat(m, 1, 1)

    for i in range(k):
        v = V[:, i, :]
        v = v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)
        vv_t = torch.einsum("mi,mj->mij", v, v)
        H = eye.unsqueeze(0) - 2.0 * vv_t
        R = torch.matmul(R, H)
    return R

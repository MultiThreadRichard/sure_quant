"""MSE reconstruction loss for rotation quantisation."""

import torch


def reconstruction_loss(x_blk: torch.Tensor, x_hat_blk: torch.Tensor) -> torch.Tensor:
    """Mean squared error between original and reconstructed block tensor.

    Args:
        x_blk: Original block tensor ``[N, M, g]``.
        x_hat_blk: Reconstructed block tensor ``[N, M, g]``.

    Returns:
        Scalar MSE loss.
    """
    return torch.mean((x_blk - x_hat_blk) ** 2)
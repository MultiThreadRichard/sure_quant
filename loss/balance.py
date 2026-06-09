"""Balance loss – encourages equal variance across coordinates within each block."""

import torch


def balance_loss(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Penalise variance imbalance across coordinates within a block.

    If some coordinates show much higher variance than others, quantization
    quality degrades because the common scale cannot capture both.

    Args:
        z: Rotated block tensor ``[N, M, g]``.
        eps: Small constant for numerical stability.

    Returns:
        Scalar balance loss.
    """
    var = torch.var(z, dim=0, unbiased=False)  # [M, g]
    var_mean = var.mean(dim=-1, keepdim=True)  # [M, 1]
    return ((var - var_mean) ** 2).mean()
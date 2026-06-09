"""Range loss – penalises large dynamic range across blocks.

Encourages blocks to have similar magnitude ranges so that a per‑block
uniform quantiser works well across all blocks.
"""

import torch


def range_loss(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Penalise excessive max‑to‑mean ratio across blocks.

    Args:
        z: Rotated block tensor ``[N, M, g]``.
        eps: Small constant for numerical stability.

    Returns:
        Scalar range loss.
    """
    block_max = z.abs().amax(dim=(0, 2))  # [M]
    block_mean = z.abs().mean(dim=(0, 2))  # [M]
    ratio = block_max / (block_mean + eps)
    return ratio.mean()
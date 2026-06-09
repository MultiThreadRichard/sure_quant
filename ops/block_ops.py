"""Blockify / deblockify utilities for block‑wise rotation quantization.

Converts between [N, D] and [N, M, g] tensors (M = D // block_size, g = block_size).
"""

import torch


def blockify(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Reshape a 2-D tensor into blocks.

    Args:
        x: Input tensor of shape ``[N, D]``.
        block_size: Block dimension ``g``. ``D`` must be divisible by ``g``.

    Returns:
        Tensor of shape ``[N, M, g]`` where ``M = D // g``.
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2D [N, D], got shape {x.shape}")
    n, d = x.shape
    if d % block_size != 0:
        raise ValueError(f"D={d} must be divisible by block_size={block_size}")
    m = d // block_size
    return x.contiguous().view(n, m, block_size)


def deblockify(x_blk: torch.Tensor) -> torch.Tensor:
    """Flatten a 3-D block tensor back to 2-D.

    Args:
        x_blk: Tensor of shape ``[N, M, g]``.

    Returns:
        Tensor of shape ``[N, D]`` where ``D = M * g``.
    """
    if x_blk.dim() != 3:
        raise ValueError(f"x_blk must be 3D [N, M, g], got shape {x_blk.shape}")
    n, m, g = x_blk.shape
    return x_blk.contiguous().view(n, m * g)
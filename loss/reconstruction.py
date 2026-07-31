"""Reconstruction objectives for rotation quantization."""

import torch
import torch.nn.functional as F


def reconstruction_loss(x_blk: torch.Tensor, x_hat_blk: torch.Tensor) -> torch.Tensor:
    """Mean squared error between original and reconstructed block tensor.

    Args:
        x_blk: Original block tensor ``[N, M, g]``.
        x_hat_blk: Reconstructed block tensor ``[N, M, g]``.

    Returns:
        Scalar MSE loss.
    """
    return torch.mean((x_blk - x_hat_blk) ** 2)


def kl_reconstruction_loss(
    x_blk: torch.Tensor,
    x_hat_blk: torch.Tensor,
    *,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL divergence between original and reconstructed block distributions.

    Signed values are represented as ``2g`` positive/negative squared-energy
    categories. This measures whether quantization preserves each block's
    sign and relative energy distribution. Computation is performed in
    float32 for mixed-precision stability.
    """
    if x_blk.shape != x_hat_blk.shape:
        raise ValueError(
            f"x_blk and x_hat_blk must have the same shape, got "
            f"{x_blk.shape} and {x_hat_blk.shape}"
        )
    if x_blk.dim() < 2:
        raise ValueError("KL reconstruction expects a block dimension")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    target = x_blk.float()
    reconstructed = x_hat_blk.float()

    def signed_energy_logits(value: torch.Tensor) -> torch.Tensor:
        energy = torch.cat(
            (value.clamp_min(0).square(), (-value).clamp_min(0).square()),
            dim=-1,
        )
        return energy.add(eps).log() / temperature

    target_logits = signed_energy_logits(target)
    reconstructed_logits = signed_energy_logits(reconstructed)
    target_probability = F.softmax(
        target_logits.detach(), dim=-1
    )
    reconstructed_log_probability = F.log_softmax(
        reconstructed_logits, dim=-1
    )
    divergence = F.kl_div(
        reconstructed_log_probability,
        target_probability,
        reduction="none",
    ).sum(dim=-1)
    return divergence.mean()

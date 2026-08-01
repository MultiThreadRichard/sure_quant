"""Total loss builder – weighted sum of all sub‑losses."""

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sure_quant.config.default_config import RotationQuantConfig


def build_total_loss(
    loss_rec: torch.Tensor,
    loss_dk: torch.Tensor,
    loss_bal: torch.Tensor,
    loss_range: torch.Tensor,
    cfg: "RotationQuantConfig",
) -> torch.Tensor:
    """Combine reconstruction, DKoleo, balance, and range losses.

    Args:
        loss_rec: Reconstruction (MSE) loss.
        loss_dk: DKoleo loss.
        loss_bal: Balance loss.
        loss_range: Range loss.
        cfg: Configuration with ``lambda_*`` weights.

    Returns:
        Scalar weighted total loss.
    """
    return (
        cfg.lambda_rec * loss_rec
        + cfg.lambda_dk * loss_dk
        + cfg.lambda_bal * loss_bal
        + cfg.lambda_range * loss_range
    )
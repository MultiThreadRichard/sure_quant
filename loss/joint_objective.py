"""Joint objective for quantization + DKoleo + balance regularization."""

from __future__ import annotations

import torch

from loss.balance import balance_loss
from loss.dkoleo import DKoleoLoss


class JointObjective:
    """Weighted joint objective used in Stiefel-constrained quantization.

    total_loss = lambda_q * L_q + lambda_d * L_d + lambda_b * L_b

    where:
        - L_q: quantization loss, MSE(z, qz)
        - L_d: DKoleo regularization on quantized representation qz
        - L_b: balance regularization on quantized representation qz

    Args:
        lambda_q: Weight of quantization MSE term.
        lambda_d: Weight of DKoleo term.
        lambda_b: Weight of balance term.
        dk_sample_size: Sub-sample size used by DKoleo.
        eps: Numerical stability epsilon for DKoleo.
    """

    def __init__(
        self,
        lambda_q: float,
        lambda_d: float,
        lambda_b: float,
        dk_sample_size: int = 128,
        eps: float = 1e-6,
    ):
        for name, value in (
            ("lambda_q", lambda_q),
            ("lambda_d", lambda_d),
            ("lambda_b", lambda_b),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        self.lambda_q = float(lambda_q)
        self.lambda_d = float(lambda_d)
        self.lambda_b = float(lambda_b)
        self.dkoleo = DKoleoLoss(eps=eps, sample_size=dk_sample_size)

    def compute(self, z: torch.Tensor, qz: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute joint objective and return each loss component.

        Args:
            z: Pre-quantization tensor, shape ``[N, M, g]``.
            qz: Quantized (or fake-quantized) tensor, shape ``[N, M, g]``.

        Returns:
            Dict with:
                - ``total_loss``: weighted scalar loss
                - ``loss_q``: quantization MSE term
                - ``loss_d``: DKoleo term
                - ``loss_b``: balance term
        """
        if z.shape != qz.shape:
            raise ValueError(f"z and qz must have same shape, got {z.shape} vs {qz.shape}")
        if z.dim() != 3:
            raise ValueError(f"z and qz must be 3D [N, M, g], got {z.shape}")

        loss_q = torch.mean((z - qz) ** 2)
        loss_d = self.dkoleo(qz)
        loss_b = balance_loss(qz)
        total_loss = self.lambda_q * loss_q + self.lambda_d * loss_d + self.lambda_b * loss_b
        return {
            "total_loss": total_loss,
            "loss_q": loss_q,
            "loss_d": loss_d,
            "loss_b": loss_b,
        }

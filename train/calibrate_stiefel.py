"""Stiefel-constrained calibration via Householder parameterization.

This module implements a standalone training loop that optimizes block-wise
orthogonal transforms with k Householder reflectors using the joint objective:

    L = lambda_q * L_q + lambda_d * L_d + lambda_b * L_b

where L_q is quantization MSE, L_d is DKoleo regularization, and L_b is
balance regularization.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from config.default_config import SureQuantConfig
from loss.joint_objective import JointObjective
from ops.block_ops import blockify
from quant.fake_quant import BlockUniformQuantizer
from train.stiefel_optimizer import (
    StiefelOptimizer,
    apply_householder_batch,
    reflectors_to_rotation_matrix,
)


def calibrate_stiefel(
    sample_tensor: torch.Tensor,
    cfg: SureQuantConfig,
) -> Dict[str, torch.Tensor | List[Dict[str, float]]]:
    """Train block-wise rotation matrices with Stiefel constraints.

    Args:
        sample_tensor: Calibration tensor of shape ``[N, D]``.
        cfg: SureQuantConfig for block size, bits, learning rate and steps.

    Returns:
        Dict containing:
            - ``rotations``: learned block rotations ``[M, g, g]``
            - ``reflectors``: learned Householder reflectors ``[M, k, g]``
            - ``logs``: per-step training logs
    """
    if sample_tensor.dim() != 2:
        raise ValueError(f"sample_tensor must be 2D [N, D], got {sample_tensor.shape}")

    device = torch.device(cfg.device)
    x_all = sample_tensor.to(device)
    g = cfg.block_size
    n, d = x_all.shape
    if d % g != 0:
        raise ValueError(f"D={d} must be divisible by block_size={g}")

    m = d // g
    k = int(getattr(cfg, "stiefel_num_reflectors", g))
    if k <= 0:
        raise ValueError(f"stiefel_num_reflectors must be positive, got {k}")

    # Block-wise Householder reflectors [M, k, g].
    V = torch.randn(m, k, g, device=device, dtype=x_all.dtype).requires_grad_(True)

    stiefel_opt = StiefelOptimizer(lr=cfg.calibration_lr)
    objective = JointObjective(
        lambda_q=cfg.lambda_rec,
        lambda_d=cfg.lambda_dk,
        lambda_b=cfg.lambda_bal,
        dk_sample_size=cfg.dk_sample_size,
    )
    quantizer = BlockUniformQuantizer(cfg.num_bits)

    logs: List[Dict[str, float]] = []

    for step in range(cfg.calibration_steps):
        if n > cfg.calibration_batch_size:
            idx = torch.randperm(n, device=device)[: cfg.calibration_batch_size]
            x = x_all[idx]
        else:
            x = x_all

        xb = blockify(x, g)  # [B, M, g]
        z = apply_householder_batch(xb, V)
        qz, _ = quantizer(z)

        loss_info = objective.compute(z, qz)
        loss = loss_info["total_loss"]
        loss_q = loss_info["loss_q"]
        loss_d = loss_info["loss_d"]
        loss_b = loss_info["loss_b"]

        if V.grad is not None:
            V.grad.zero_()
        loss.backward()

        with torch.no_grad():
            V_new = stiefel_opt.step(V, V.grad)
            V.copy_(V_new)
            V.grad.zero_()

        logs.append(
            {
                "step": float(step),
                "loss": float(loss.item()),
                "loss_q": float(loss_q.item()),
                "loss_d": float(loss_d.item()),
                "loss_b": float(loss_b.item()),
            }
        )

        if step % 100 == 0 or step == cfg.calibration_steps - 1:
            print(
                f"[{step:4d}/{cfg.calibration_steps}] "
                f"stiefel_total={loss.item():.6f} "
                f"loss_q={loss_q.item():.6f} "
                f"loss_d={loss_d.item():.6f} "
                f"loss_b={loss_b.item():.6f}"
            )

    with torch.no_grad():
        rotations = reflectors_to_rotation_matrix(V)

    return {
        "rotations": rotations.detach(),
        "reflectors": V.detach(),
        "logs": logs,
    }

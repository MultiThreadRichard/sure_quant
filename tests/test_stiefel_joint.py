"""Tests for StiefelOptimizer and JointObjective."""

import torch

from train.stiefel_optimizer import StiefelOptimizer
from loss.joint_objective import JointObjective


def test_stiefel_optimizer_keeps_orthonormal_columns():
    n, p = 16, 8
    R0, _ = torch.linalg.qr(torch.randn(n, p), mode="reduced")
    grad = torch.randn(n, p)

    opt = StiefelOptimizer(lr=1e-2)
    R1 = opt.step(R0, grad)

    eye = torch.eye(p)
    assert R1.shape == (n, p)
    assert torch.allclose(R1.T @ R1, eye, atol=1e-5)


def test_stiefel_optimizer_supports_batch_input():
    b, n, p = 4, 12, 6
    R0, _ = torch.linalg.qr(torch.randn(b, n, p), mode="reduced")
    grad = torch.randn(b, n, p)

    opt = StiefelOptimizer(lr=5e-3)
    R1 = opt.step(R0, grad)

    eye = torch.eye(p).expand(b, p, p)
    gram = torch.matmul(R1.transpose(-1, -2), R1)
    assert torch.allclose(gram, eye, atol=1e-5)


def test_joint_objective_runs_and_has_gradient():
    z = torch.randn(32, 4, 8, requires_grad=True)
    qz = z + 0.05 * torch.randn_like(z)

    obj = JointObjective(lambda_q=1.0, lambda_d=0.1, lambda_b=0.01, dk_sample_size=32)
    loss_info = obj.compute(z, qz)
    loss = loss_info["total_loss"]
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()
    assert torch.isfinite(loss_info["loss_q"]).item()
    assert torch.isfinite(loss_info["loss_d"]).item()
    assert torch.isfinite(loss_info["loss_b"]).item()

    loss.backward()
    assert z.grad is not None
    assert z.grad.abs().sum() > 0


def test_joint_objective_reduces_when_quant_error_reduces():
    z = torch.randn(64, 2, 8)
    qz_bad = z + 0.5 * torch.randn_like(z)
    qz_good = z + 0.05 * torch.randn_like(z)

    obj = JointObjective(lambda_q=1.0, lambda_d=0.0, lambda_b=0.0)
    l_bad = obj.compute(z, qz_bad)["total_loss"]
    l_good = obj.compute(z, qz_good)["total_loss"]

    assert l_good.item() < l_bad.item()

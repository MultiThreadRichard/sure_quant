"""Tests for StiefelOptimizer and JointObjective."""

import torch

from train.stiefel_optimizer import (
    StiefelOptimizer,
    apply_householder_batch,
    reflectors_to_rotation_matrix,
)
from loss.joint_objective import JointObjective


def test_stiefel_optimizer_updates_reflectors_shape():
    m, k, g = 4, 6, 8
    V0 = torch.randn(m, k, g)
    grad = torch.randn(m, k, g)

    opt = StiefelOptimizer(lr=1e-2)
    V1 = opt.step(V0, grad)

    assert V1.shape == (m, k, g)
    assert not torch.allclose(V0, V1)


def test_householder_rotation_is_orthogonal_and_matches_batch_apply():
    b, m, k, g = 5, 3, 4, 8
    x = torch.randn(b, m, g)
    V = torch.randn(m, k, g)

    y1 = apply_householder_batch(x, V)
    R = reflectors_to_rotation_matrix(V)
    y2 = torch.einsum("bmg,mgk->bmk", x, R)

    gram = torch.matmul(R.transpose(-1, -2), R)
    eye = torch.eye(g).unsqueeze(0).expand_as(gram)

    assert torch.allclose(gram, eye, atol=1e-5)
    assert torch.allclose(y1, y2, atol=1e-5)


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

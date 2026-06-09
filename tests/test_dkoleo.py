"""Tests for DKoleo loss module."""

import torch

from loss.dkoleo import DKoleoLoss


def test_dkoleo_runs():
    """DKoleo produces a finite scalar."""
    z = torch.randn(128, 4, 16)
    loss_fn = DKoleoLoss(sample_size=64)
    loss = loss_fn(z)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_dkoleo_reduces_with_more_dispersion():
    """Pushing points apart (multiplying by a larger constant) should reduce loss."""
    z = torch.randn(64, 2, 8)
    loss_fn = DKoleoLoss(sample_size=64)

    loss_tight = loss_fn(z)  # points are close together
    loss_wide = loss_fn(z * 10.0)  # points are far apart

    assert loss_wide.item() < loss_tight.item()


def test_dkoleo_small_sample():
    """Works with fewer samples than sample_size."""
    z = torch.randn(10, 3, 8)
    loss_fn = DKoleoLoss(sample_size=64)
    loss = loss_fn(z)
    assert torch.isfinite(loss).item()


def test_dkoleo_deterministic_for_small_n():
    """When n <= sample_size, no sub‑sampling → deterministic."""
    torch.manual_seed(42)
    z = torch.randn(32, 2, 8)
    loss_fn = DKoleoLoss(sample_size=64)
    l1 = loss_fn(z)
    l2 = loss_fn(z)
    assert torch.allclose(l1, l2)


def test_dkoleo_gradient():
    """Gradients flow through DKoleo."""
    z = torch.randn(16, 2, 4, requires_grad=True)
    loss_fn = DKoleoLoss(sample_size=16)
    loss = loss_fn(z)
    loss.backward()
    assert z.grad is not None
    assert z.grad.abs().sum() > 0


def test_dkoleo_zero_context():
    """Identical vectors give a finite (negative log small) loss."""
    z = torch.ones(8, 1, 4)
    loss_fn = DKoleoLoss(sample_size=8)
    loss = loss_fn(z)
    assert torch.isfinite(loss).item()
"""Tests for block‑wise learnable Givens rotations."""

import pytest
import torch

from ops.givens import BlockGivensRotation, build_butterfly_pairs


def test_givens_inverse():
    """Round‑trip through Givens and inverse returns original."""
    x = torch.randn(12, 3, 8)
    mod = BlockGivensRotation(block_size=8, num_blocks=3)
    y = mod(x)
    xr = mod.inverse(y)
    assert torch.allclose(x, xr, atol=1e-5, rtol=1e-5)


def test_givens_learnable():
    """Gradients flow through Givens parameters."""
    x = torch.randn(4, 2, 4, requires_grad=True)
    mod = BlockGivensRotation(block_size=4, num_blocks=2)
    # Set non‑zero theta
    mod.theta.data = torch.randn_like(mod.theta.data) * 0.1

    y = mod(x)
    loss = y.sum()
    loss.backward()

    assert mod.theta.grad is not None
    assert mod.theta.grad.abs().sum() > 0


def test_givens_orthogonal_identity():
    """With theta=0, Givens is identity."""
    mod = BlockGivensRotation(block_size=8, num_blocks=2)
    # theta defaults to zero
    x = torch.randn(10, 2, 8)
    y = mod(x)
    assert torch.allclose(x, y, atol=1e-6)


def test_givens_shape_mismatch():
    """Raises on shape mismatch."""
    mod = BlockGivensRotation(block_size=8, num_blocks=3)
    with pytest.raises(ValueError):
        mod(torch.randn(10, 4, 8))
    with pytest.raises(ValueError):
        mod(torch.randn(10, 3, 16))


def test_build_butterfly_pairs():
    """Butterfly pairs cover all coordinates."""
    g = 8
    pairs = build_butterfly_pairs(g)
    seen = set()
    for p, q in pairs:
        assert 0 <= p < g
        assert 0 <= q < g
        assert p < q
        seen.add(p)
        seen.add(q)
    # Every coordinate should appear in at least one pair
    assert seen == set(range(g))


def test_givens_custom_pairs():
    """Accepts custom pair topology."""
    pairs = [(0, 2), (1, 3)]
    mod = BlockGivensRotation(block_size=4, num_blocks=1, pairs=pairs)
    assert mod.num_pairs == 2
    x = torch.randn(5, 1, 4)
    y = mod(x)
    xr = mod.inverse(y)
    assert torch.allclose(x, xr, atol=1e-5)


def test_givens_det_preserving():
    """A Givens matrix has det=1; composite should also."""
    mod = BlockGivensRotation(block_size=4, num_blocks=1)
    mod.theta.data = torch.randn_like(mod.theta.data) * 0.3
    # Build the effective matrix: apply all pairs to basis vectors.
    # Each row of x_tiled is a basis vector e_i.
    x_tiled = torch.eye(4).unsqueeze(1)  # [4, 1, 4] — 4 "vectors", each is e_i
    y = mod(x_tiled)  # [4, 1, 4] — each vec is the i-th column of the matrix
    matrix = y[:, 0, :]  # [4, 4]
    det = torch.linalg.det(matrix)
    assert torch.allclose(det, torch.tensor(1.0), atol=1e-5)
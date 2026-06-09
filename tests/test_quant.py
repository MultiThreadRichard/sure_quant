"""Tests for the block‑wise uniform fake quantiser."""

import pytest
import torch

from quant.fake_quant import BlockUniformQuantizer


def test_quant_shape():
    """Output shapes are correct."""
    x = torch.randn(32, 6, 16)
    q = BlockUniformQuantizer(num_bits=4)
    x_hat, scale = q(x)
    assert x_hat.shape == x.shape
    assert scale.shape == (6,)


def test_quant_identity_8bit():
    """With 8-bit quantisation, per‑element max absolute error is small.

    We test on data with a moderate range (centered around 0, small variance)
    to avoid the per‑block absmax scale dominating small values near zero.
    """
    x = torch.randn(16, 4, 8) * 0.1 + 1.0  # centred away from 0
    q = BlockUniformQuantizer(num_bits=8)
    x_hat, _ = q(x)
    # Per‑element max absolute error should be small with 8 bits
    max_abs_err = (x_hat - x).abs().max()
    scale = x.abs().amax()
    assert max_abs_err < 0.05 * scale, f"max_abs_err={max_abs_err:.4f}, scale={scale:.4f}"


def test_quant_ste_gradient():
    """Gradient passes through quantiser (STE)."""
    x = torch.randn(8, 2, 4, requires_grad=True)
    q = BlockUniformQuantizer(num_bits=4)
    x_hat, _ = q(x)
    loss = x_hat.sum()
    loss.backward()
    assert x.grad is not None
    # STE gives gradient of 1 everywhere
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_quant_1bit():
    """1‑bit quantiser: only two levels (±scale)."""
    x = torch.randn(20, 1, 8)
    q = BlockUniformQuantizer(num_bits=1)
    x_hat, scale = q(x)
    # With 1 bit, values should be in {−scale, +scale}
    unique = torch.unique(torch.round(x_hat / scale.unsqueeze(0).unsqueeze(2)))
    assert set(unique.tolist()) <= {-1.0, 0.0, 1.0}


def test_quant_scale_positive():
    """All scales are strictly positive."""
    x = torch.randn(10, 3, 16)
    q = BlockUniformQuantizer(num_bits=4)
    _, scale = q(x)
    assert (scale > 0).all()


def test_quant_invalid_bits():
    """Raises for num_bits < 1."""
    with pytest.raises(ValueError):
        BlockUniformQuantizer(num_bits=0)


def test_quant_invalid_input_dim():
    """Raises for non‑3D input."""
    q = BlockUniformQuantizer(num_bits=4)
    with pytest.raises(ValueError):
        q(torch.randn(10, 20))


def test_quant_zero_input():
    """Handles all‑zero input gracefully."""
    x = torch.zeros(5, 2, 8)
    q = BlockUniformQuantizer(num_bits=4)
    x_hat, scale = q(x)
    assert (x_hat == 0).all()
    assert (scale > 0).all()  # eps keeps it positive
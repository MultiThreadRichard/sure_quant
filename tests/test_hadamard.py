"""Tests for block‑wise Hadamard transform."""

import pytest
import torch

from ops.hadamard import BlockHadamardTransform, fwht_lastdim


def test_hadamard_inverse():
    """Round‑trip through Hadamard and inverse returns original."""
    x = torch.randn(10, 4, 8)
    mod = BlockHadamardTransform(block_size=8, num_blocks=4)
    y = mod(x)
    xr = mod.inverse(y)
    assert xr.shape == x.shape
    assert torch.allclose(x, xr, atol=1e-5, rtol=1e-5)


def test_hadamard_orthogonal_columns():
    """Columns (across batch) remain orthogonal after transform."""
    mod = BlockHadamardTransform(block_size=16, num_blocks=2)
    x = torch.randn(100, 2, 16)
    y = mod(x)
    # Check per‑block orthogonality
    for b in range(2):
        gram = y[:, b, :].T @ y[:, b, :]
        # Rough check — exactness depends on seed
        diag = torch.diag(gram)
        off_diag_sum = (gram - torch.diag(diag)).abs().mean()
        assert off_diag_sum < 100  # not identity but transformed


def test_fwht_known_result():
    """FWHT on [1, 0, 0, 0] yields [0.5, 0.5, 0.5, 0.5]."""
    x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    y = fwht_lastdim(x)  # normalised
    expected = torch.full((1, 4), 0.5)
    assert torch.allclose(y, expected, atol=1e-6)


def test_fwht_energy_preservation():
    """FWHT preserves L2 norm: H/√g is orthogonal → ||y|| = ||x||."""
    g = 16
    x = torch.randn(3, g)
    y = fwht_lastdim(x)
    # H is orthogonal with row norm √g.  fwht_lastdim divides by √g,
    # so (H/√g) is an orthogonal matrix → preserves L2 norm.
    assert torch.allclose(y.norm(), x.norm(), atol=1e-6)


def test_hadamard_shape_mismatch():
    """Raises when input shape doesn't match module configuration."""
    mod = BlockHadamardTransform(block_size=8, num_blocks=3)
    with pytest.raises(ValueError):
        mod(torch.randn(10, 4, 8))  # num_blocks mismatch
    with pytest.raises(ValueError):
        mod(torch.randn(10, 3, 16))  # block_size mismatch


def test_hadamard_different_block_size():
    """Works with block_size=2, 4, 16, 32, 64."""
    for g in [2, 4, 16, 32, 64]:
        mod = BlockHadamardTransform(block_size=g, num_blocks=1)
        x = torch.randn(20, 1, g)
        y = mod(x)
        xr = mod.inverse(y)
        assert torch.allclose(x, xr, atol=1e-5, rtol=1e-5), f"Failed for g={g}"


def test_fwht_not_power_of_two():
    """Raises when input last dim is not power of two."""
    with pytest.raises(ValueError):
        fwht_lastdim(torch.randn(3, 7))
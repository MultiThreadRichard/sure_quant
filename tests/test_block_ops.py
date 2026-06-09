"""Tests for block_ops: blockify / deblockify."""

import pytest
import torch

from ops.block_ops import blockify, deblockify


def test_blockify_deblockify():
    """Round‑trip preserves values."""
    x = torch.randn(8, 16)
    xb = blockify(x, 4)
    assert xb.shape == (8, 4, 4)
    xr = deblockify(xb)
    assert xr.shape == x.shape
    assert torch.allclose(x, xr)


def test_blockify_large():
    """Works with larger dimensions."""
    x = torch.randn(32, 128)
    xb = blockify(x, 16)
    assert xb.shape == (32, 8, 16)
    xr = deblockify(xb)
    assert torch.allclose(x, xr)


def test_blockify_dvalues_after_roundtrip():
    """Values are a simple reshape — check random values survive."""
    x = torch.randn(5, 12) * 100
    xb = blockify(x, 3)
    xr = deblockify(xb)
    assert torch.equal(x, xr)


def test_blockify_invalid_input_dim():
    """Rejects non‑2D input."""
    with pytest.raises(ValueError):
        blockify(torch.randn(2, 3, 4), 2)


def test_blockify_invalid_divisible():
    """Rejects D not divisible by block_size."""
    with pytest.raises(ValueError):
        blockify(torch.randn(4, 10), 3)


def test_deblockify_invalid_input_dim():
    """Rejects non‑3D input."""
    with pytest.raises(ValueError):
        deblockify(torch.randn(2, 3))


def test_blockify_single_block():
    """Works when block_size == D."""
    x = torch.randn(4, 8)
    xb = blockify(x, 8)
    assert xb.shape == (4, 1, 8)
    xr = deblockify(xb)
    assert torch.allclose(x, xr)
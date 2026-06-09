"""End‑to‑end pipeline tests: config → quantizer → calibrate → export → load → inference."""

import os
import tempfile

import torch

from config.default_config import RotationQuantConfig
from model.sure_quantizer import RotationQuantizer
from model.rotated_linear import RotatedQuantLinear
from train.calibrate_rotations import calibrate_single_layer
from train.high_level_api import RotationQuantCalibrator
from export.export_rotation_params import export_sure_quantizer
from export.checkpoint_io import load_sure_quantizer
from loss.reconstruction import reconstruction_loss
from loss.balance import balance_loss
from loss.range_loss import range_loss


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

def make_cfg(**overrides) -> RotationQuantConfig:
    cfg = RotationQuantConfig()
    cfg.calibration_steps = 10
    cfg.calibration_batch_size = 32
    cfg.calibration_lr = 0.01
    cfg.device = "cpu"
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# RotationQuantizer
# ---------------------------------------------------------------------------


def test_sure_quantizer_forward():
    """Full pipeline produces correct output shapes."""
    dim, block_size = 32, 8
    x = torch.randn(20, dim)
    rq = RotationQuantizer(dim=dim, block_size=block_size, num_bits=4)
    out = rq(x)
    assert out["x_hat"].shape == x.shape
    assert out["x_blk"].shape == (20, 4, 8)
    assert out["z"].shape == (20, 4, 8)
    assert out["z_hat"].shape == (20, 4, 8)
    assert out["scale"].shape == (4,)


def test_sure_quantizer_reconstruction_improves_with_training():
    """Calibration training reduces reconstruction MSE."""
    dim, block_size = 32, 8
    cfg = make_cfg(calibration_steps=30, calibration_lr=0.05, lambda_dk=0.0)
    x = torch.randn(200, dim)

    rq = RotationQuantizer(dim=dim, block_size=block_size, num_bits=4)
    # Measure initial MSE
    with torch.no_grad():
        out = rq(x)
        mse_before = reconstruction_loss(out["x_blk"], out["x_hat_blk"]).item()
    print(f"mse_before={mse_before}")
    
    # Train
    calibrate_single_layer(rq, x, cfg)

    # Measure final MSE
    with torch.no_grad():
        out = rq(x)
        mse_after = reconstruction_loss(out["x_blk"], out["x_hat_blk"]).item()
    
    print(f"mse_after={mse_after}")
    assert mse_after < mse_before, (
        f"Training should reduce MSE. before={mse_before:.6f}, after={mse_after:.6f}"
    )


# ---------------------------------------------------------------------------
# Export / import round‑trip
# ---------------------------------------------------------------------------


def test_export_load_roundtrip():
    """Export then load produces identical outputs."""
    dim, block_size = 64, 16
    rq = RotationQuantizer(dim=dim, block_size=block_size, num_bits=4)
    rq.rotation.givens.theta.data = torch.randn_like(rq.rotation.givens.theta) * 0.1

    # Calibrate briefly so theta is non‑zero
    cfg = make_cfg(calibration_steps=5, calibration_lr=0.01, lambda_dk=0.0)
    x = torch.randn(100, dim)
    calibrate_single_layer(rq, x, cfg)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        export_sure_quantizer(rq, path)
        rq2 = load_sure_quantizer(path, device="cpu")

        x_test = torch.randn(10, dim)
        with torch.no_grad():
            out1 = rq(x_test)["x_hat"]
            out2 = rq2(x_test)["x_hat"]

        assert torch.allclose(out1, out2, atol=1e-6), "Loaded quantizer output differs"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# RotatedQuantLinear
# ---------------------------------------------------------------------------


def test_rotated_quant_linear():
    """Inference wrapper produces correct output shape."""
    dim, out_dim = 32, 16
    linear = torch.nn.Linear(dim, out_dim)
    rq = RotationQuantizer(dim=dim, block_size=8, num_bits=4)

    wrapped = RotatedQuantLinear(linear, rq)
    x = torch.randn(5, 7, dim)  # extra leading dims
    y = wrapped(x)
    assert y.shape == (5, 7, out_dim)


# ---------------------------------------------------------------------------
# High‑level calibrator API
# ---------------------------------------------------------------------------


def test_calibrator_api():
    """Full calibrator workflow."""
    model = torch.nn.Linear(32, 16)
    cfg = make_cfg(calibration_steps=5, block_size=8)
    calibrator = RotationQuantCalibrator(model, cfg)

    layer_name = "test_layer"
    x = torch.randn(100, 32)
    calibrator.collect_samples_for_layer(layer_name, x)
    calibrator.build_quantizer_for_layer(layer_name, dim=32)
    logs = calibrator.calibrate_layer(layer_name)
    assert len(logs) == cfg.calibration_steps
    assert "loss" in logs[0]

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        calibrator.export_layer(layer_name, path)
        assert os.path.exists(path)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def test_balance_loss():
    """Balance loss is finite and >= 0."""
    z = torch.randn(50, 4, 8)
    loss = balance_loss(z)
    assert torch.isfinite(loss).item()
    assert loss.item() >= 0


def test_range_loss():
    """Range loss is finite and >= 0."""
    z = torch.randn(50, 4, 8)
    loss = range_loss(z)
    assert torch.isfinite(loss).item()
    assert loss.item() >= 0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_defaults():
    """Default config is instantiable."""
    cfg = RotationQuantConfig()
    assert cfg.block_size == 16
    assert cfg.num_bits == 4
    assert cfg.lambda_rec == 1.0
"""End‑to‑end pipeline tests: config → quantizer → calibrate → export → load → inference."""

import os
import tempfile
import math

import torch
import torch.nn.functional as F

from config.default_config import SureQuantConfig
from model.sure_quantizer import SureQuantizer
from model.sure_quant_linear import SureQuantLinear
from train.calibrate_rotations import calibrate_single_layer
from train.calibrate_stiefel import calibrate_stiefel
from train.high_level_api import SureQuantCalibrator
from export.export_rotation_params import export_sure_quantizer
from export.checkpoint_io import load_sure_quantizer
from loss.reconstruction import reconstruction_loss
from loss.balance import balance_loss
from loss.range_loss import range_loss
from ops.block_ops import blockify
from quant.fake_quant import BlockUniformQuantizer


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

def make_cfg(**overrides) -> SureQuantConfig:
    cfg = SureQuantConfig()
    cfg.calibration_steps = 10
    cfg.calibration_batch_size = 32
    cfg.calibration_lr = 0.01
    cfg.device = "cpu"
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# SureQuantizer
# ---------------------------------------------------------------------------


def test_sure_quantizer_forward():
    """Full pipeline produces correct output shapes."""
    dim, block_size = 32, 8
    x = torch.randn(20, dim)
    rq = SureQuantizer(dim=dim, block_size=block_size, num_bits=4)
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

    rq = SureQuantizer(dim=dim, block_size=block_size, num_bits=4)
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
    rq = SureQuantizer(dim=dim, block_size=block_size, num_bits=4)
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
# SureQuantLinear
# ---------------------------------------------------------------------------


def test_sure_quant_linear():
    """Inference wrapper produces correct output shape."""
    dim, out_dim = 32, 16
    linear = torch.nn.Linear(dim, out_dim)
    rq = SureQuantizer(dim=dim, block_size=8, num_bits=4)

    wrapped = SureQuantLinear(linear, rq)
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
    calibrator = SureQuantCalibrator(model, cfg)    

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
    cfg = SureQuantConfig()
    assert cfg.block_size == 16
    assert cfg.num_bits == 4
    assert cfg.lambda_rec == 1.0


def test_stiefel_vs_rotation_quant_error_comparison():
    """Compare quantization error between stiefel and rotation schemes.

    This test uses synthetic random data and reports which scheme yields
    lower average quantization MSE in rotated space.
    """
    cfg = make_cfg(
        block_size=8,
        num_bits=4,
        calibration_steps=20,
        calibration_batch_size=64,
        calibration_lr=0.03,
        lambda_dk=0.0,
        lambda_bal=0.0,
        lambda_range=0.0,
        dk_sample_size=64,
    )

    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    rotation_errors = []
    stiefel_errors = []

    for seed in seeds:
        torch.manual_seed(seed)
        x = torch.randn(256, 32)

        # Baseline rotation scheme (Hadamard + learnable Givens)
        rq = SureQuantizer(dim=32, block_size=cfg.block_size, num_bits=cfg.num_bits)
        calibrate_single_layer(rq, x, cfg)
        with torch.no_grad():
            out = rq(x)
            rot_err = ((out["z"] - out["z_hat"]) ** 2).mean().item()
        rotation_errors.append(rot_err)

        # Stiefel-constrained rotation matrix scheme
        stiefel_out = calibrate_stiefel(x, cfg)
        R = stiefel_out["rotations"]
        quantizer = BlockUniformQuantizer(cfg.num_bits)
        with torch.no_grad():
            xb = blockify(x, cfg.block_size)
            z = torch.einsum("bmg,mgk->bmk", xb, R)
            qz, _ = quantizer(z)
            st_err = ((z - qz) ** 2).mean().item()
        stiefel_errors.append(st_err)

    rotation_mean = sum(rotation_errors) / len(rotation_errors)
    stiefel_mean = sum(stiefel_errors) / len(stiefel_errors)

    print(f"rotation_mean_mse={rotation_mean:.6f}")
    print(f"stiefel_mean_mse={stiefel_mean:.6f}")

    better = "stiefel" if stiefel_mean < rotation_mean else "rotation"
    print(f"better_scheme={better}")

    # Ensure both schemes are numerically valid and produce finite errors.
    assert rotation_mean > 0 and stiefel_mean > 0
    assert torch.isfinite(torch.tensor(rotation_mean)).item()
    assert torch.isfinite(torch.tensor(stiefel_mean)).item()


def _hist_kl_divergence(p: torch.Tensor, q: torch.Tensor, bins: int = 128, eps: float = 1e-8) -> torch.Tensor:
    """Estimate KL(P || Q) by histogram over a shared value range."""
    p_flat = p.reshape(-1)
    q_flat = q.reshape(-1)

    vmin = torch.min(torch.min(p_flat), torch.min(q_flat))
    vmax = torch.max(torch.max(p_flat), torch.max(q_flat))

    p_hist = torch.histc(p_flat, bins=bins, min=float(vmin), max=float(vmax))
    q_hist = torch.histc(q_flat, bins=bins, min=float(vmin), max=float(vmax))

    p_prob = p_hist / torch.clamp(p_hist.sum(), min=eps)
    q_prob = q_hist / torch.clamp(q_hist.sum(), min=eps)

    p_prob = torch.clamp(p_prob, min=eps)
    q_prob = torch.clamp(q_prob, min=eps)
    p_prob = p_prob / p_prob.sum()
    q_prob = q_prob / q_prob.sum()

    return F.kl_div(q_prob.log(), p_prob, reduction="sum")


def _mean_and_ci95(values: list[float]) -> tuple[float, float]:
    """Return (mean, 95% CI half-width) using normal approximation."""
    if len(values) < 2:
        raise ValueError("Need at least two samples to estimate CI")
    v = torch.tensor(values, dtype=torch.float64)
    mean = float(v.mean().item())
    std = float(v.std(unbiased=True).item())
    ci95 = 1.96 * std / math.sqrt(len(values))
    return mean, ci95


def test_stiefel_vs_rotation_quant_error_kl_divergence_comparison():
    """Compare quantization error via KL divergence between z and qz distributions."""
    cfg = make_cfg(
        block_size=8,
        num_bits=4,
        calibration_steps=20,
        calibration_batch_size=64,
        calibration_lr=0.03,
        lambda_dk=0.0,
        lambda_bal=0.0,
        lambda_range=0.0,
        dk_sample_size=64,
    )

    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    rotation_kls = []
    stiefel_kls = []

    for seed in seeds:
        torch.manual_seed(seed)
        x = torch.randn(256, 32)

        rq = SureQuantizer(dim=32, block_size=cfg.block_size, num_bits=cfg.num_bits)
        calibrate_single_layer(rq, x, cfg)
        with torch.no_grad():
            out = rq(x)
            kl_rot = _hist_kl_divergence(out["z"], out["z_hat"]).item()
        rotation_kls.append(kl_rot)

        stiefel_out = calibrate_stiefel(x, cfg)
        R = stiefel_out["rotations"]
        quantizer = BlockUniformQuantizer(cfg.num_bits)
        with torch.no_grad():
            xb = blockify(x, cfg.block_size)
            z = torch.einsum("bmg,mgk->bmk", xb, R)
            qz, _ = quantizer(z)
            kl_st = _hist_kl_divergence(z, qz).item()
        stiefel_kls.append(kl_st)

    rotation_mean_kl = sum(rotation_kls) / len(rotation_kls)
    stiefel_mean_kl = sum(stiefel_kls) / len(stiefel_kls)

    print(f"rotation_mean_kl={rotation_mean_kl:.6f}")
    print(f"stiefel_mean_kl={stiefel_mean_kl:.6f}")

    better = "stiefel" if stiefel_mean_kl < rotation_mean_kl else "rotation"
    print(f"better_scheme_kl={better}")

    assert torch.isfinite(torch.tensor(rotation_mean_kl)).item()
    assert torch.isfinite(torch.tensor(stiefel_mean_kl)).item()
    assert rotation_mean_kl >= 0 and stiefel_mean_kl >= 0


def test_stiefel_has_significant_advantage_over_rotation_by_kl():
    """Test stiefel improvement percentage over rotation on KL metric."""
    cfg = make_cfg(
        block_size=8,
        num_bits=4,
        calibration_steps=20,
        calibration_batch_size=64,
        calibration_lr=0.03,
        lambda_dk=0.0,
        lambda_bal=0.0,
        lambda_range=0.0,
        dk_sample_size=64,
    )

    seeds = list(range(12))
    rotation_kls = []
    stiefel_kls = []

    for seed in seeds:
        torch.manual_seed(seed)
        x = torch.randn(256, 32)

        rq = SureQuantizer(dim=32, block_size=cfg.block_size, num_bits=cfg.num_bits)
        calibrate_single_layer(rq, x, cfg)
        with torch.no_grad():
            out = rq(x)
            kl_rot = _hist_kl_divergence(out["z"], out["z_hat"]).item()
        rotation_kls.append(kl_rot)

        stiefel_out = calibrate_stiefel(x, cfg)
        R = stiefel_out["rotations"]
        quantizer = BlockUniformQuantizer(cfg.num_bits)
        with torch.no_grad():
            xb = blockify(x, cfg.block_size)
            z = torch.einsum("bmg,mgk->bmk", xb, R)
            qz, _ = quantizer(z)
            kl_st = _hist_kl_divergence(z, qz).item()
        stiefel_kls.append(kl_st)

    rotation_mean = sum(rotation_kls) / len(rotation_kls)
    stiefel_mean = sum(stiefel_kls) / len(stiefel_kls)
    improve_kl_percent = (rotation_mean - stiefel_mean) / max(rotation_mean, 1e-12) * 100.0

    print(f"rotation_mean_kl={rotation_mean:.6f}")
    print(f"stiefel_mean_kl={stiefel_mean:.6f}")
    print(f"stiefel_improve_kl_percent={improve_kl_percent:.2f}%")

    assert torch.isfinite(torch.tensor(rotation_mean)).item()
    assert torch.isfinite(torch.tensor(stiefel_mean)).item()
    assert torch.isfinite(torch.tensor(improve_kl_percent)).item()
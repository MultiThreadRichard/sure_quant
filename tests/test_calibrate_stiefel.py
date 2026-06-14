"""Tests for Stiefel-constrained calibration loop."""

import torch

from config.default_config import SureQuantConfig
from train.calibrate_stiefel import calibrate_stiefel


def test_calibrate_stiefel_runs_and_returns_shapes():
    cfg = SureQuantConfig()
    cfg.device = "cpu"
    cfg.block_size = 8
    cfg.num_bits = 4
    cfg.calibration_steps = 5
    cfg.calibration_batch_size = 32
    cfg.calibration_lr = 1e-2
    cfg.dk_sample_size = 16

    x = torch.randn(64, 32)
    out = calibrate_stiefel(x, cfg)

    rotations = out["rotations"]
    logs = out["logs"]

    assert rotations.shape == (4, 8, 8)
    assert len(logs) == cfg.calibration_steps
    assert "loss" in logs[0]
    assert "loss_q" in logs[0]
    assert "loss_d" in logs[0]
    assert "loss_b" in logs[0]


def test_calibrate_stiefel_keeps_orthogonality():
    cfg = SureQuantConfig()
    cfg.device = "cpu"
    cfg.block_size = 8
    cfg.calibration_steps = 3
    cfg.calibration_batch_size = 16

    x = torch.randn(32, 16)
    out = calibrate_stiefel(x, cfg)
    rotations = out["rotations"]

    gram = torch.matmul(rotations.transpose(-1, -2), rotations)
    eye = torch.eye(cfg.block_size).unsqueeze(0).expand_as(gram)
    assert torch.allclose(gram, eye, atol=1e-5)

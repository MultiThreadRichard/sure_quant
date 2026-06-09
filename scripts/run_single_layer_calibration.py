#!/usr/bin/env python3
"""Single‑layer rotation quantisation calibration script.

Usage::

    python scripts/run_single_layer_calibration.py [--dim 4096] [--steps 500] ...

Or import ``main()`` for programmatic use.
"""

import argparse

import torch

from sure_quant.config.default_config import RotationQuantConfig
from sure_quant.model.sure_quantizer import RotationQuantizer
from sure_quant.train.calibrate_rotations import calibrate_single_layer
from sure_quant.export.export_rotation_params import export_sure_quantizer


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate a single‑layer rotation quantizer")
    p.add_argument("--dim", type=int, default=4096, help="Input dimension D")
    p.add_argument("--block-size", type=int, default=16, help="Block size g")
    p.add_argument("--num-bits", type=int, default=4, help="Quantisation bits")
    p.add_argument("--steps", type=int, default=500, help="Calibration steps")
    p.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=256, help="Calibration batch size")
    p.add_argument("--lambda-dk", type=float, default=0.05, help="DKoleo weight")
    p.add_argument("--n-samples", type=int, default=2048, help="Number of calibration samples")
    p.add_argument("--output", type=str, default="layer_sure_quant.pt", help="Export path")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Build config
    cfg = RotationQuantConfig()
    cfg.block_size = args.block_size
    cfg.num_bits = args.num_bits
    cfg.calibration_steps = args.steps
    cfg.calibration_lr = args.lr
    cfg.calibration_batch_size = args.batch_size
    cfg.lambda_dk = args.lambda_dk
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {cfg.device}")
    print(f"Configuration: {cfg}")

    # Generate synthetic calibration data
    # (in production, replace with real layer activations)
    print(f"Generating {args.n_samples} synthetic samples of dim={args.dim} ...")
    x = torch.randn(args.n_samples, args.dim, device=cfg.device)

    # Build quantizer
    rq = RotationQuantizer(
        dim=args.dim,
        block_size=cfg.block_size,
        num_bits=cfg.num_bits,
        order=cfg.order,
    ).to(cfg.device)

    # Calibrate
    print(f"Starting calibration ({cfg.calibration_steps} steps) ...")
    logs = calibrate_single_layer(rq, x, cfg)

    # Export
    export_sure_quantizer(rq, args.output)
    print(f"Exported to {args.output}")

    # Final report
    final = logs[-1]
    print(f"\nFinal: rec={final['loss_rec']:.6f}  dk={final['loss_dk']:.4f}  "
          f"bal={final['loss_bal']:.4f}  rng={final['loss_rng']:.4f}")


if __name__ == "__main__":
    main()
"""Configuration for rotation quantization training and inference."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SureQuantConfig:
    """Central configuration for the sure quantization framework.

    Recommended defaults for first release:
        block_size=16, num_bits=4, num_givens_layers=2, lambda_dk=0.05.
        If calibration is unstable, set lambda_dk=0 first, verify reconstruction
        loss converges, then gradually re‑enable DKoleo.
    """

    # --- Block & quantization ---
    block_size: int = 16
    num_bits: int = 4

    # --- Givens rotation ---
    givens_pairs_strategy: str = "butterfly"
    num_givens_layers: int = 2
    num_pairs_per_layer: int = 8

    # --- Composite rotation ---
    order: str = "hadamard_givens"  # "hadamard_givens" | "givens_hadamard"

    # --- Loss weights ---
    lambda_rec: float = 1.0
    lambda_dk: float = 0.05
    lambda_bal: float = 0.01
    lambda_range: float = 0.01
    lambda_orth: float = 0.0

    # --- Calibration training ---
    calibration_steps: int = 500
    calibration_lr: float = 1e-2
    calibration_batch_size: int = 256
    dk_sample_size: int = 128

    # --- Quantisation mode ---
    scale_mode: str = "per_block_absmax"
    target_types: Tuple[str, ...] = ("weight", "activation")

    # --- Device / dtype ---
    device: str = "cuda"
    dtype: str = "float32"
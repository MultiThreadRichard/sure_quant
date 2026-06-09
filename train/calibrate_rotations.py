"""Single‑layer calibration training for learnable Givens rotations.

Only the Givens ``theta`` parameters are optimised; the Hadamard sign
matrix and the quantiser are frozen.

--------------------------------------------------------------------
Training strategy

We use *post‑training quantization with learnable rotation* (PTQ‑LR):

  1. Collect a representative sample of the target layer's activations
     (or weights) by running the model on calibration data.
  2. Freeze the base model and the Hadamard signs.
  3. Train only the Givens rotation angles θ to minimise a composite
     loss that balances reconstruction fidelity (MSE) with quantization
     friendliness (DKoleo + balance + range).

This is efficient because:
  - Only a few hundred scalars (θ angles) are optimised per layer.
  - The calibration set can be modest (a few thousand vectors).
  - Each layer is calibrated independently — no back‑prop through the
    full model is needed.

--------------------------------------------------------------------
Loss function breakdown

    total = λ_rec · MSE  +  λ_dk · DKoleo  +  λ_bal · Balance  +  λ_rng · Range

  - **MSE** (reconstruction): ‖x − x̂‖² — fidelity of the round‑trip.
  - **DKoleo**: −log(min pairwise distance) — spread rotated vectors
    uniformly on the unit sphere.
  - **Balance**: penalises variance imbalance across coordinates within
    a block.  If some coordinates have much higher variance than others,
    the common quantization scale loses precision on low‑variance dims.
  - **Range**: penalises large max‑to‑mean ratio across blocks, so no
    single block dominates the scale and wastes bits.

Recommended training order (from the design doc):
  1. First train with only λ_rec > 0 until MSE converges.
  2. Then add DKoleo (λ_dk) for distribution uniformity.
  3. Finally add balance (λ_bal) and range (λ_rng) for fine‑tuning.
"""

from typing import Dict, List

import torch
from torch.optim import Adam

from config.default_config import SureQuantConfig
from model.sure_quantizer import SureQuantizer
from loss.reconstruction import reconstruction_loss
from loss.dkoleo import DKoleoLoss
from loss.balance import balance_loss
from loss.range_loss import range_loss
from loss.total_loss import build_total_loss


def calibrate_single_layer(
    sure_quantizer: SureQuantizer,
    sample_tensor: torch.Tensor,
    cfg: SureQuantConfig,
) -> List[Dict]:
    """Train the Givens rotation parameters on a single layer's data.

    The quantiser is placed in train mode during calibration so that
    batch‑norm‑style statistics (if any) are updated and the Givens θ
    gradients flow.  After calibration it is set to eval mode for
    deterministic inference.

    Args:
        sure_quantizer: The ``SureQuantizer`` to calibrate.
            Must already be on the correct device.
        sample_tensor: Calibration data ``[N, D]`` on the target device.
        cfg: Training hyper‑parameters (steps, LR, loss weights, etc.).

    Returns:
        List of per‑step log dicts with keys
        ``step, loss, loss_rec, loss_dk, loss_bal, loss_rng``.
    """
    # ---- Setup ----
    sure_quantizer.train()

    # Only the Givens angles are learnable.  Hadamard signs are a buffer
    # and the quantiser has no parameters, so this selects only θ.
    optimizer = Adam(
        sure_quantizer.rotation.givens.parameters(),
        lr=cfg.calibration_lr,
    )
    # DKoleo is stateful only for the sub‑sampling; instantiate once.
    dk_loss_fn = DKoleoLoss(sample_size=cfg.dk_sample_size)

    logs: List[Dict] = []
    n = sample_tensor.shape[0]

    # ---- Training loop ----
    for step in range(cfg.calibration_steps):
        # --- Mini‑batch sampling ---
        # Random shuffling each step prevents the optimiser from over‑fitting
        # to the order of the calibration data (which is typically IID anyway).
        if n > cfg.calibration_batch_size:
            idx = torch.randperm(n, device=sample_tensor.device)[
                : cfg.calibration_batch_size
            ]
            x = sample_tensor[idx]
        else:
            x = sample_tensor

        # --- Forward pass through the full quantization pipeline ---
        # out["x_blk"]: original in block form        [B, M, g]
        # out["z"]:     rotated (pre‑quant)            [B, M, g]
        # out["x_hat_blk"]: reconstructed blocks       [B, M, g]
        # out["x_hat"]:    reconstructed flat vectors  [B, D]
        out = sure_quantizer(x)
        x_blk = out["x_blk"]
        x_hat_blk = out["x_hat_blk"]
        z = out["z"]

        # --- Compute loss components ---
        # Primary objective: minimise reconstruction error.
        loss_rec = reconstruction_loss(x_blk, x_hat_blk)

        # Auxiliary objectives computed on the rotated (pre‑quant) space z.
        # We regularise z because that is what the quantiser sees.
        loss_dk = dk_loss_fn(z)          # spread vectors uniformly
        loss_bal = balance_loss(z)       # equal variance across coords
        loss_rng = range_loss(z)         # avoid extreme block scales

        # Weighted sum
        loss = build_total_loss(loss_rec, loss_dk, loss_bal, loss_rng, cfg)

        # --- Gradient step ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- Logging ---
        log_item = {
            "step": step,
            "loss": float(loss.item()),
            "loss_rec": float(loss_rec.item()),
            "loss_dk": float(loss_dk.item()),
            "loss_bal": float(loss_bal.item()),
            "loss_rng": float(loss_rng.item()),
        }
        logs.append(log_item)

        if step % 100 == 0 or step == cfg.calibration_steps - 1:
            print(
                f"[{step:4d}/{cfg.calibration_steps}] "
                f"total={loss.item():.6f}  "
                f"rec={loss_rec.item():.6f}  "
                f"dk={loss_dk.item():.4f}  "
                f"bal={loss_bal.item():.4f}  "
                f"rng={loss_rng.item():.4f}"
            )

    # ---- Cleanup ----
    sure_quantizer.eval()
    return logs
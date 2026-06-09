"""Block‑wise symmetric uniform fake quantizer with STE.

--------------------------------------------------------------------
Why fake quantisation?

During calibration training we need to simulate the effect of quantisation
on the forward pass while keeping gradients flowing to the learnable
rotation parameters.  "Fake" quantisation means:

  - Forward: actual rounding + clamping to integer grid (simulates int8/int4).
  - Backward: identity gradient (Straight‑Through Estimator, STE).

This is the standard approach in Quantisation‑Aware Training (QAT).

--------------------------------------------------------------------
Why per‑block (not per‑tensor) scale?

A single scale factor for the entire tensor [N, M, g] would be dominated
by the block with the largest magnitude, causing other blocks to be
quantised very coarsely.  Using one scale per block (dimension M) allows
each block to adapt its quantisation range independently.

--------------------------------------------------------------------
Why symmetric (not affine) quantisation?

Symmetric quantisation has zero as a representable value and uses equal
numbers of positive and negative levels.  After Hadamard + Givens rotation,
the coordinate distribution is roughly zero‑mean and symmetric, so a
symmetric grid is near‑optimal and uses fewer bits than affine (no zero‑point
to store).  For b bits the levels are:

    −2^{b-1}, …, −1, 0, +1, …, +2^{b-1}−1

giving 2^b levels total (e.g. b=4 → 16 levels from −8 to +7).

--------------------------------------------------------------------
Straight‑Through Estimator (STE) detail:

    z_q = round(z / scale) * scale          ← forward: hard quantisation
    z_hat = z + (z_q − z).detach()          ← backward: ∂z_hat/∂z = I

The .detach() on (z_q − z) means this difference is treated as a constant
in the backward pass, so gradients flow through z_hat as if no rounding
happened.  This is a biased gradient estimator but works well in practice.
"""

import torch
import torch.nn as nn


class BlockUniformQuantizer(nn.Module):
    """Symmetric per‑block uniform fake quantizer.

    Args:
        num_bits: Number of bits for quantisation (e.g. 4 → 16 levels).
        eps: Small constant for numerical stability of scale.
    """

    def __init__(self, num_bits: int, eps: float = 1e-8):
        super().__init__()
        if num_bits < 1:
            raise ValueError(f"num_bits must be >= 1, got {num_bits}")
        self.num_bits = num_bits
        self.eps = eps
        # For b bits, the integer grid is [−2^{b-1}, 2^{b-1}−1].
        # Example b=4: qmin=−8, qmax=7.
        self.qmax = 2 ** (num_bits - 1) - 1
        self.qmin = -(2 ** (num_bits - 1))

    def forward(self, z: torch.Tensor):
        """Quantise a block‑partitioned tensor.

        Algorithm:
            1. Compute per‑block scale = max(|z_block|) / qmax.
            2. Round z / scale to nearest integer and clamp to [qmin, qmax].
            3. Reconstruct with STE for differentiable training.

        Args:
            z: Tensor of shape ``[N, M, g]``.

        Returns:
            ``(z_hat, scale)`` where:
                - ``z_hat``: ``[N, M, g]`` — differentiable approximation.
                - ``scale``: ``[M]`` — per‑block scale for deployment.
        """
        if z.dim() != 3:
            raise ValueError(f"z must be 3D [N, M, g], got shape {z.shape}")

        # ---- 1. Compute per‑block scale via absmax ----
        # scale[m] = max_{n, j} |z[n, m, j]| / qmax
        # amax over dims 0 (batch) and 2 (within‑block) gives [M].
        scale = z.abs().amax(dim=(0, 2)) / max(self.qmax, 1)
        # Clamp scale away from zero so division is safe.
        scale = torch.clamp(scale, min=self.eps)
        # Reshape to [1, M, 1] for broadcasting over [N, M, g].
        scale_bc = scale.view(1, -1, 1)

        # ---- 2. Quantise: float → int → float ----
        # Integer indices: q = round(z / scale)
        q = torch.round(z / scale_bc)
        # Clamp to representable range (simulates integer overflow in HW).
        q = torch.clamp(q, self.qmin, self.qmax)
        # Dequantised value: z_q = q * scale
        z_q = q * scale_bc

        # ---- 3. Straight‑Through Estimator ----
        # Forward: z_hat = z_q  (hard quantised value).
        # Backward: ∂z_hat/∂z = I (gradient passes through unchanged).
        # We achieve this by z + (z_q − z).detach():
        #   → forward:  z + (z_q − z) = z_q ✓
        #   → backward: grad flows only through the 'z' term ✓
        z_hat = z + (z_q - z).detach()
        return z_hat, scale
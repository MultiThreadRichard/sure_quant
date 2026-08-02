"""Block‑wise symmetric uniform fake quantizer with STE.

--------------------------------------------------------------------
Why fake quantization?

During calibration training we need to simulate the effect of quantization
on the forward pass while keeping gradients flowing to the learnable
rotation parameters.  "Fake" quantization means:

  - Forward: actual rounding + clamping to integer grid (simulates int8/int4).
  - Backward: identity gradient (Straight‑Through Estimator, STE).

This is the standard approach in Quantisation‑Aware Training (QAT).

--------------------------------------------------------------------
Why per‑block (not per‑tensor) scale?

A single scale factor for the entire tensor [N, M, g] would be dominated
by the block with the largest magnitude, causing other blocks to be
quantised very coarsely. A scale can be shared by a block across all input
vectors (``per_block``), or computed for every vector and block
(``per_vector_block``). The latter prevents one token or transposed weight
row from reducing the effective precision of every other vector.

--------------------------------------------------------------------
Why symmetric (not affine) quantization?

Symmetric quantization has zero as a representable value and uses equal
numbers of positive and negative levels.  After Hadamard + Givens rotation,
the coordinate distribution is roughly zero‑mean and symmetric, so a
symmetric grid is near‑optimal and uses fewer bits than affine (no zero‑point
to store).  For b bits the levels are:

    −2^{b-1}, …, −1, 0, +1, …, +2^{b-1}−1

giving 2^b levels total (e.g. b=4 → 16 levels from −8 to +7).

--------------------------------------------------------------------
Straight‑Through Estimator (STE) detail:

    z_q = round(z / scale) * scale          ← forward: hard quantization
    z_hat = z + (z_q − z).detach()          ← backward: ∂z_hat/∂z = I

The .detach() on (z_q − z) means this difference is treated as a constant
in the backward pass, so gradients flow through z_hat as if no rounding
happened.  This is a biased gradient estimator but works well in practice.
"""

import torch
import torch.nn as nn


def asym_quant_dequant(x, scale, zero, maxq):
    """Asymmetric quantization followed by dequantization."""
    q = torch.clamp(torch.round(x / scale + zero), 0, maxq)
    return (q - zero) * scale


class BlockUniformQuantizer(nn.Module):
    """Symmetric per‑block uniform fake quantizer.

    Args:
        num_bits: Number of bits for quantization (e.g. 4 → 16 levels).
        scale_granularity: ``"per_block"`` returns ``[M]`` scales, while
            ``"per_vector_block"`` returns ``[N, M]`` scales.
        clip_ratio: Fraction of the absolute maximum represented without
            saturation. Values below 1 trade clipping of outliers for a
            smaller quantization step.
        eps: Small constant for numerical stability of scale.
    """

    def __init__(
        self,
        num_bits: int,
        scale_granularity: str = "per_block",
        clip_ratio: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        if num_bits < 1:
            raise ValueError(f"num_bits must be >= 1, got {num_bits}")
        if scale_granularity not in ("per_block", "per_vector_block"):
            raise ValueError(
                "scale_granularity must be 'per_block' or "
                f"'per_vector_block', got {scale_granularity!r}"
            )
        if not 0.0 < clip_ratio <= 1.0:
            raise ValueError(f"clip_ratio must be in (0, 1], got {clip_ratio}")
        self.num_bits = num_bits
        self.scale_granularity = scale_granularity
        self.clip_ratio = float(clip_ratio)
        self.eps = eps
        # For b bits, the integer grid is [−2^{b-1}, 2^{b-1}−1].
        # Example b=4: qmin=−8, qmax=7.
        self.qmax = 2 ** (num_bits - 1) - 1
        self.qmin = -(2 ** (num_bits - 1))

    def forward(self, z: torch.Tensor):
        """Quantise a block‑partitioned tensor.

        Algorithm:
            1. Compute scale = clip_ratio * max(|z_block|) / qmax.
            2. Round z / scale to nearest integer and clamp to [qmin, qmax].
            3. Reconstruct with STE for differentiable training.

        Args:
            z: Tensor of shape ``[N, M, g]``.

        Returns:
            ``(z_hat, scale)`` where:
                - ``z_hat``: ``[N, M, g]`` — differentiable approximation.
                - ``scale``: ``[M]`` or ``[N, M]`` according to the configured
                  granularity.
        """
        if z.dim() != 3:
            raise ValueError(f"z must be 3D [N, M, g], got shape {z.shape}")

        # ``per_vector_block`` isolates tokens/weight rows; legacy
        # ``per_block`` shares one scale across the leading dimension.
        if self.scale_granularity == "per_vector_block":
            absolute_max = z.abs().amax(dim=2)
        else:
            absolute_max = z.abs().amax(dim=(0, 2))
        scale = absolute_max * self.clip_ratio / max(self.qmax, 1)
        # Clamp scale away from zero so division is safe.
        scale = torch.clamp(scale, min=self.eps)
        scale_bc = self.broadcast_scale(scale)

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

    @staticmethod
    def broadcast_scale(scale: torch.Tensor) -> torch.Tensor:
        """Reshape persisted 1-D or 2-D scales for ``[N, M, g]`` tensors."""
        if scale.dim() == 1:
            return scale.view(1, -1, 1)
        if scale.dim() == 2:
            return scale.unsqueeze(-1)
        raise ValueError(f"scale must have shape [M] or [N, M], got {scale.shape}")


class WeightQuantizer(torch.nn.Module):
    """Asymmetric weight quantizer for int4 quantization.

    This quantizer performs asymmetric quantization on weights with per-channel
    scaling. It is designed for weight-only quantization without calibration.

    Args:
        shape: Shape for scale and zero-point buffers.
    """

    def __init__(self, num_bits: int = 4, shape=1):
        super(WeightQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))
        self.num_bits = num_bits  # Fixed to int4
        self.perchannel = True  # Fixed to per-channel
        self.sym = False  # Fixed to asymmetric
        self.mse = False  # No MSE optimization
        self.maxq = torch.tensor((1 << num_bits) - 1)  # 15 for int4

    def find_params(self, x):
        """Compute scale and zero-point for asymmetric quantization.

        Args:
            x: Input tensor to be quantized.
        """
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        # Handle zero input case
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
        self.zero = torch.round(-xmin / self.scale)

        if not self.perchannel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)

    def quantize(self, x):
        """Apply quantization and dequantization to the input tensor.

        Args:
            x: Input tensor to be quantized.

        Returns:
            Quantized and dequantized tensor.
        """
        x_dtype = x.dtype
        if self.ready():
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
        return x

    def enabled(self):
        """Check if quantizer is enabled."""
        return self.maxq > 0

    def ready(self):
        """Check if quantizer parameters are ready."""
        return torch.all(self.scale != 0)

"""High‑level Sure Quantizer that chains blockify → rotate → quantise → inverse.

This is the central module that calibrators train and inference wrappers consume.
      
    x [N, D]                              ← original vectors (float32)
      │
      ▼ blockify(dim=D, block_size=g)
    x_blk [N, M, g]                       ← partitioned into M = D/g blocks
      │
      ▼ CompositeBlockRotation
      │   ├─ BlockHadamardTransform        ← D₂ · H · D₁ (fixed, random)
      │   └─ BlockGivensRotation           ← G_{K-1} · … · G_0 (learnable)
    z [N, M, g]                            ← rotated, pre‑quantisation
      │
      ▼ BlockUniformQuantizer(num_bits)
    z_hat [N, M, g]                        ← quantised + STE
      │
      ▼ CompositeBlockRotation.inverse()
      │   ├─ BlockGivensRotation.inverse   ← G_0ᵀ · … · G_{K-1}ᵀ
      │   └─ BlockHadamardTransform.inverse  ← D₁ · H · D₂
    x_hat_blk [N, M, g]                   ← reconstructed blocks
      │
      ▼ deblockify()
    x_hat [N, D]                           ← reconstructed vectors

The returned dict exposes intermediate tensors so that loss functions
can operate on the rotated space z (for DKoleo, balance, range) as well
as the reconstruction x_hat_blk (for MSE).

--------------------------------------------------------------------
Design rationale

Why keep Hadamard and Givens separate (not a single learned matrix)?

  - The Hadamard (D₂·H·D₁) provides a strong, randomness‑driven baseline
    that gaussianises the data.  This is a fixed, non‑learned transform.
  - The Givens (G_K·…·G_0) provides a light‑weight learned refinement
    that adapts to the specific data distribution.

This separation is analogous to a frozen pre‑trained backbone followed
by a small learnable adapter: the Hadamard does the heavy lifting and
the Givens fine‑tunes.  Because each Givens pair only costs O(1), the
learnable component adds negligible overhead during inference.

--------------------------------------------------------------------
Module hierarchy

SureQuantizer
  ├── rotation: CompositeBlockRotation
  │     ├── hadamard: BlockHadamardTransform  (fixed)
  │     └── givens:   BlockGivensRotation     (learnable θ)
  └── quantizer: BlockUniformQuantizer        (stateless, STE)
"""

import torch
import torch.nn as nn

from sure_quant.ops.block_ops import blockify, deblockify
from sure_quant.ops.hadamard import BlockHadamardTransform
from sure_quant.ops.givens import BlockGivensRotation
from sure_quant.model.wrappers import CompositeBlockRotation
from sure_quant.quant.fake_quant import BlockUniformQuantizer


class SureQuantizer(nn.Module):
    """Full rotation‑quantisation pipeline.

    Args:
        dim: Input dimension ``D`` (e.g. 4096 for a typical FFN layer).
        block_size: Block size ``g``.  Must divide ``D`` and be a power
            of two.  Typical: 16.
        num_bits: Quantisation bit‑width.  Typical: 4 (16 levels).
        order: Rotation order.  ``"hadamard_givens"`` is recommended.
    """

    def __init__(
        self,
        dim: int,
        block_size: int,
        num_bits: int,
        order: str = "hadamard_givens",
    ):
        super().__init__()
        if dim % block_size != 0:
            raise ValueError(f"dim={dim} must be divisible by block_size={block_size}")
        self.dim = dim
        self.block_size = block_size
        self.num_blocks = dim // block_size

        # ---- Sub‑modules ----
        # Fixed pre‑rotation: random diagonal ±1 × Hadamard.
        hadamard = BlockHadamardTransform(block_size, self.num_blocks)
        # Learnable rotation: Givens pairs with optimisable θ.
        givens = BlockGivensRotation(block_size, self.num_blocks)
        # Compose them in the specified order.
        self.rotation = CompositeBlockRotation(hadamard, givens, order=order)
        # Symmetric per‑block uniform quantiser (STE for training).
        self.quantizer = BlockUniformQuantizer(num_bits)

    def forward(self, x: torch.Tensor):
        """Quantise and reconstruct a batch of vectors (training & eval).

        Args:
            x: ``[N, D]`` input tensor.

        Returns:
            Dict with intermediate and final tensors:
                - ``x_blk``: ``[N, M, g]`` — blockified input.
                - ``z``: ``[N, M, g]`` — rotated, before quantization.
                - ``z_hat``: ``[N, M, g]`` — after fake quantization.
                - ``x_hat_blk``: ``[N, M, g]`` — reconstructed blocks.
                - ``x_hat``: ``[N, D]`` — reconstructed flat vectors.
                - ``scale``: ``[M]`` — per‑block quantization scale.
        """
        if x.dim() != 2 or x.shape[1] != self.dim:
            raise ValueError(
                f"Expected x of shape [N, {self.dim}], got {x.shape}"
            )

        # ---- Stage 1: Partition into blocks ----
        # x [N, D] → x_blk [N, M, g]  where M = D / g
        # This is a pure reshape — no data movement if the tensor is contiguous.
        x_blk = blockify(x, self.block_size)

        # ---- Stage 2: Rotate (Hadamard + Givens) ----
        # The rotation mixes coordinates within each block, making the
        # distribution more Gaussian and easier to quantise.
        z = self.rotation(x_blk)

        # ---- Stage 3: Quantise (fake, with STE) ----
        # z_hat ≈ z after rounding to the 2^num_bits levels.
        # scale is preserved for deployment.
        z_hat, scale = self.quantizer(z)

        # ---- Stage 4: Inverse rotation ----
        # Because both Hadamard and Givens are orthogonal, the inverse
        # is exact (no matrix solve needed).
        x_hat_blk = self.rotation.inverse(z_hat)

        # ---- Stage 5: Flatten blocks back ----
        # x_hat_blk [N, M, g] → x_hat [N, D]
        x_hat = deblockify(x_hat_blk)

        return {
            "x_blk": x_blk,
            "z": z,
            "z_hat": z_hat,
            "x_hat_blk": x_hat_blk,
            "x_hat": x_hat,
            "scale": scale,
        }
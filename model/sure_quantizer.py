"""High-level quantizer with pluggable rotation strategy (Strategy Pattern)."""

import torch
import torch.nn as nn

from ops.block_ops import blockify, deblockify
from ops.hadamard import BlockHadamardTransform
from ops.givens import BlockGivensRotation
from model.wrappers import CompositeBlockRotation, StiefelHouseholderRotation
from quant.fake_quant import BlockUniformQuantizer


class SureQuantizer(nn.Module):
    """Full rotation-quantization pipeline.

    Args:
        dim: Input dimension ``D``.
        block_size: Block size ``g``.
        num_bits: Quantization bit-width.
        order: Order for rotation strategy "rotation".
        rotation_strategy: ``"rotation"`` or ``"stiefel"``.
        rotation_module: Optional pre-built strategy module for dependency injection.
        stiefel_num_reflectors: Reflector count when using stiefel strategy.
    """

    def __init__(
        self,
        dim: int,
        block_size: int,
        num_bits: int,
        order: str = "hadamard_givens",
        rotation_strategy: str = "rotation",
        rotation_module: nn.Module | None = None,
        stiefel_num_reflectors: int = 8,
    ):
        super().__init__()
        if dim % block_size != 0:
            raise ValueError(f"dim={dim} must be divisible by block_size={block_size}")
        if rotation_strategy not in ("rotation", "stiefel"):
            raise ValueError(
                f"rotation_strategy must be 'rotation' or 'stiefel', got '{rotation_strategy}'"
            )

        self.dim = dim
        self.block_size = block_size
        self.num_blocks = dim // block_size
        self.rotation_strategy = rotation_strategy
        self.order = order
        self.stiefel_num_reflectors = int(stiefel_num_reflectors)

        if rotation_module is not None:
            self.rotation = rotation_module
        else:
            self.rotation = self._build_rotation_strategy()

        self.quantizer = BlockUniformQuantizer(num_bits)

    def _build_rotation_strategy(self) -> nn.Module:
        if self.rotation_strategy == "rotation":
            hadamard = BlockHadamardTransform(self.block_size, self.num_blocks)
            givens = BlockGivensRotation(self.block_size, self.num_blocks)
            return CompositeBlockRotation(hadamard, givens, order=self.order)

        return StiefelHouseholderRotation(
            block_size=self.block_size,
            num_blocks=self.num_blocks,
            num_reflectors=self.stiefel_num_reflectors,
        )

    def forward(self, x: torch.Tensor):
        if x.dim() != 2 or x.shape[1] != self.dim:
            raise ValueError(f"Expected x of shape [N, {self.dim}], got {x.shape}")

        x_blk = blockify(x, self.block_size)
        z = self.rotation(x_blk)
        z_hat, scale = self.quantizer(z)
        x_hat_blk = self.rotation.inverse(z_hat)
        x_hat = deblockify(x_hat_blk)

        return {
            "x_blk": x_blk,
            "z": z,
            "z_hat": z_hat,
            "x_hat_blk": x_hat_blk,
            "x_hat": x_hat,
            "scale": scale,
        }

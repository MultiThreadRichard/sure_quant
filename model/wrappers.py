"""Rotation strategy modules for SureQuantizer."""

import torch
import torch.nn as nn

from ops.hadamard import BlockHadamardTransform
from ops.givens import BlockGivensRotation


class CompositeBlockRotation(nn.Module):
    """Rotation strategy: Hadamard + Givens composition."""

    strategy_name = "rotation"

    def __init__(
        self,
        hadamard_module: BlockHadamardTransform,
        givens_module: BlockGivensRotation,
        order: str = "hadamard_givens",
    ):
        super().__init__()
        if order not in ("hadamard_givens", "givens_hadamard"):
            raise ValueError(
                f"order must be 'hadamard_givens' or 'givens_hadamard', got '{order}'"
            )
        self.hadamard = hadamard_module
        self.givens = givens_module
        self.order = order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.order == "hadamard_givens":
            x = self.hadamard(x)
            x = self.givens(x)
        else:
            x = self.givens(x)
            x = self.hadamard(x)
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self.order == "hadamard_givens":
            x = self.givens.inverse(x)
            x = self.hadamard.inverse(x)
        else:
            x = self.hadamard.inverse(x)
            x = self.givens.inverse(x)
        return x


class StiefelHouseholderRotation(nn.Module):
    """Rotation strategy: k-reflector Householder parameterization."""

    strategy_name = "stiefel"

    def __init__(self, block_size: int, num_blocks: int, num_reflectors: int):
        super().__init__()
        if num_reflectors <= 0:
            raise ValueError(f"num_reflectors must be positive, got {num_reflectors}")
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_reflectors = int(num_reflectors)
        self.reflectors = nn.Parameter(torch.randn(num_blocks, self.num_reflectors, block_size))

    @staticmethod
    def _normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)

    def _apply_reflectors(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        if x.shape[1] != self.num_blocks or x.shape[2] != self.block_size:
            raise ValueError(
                f"Expected shape (N, {self.num_blocks}, {self.block_size}), got {x.shape}"
            )
        y = x
        indices = range(self.num_reflectors - 1, -1, -1) if reverse else range(self.num_reflectors)
        for i in indices:
            v = self._normalize(self.reflectors[:, i, :])  # [M, g]
            dot = (y * v.unsqueeze(0)).sum(dim=-1, keepdim=True)  # [N, M, 1]
            y = y - 2.0 * dot * v.unsqueeze(0)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_reflectors(x, reverse=False)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        # Householder reflector is self-inverse, composition inverse is reverse order.
        return self._apply_reflectors(x, reverse=True)

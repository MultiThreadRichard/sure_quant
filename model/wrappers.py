"""Composite block rotation combining Hadamard + Givens."""

import torch
import torch.nn as nn

from ops.hadamard import BlockHadamardTransform
from ops.givens import BlockGivensRotation


class CompositeBlockRotation(nn.Module):
    """Chain a Hadamard pre‑rotation with a learnable Givens rotation.

    Supports two orders:
        ``"hadamard_givens"`` – Hadamard first, then Givens.
        ``"givens_hadamard"`` – Givens first, then Hadamard.

    Args:
        hadamard_module: Pre‑built ``BlockHadamardTransform``.
        givens_module: Pre‑built ``BlockGivensRotation``.
        order: Application order.
    """

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
        """Apply the composite rotation.

        Args:
            x: ``[N, M, g]`` block tensor.

        Returns:
            Rotated tensor ``[N, M, g]``.
        """
        if self.order == "hadamard_givens":
            x = self.hadamard(x)
            x = self.givens(x)
        else:
            x = self.givens(x)
            x = self.hadamard(x)
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the inverse (transpose) composite rotation.

        Args:
            x: ``[N, M, g]`` block tensor.

        Returns:
            Inverse‑rotated tensor ``[N, M, g]``.
        """
        if self.order == "hadamard_givens":
            x = self.givens.inverse(x)
            x = self.hadamard.inverse(x)
        else:
            x = self.hadamard.inverse(x)
            x = self.givens.inverse(x)
        return x
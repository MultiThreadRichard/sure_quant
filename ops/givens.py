r"""Block‑wise learnable Givens rotations.

--------------------------------------------------------------------
What is a Givens rotation?

A Givens rotation G(p, q, θ) ∈ R^{g×g} is the identity matrix except for
four entries at rows/cols p and q:

    G_{pp} =  cos θ      G_{pq} = −sin θ
    G_{qp} =  sin θ      G_{qq} =  cos θ

It rotates coordinates p and q by angle θ in their 2‑dimensional plane,
leaving all other coordinates unchanged.

Key properties:
  - G is orthogonal:  G · Gᵀ = I.
  - det(G) = cos²θ + sin²θ = 1  (a proper rotation, not a reflection).
  - Applying G only modifies two coordinates → O(1) cost per pair.

--------------------------------------------------------------------
Why Givens rotations for quantization?

After the Hadamard pre‑rotation, the block coordinates are roughly
Gaussian, but may still have residual structure or unfavourable
energy distribution.  A composition of Givens rotations can:

  1. Further mix coordinates to make energy more uniform (helps the
     per‑block uniform quantiser, which uses a single scale per block).
  2. Align the coordinate axes with the principal directions of the
     data distribution (improves quantization SNR).
  3. Be learned via gradient descent — only the angles θ are optimised,
     keeping the rotation orthogonal by construction.

--------------------------------------------------------------------
Why a butterfly pair topology?

We use the same pair assignment pattern as the FFT butterfly (hence the
name).  For block_size = 8, the pairs are:

    Layer 0 (stride 1): (0,1) (2,3) (4,5) (6,7)
    Layer 1 (stride 2): (0,2) (1,3) (4,6) (5,7)
    Layer 2 (stride 4): (0,4) (1,5) (2,6) (3,7)

This topology guarantees that after log₂(g) layers, every coordinate has
interacted (directly or indirectly) with every other coordinate — similar
to how the FFT mixes all frequencies.  A full butterfly has g·log₂(g)/2
pairs, giving the same expressive power as a dense g×g orthogonal matrix
but with O(g log g) application cost.

--------------------------------------------------------------------
Parameterisation:

Each pair (p, q) in each block has its own learnable angle θ_{m,k},
where m indexes the block and k indexes the pair.  This gives a total of
M × K learnable scalars (typically a few hundred), which is negligible
compared to the model parameters.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def build_butterfly_pairs(block_size: int) -> List[Tuple[int, int]]:
    """Build coordinate pairs for a butterfly (FFT‑style) topology.

    The algorithm iterates over increasing strides, pairing coordinates
    that are ``stride`` apart within each ``2·stride`` window.

    For ``block_size = 8``, the iteration produces:
        stride=1: (0,1) (2,3) (4,5) (6,7)
        stride=2: (0,2) (1,3) (4,6) (5,7)
        stride=4: (0,4) (1,5) (2,6) (3,7)
    Total: 12 pairs = 8·log₂(8)/2.

    Args:
        block_size: Number of coordinates in a block.

    Returns:
        List of ``(p, q)`` index pairs, with p < q.
    """
    pairs: List[Tuple[int, int]] = []
    stride = 1
    while stride < block_size:
        step = stride * 2
        # Slide a window of size 'step' across the block
        for start in range(0, block_size, step):
            # Within each window, pair element i with element i+stride
            for i in range(stride):
                p = start + i
                q = start + i + stride
                if q < block_size:
                    pairs.append((p, q))
        stride *= 2
    return pairs


def build_non_overlapping_pair_groups(
    pairs: List[Tuple[int, int]],
) -> List[List[int]]:
    """Partition consecutive pairs into groups that can run in parallel.

    A new group starts as soon as a pair reuses a coordinate from the current
    group.  Keeping groups contiguous is important: non-adjacent rotations
    cannot in general be reordered even when their own coordinates are
    disjoint, because rotations between them may introduce dependencies.

    Returns:
        Pair indices for each consecutive, non-overlapping group.
    """
    groups: List[List[int]] = []
    current: List[int] = []
    used = set()

    for k, (p, q) in enumerate(pairs):
        if current and (p in used or q in used):
            groups.append(current)
            current = []
            used = set()
        current.append(k)
        used.update((p, q))

    if current:
        groups.append(current)
    return groups


class BlockGivensRotation(nn.Module):
    """Per‑block learnable Givens rotation.

    Forward applies a product of Givens matrices:
        R = G_{K-1} · G_{K-2} · … · G_0
    where each G_k = G(p_k, q_k, θ_k) is a Givens rotation for pair k.

    Inverse applies the transpose:
        R⁻¹ = Rᵀ = G_0ᵀ · G_1ᵀ · … · G_{K-1}ᵀ

    Since G(p,q,θ)ᵀ = G(p,q,−θ), the inverse simply applies the pairs
    in reverse order with negated angles.

    Args:
        block_size: Size of each block.
        num_blocks: Number of independent blocks (each has its own θ).
        pairs: Optional list of ``(p, q)`` index pairs.  If ``None``,
            ``build_butterfly_pairs(block_size)`` is used.
    """

    def __init__(
        self,
        block_size: int,
        num_blocks: int,
        pairs: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()
        self.block_size = block_size
        self.num_blocks = num_blocks

        if pairs is None:
            pairs = build_butterfly_pairs(block_size)
        # These non-persistent buffers avoid rebuilding tiny CUDA index tensors
        # on every call. Pair topology is already serialized separately.
        self.register_buffer(
            "_pair_p", torch.empty(0, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_pair_q", torch.empty(0, dtype=torch.long), persistent=False
        )
        self.pairs = pairs

        # Initialise all angles to zero → initial rotation = identity.
        # This ensures that training starts from a known good point where
        # quantization quality is already reasonable (from Hadamard alone).
        theta = torch.zeros(num_blocks, self.num_pairs, dtype=torch.float32)
        self.theta = nn.Parameter(theta)

    @property
    def pairs(self) -> List[Tuple[int, int]]:
        return self._pairs

    @pairs.setter
    def pairs(self, pairs: List[Tuple[int, int]]) -> None:
        """Update pair topology and all derived execution metadata."""
        pairs = list(pairs)
        if hasattr(self, "theta") and len(pairs) != self.theta.shape[1]:
            raise ValueError(
                "Cannot change the number of Givens pairs after theta is created: "
                f"expected {self.theta.shape[1]}, got {len(pairs)}"
            )

        self._pairs = pairs
        self.num_pairs = len(pairs)
        self.pair_groups = build_non_overlapping_pair_groups(pairs)
        self._pair_group_ranges = [
            (group[0], group[-1] + 1) for group in self.pair_groups
        ]

        # Preserve the current device when checkpoints replace the topology
        # after the module has already been moved to CUDA.
        device = self._pair_p.device
        self._pair_p = torch.tensor(
            [p for p, _ in pairs], dtype=torch.long, device=device
        )
        self._pair_q = torch.tensor(
            [q for _, q in pairs], dtype=torch.long, device=device
        )

    def _apply_group_in_place(
        self, y: torch.Tensor, start: int, end: int, inverse: bool = False
    ) -> None:
        """Apply a consecutive group of disjoint pairs to a work tensor.

        ``index_select`` deliberately materializes the selected coordinates.
        They therefore remain valid for autograd while ``y`` is updated in
        place, allowing the full input to be cloned only once per call.

        The 2×2 rotation kernel is:
            [ xp' ]   [  cos θ_k   −sin θ_k ] [ xp ]
            [ xq' ] = [  sin θ_k    cos θ_k ] [ xq ]

        For the inverse, we negate θ_k, which flips the sign of sin and
        therefore transposes the 2×2 matrix.

        Args:
            y: ``[N, M, g]`` work tensor, modified in place.
            start: Inclusive pair index for this group.
            end: Exclusive pair index for this group.
            inverse: If True, apply Gᵀ (negate θ).
        """
        p = self._pair_p[start:end]
        q = self._pair_q[start:end]
        # Shape: [M, P], where P is the number of pairs in this group.
        theta_k = self.theta[:, start:end]
        if inverse:
            theta_k = -theta_k  # G(p,q,θ)ᵀ = G(p,q,−θ)

        # Shape: [1, M, P], broadcast over batch dimension N.
        c = torch.cos(theta_k).unsqueeze(0)
        s = torch.sin(theta_k).unsqueeze(0)

        xp = y.index_select(-1, p)
        xq = y.index_select(-1, q)

        new_p = c * xp - s * xq
        new_q = s * xp + c * xq

        # index_copy_ requires matching dtypes; the old slice assignment
        # implicitly performed this cast for mixed-precision inputs.
        y.index_copy_(-1, p, new_p.to(dtype=y.dtype))
        y.index_copy_(-1, q, new_q.to(dtype=y.dtype))

    def _apply_pair_groups(self, x: torch.Tensor, inverse: bool) -> torch.Tensor:
        # One full-size allocation replaces one clone per Givens pair. The
        # selected coordinate copies are required to keep in-place updates
        # safe for autograd.
        y = x.clone()
        ranges = (
            reversed(self._pair_group_ranges)
            if inverse
            else self._pair_group_ranges
        )
        for start, end in ranges:
            self._apply_group_in_place(y, start, end, inverse=inverse)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the full learned Givens rotation.

        The product is applied left‑to‑right:
            R = G_{K-1} · … · G_0
        Each G_k acts on the current state, so the effect is cumulative.

        Args:
            x: ``[N, M, g]``.

        Returns:
            Rotated tensor ``[N, M, g]``.
        """
        if x.shape[1] != self.num_blocks or x.shape[2] != self.block_size:
            raise ValueError(
                f"Expected shape (N, {self.num_blocks}, {self.block_size}), got {x.shape}"
            )
        return self._apply_pair_groups(x, inverse=False)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the inverse (transpose) rotation.

        Because (G_K · … · G_0)ᵀ = G_0ᵀ · … · G_Kᵀ, we apply pairs in
        reverse order with negated angles.

        This is an *exact* inverse (up to floating‑point rounding) — no
        matrix inversion is needed because each Givens matrix is orthogonal.

        Args:
            x: ``[N, M, g]``.

        Returns:
            Inverse‑rotated tensor ``[N, M, g]``.
        """
        if x.shape[1] != self.num_blocks or x.shape[2] != self.block_size:
            raise ValueError(
                f"Expected shape (N, {self.num_blocks}, {self.block_size}), got {x.shape}"
            )
        # Pairs within a group are disjoint and commute, so only group order
        # needs to be reversed.
        return self._apply_pair_groups(x, inverse=True)

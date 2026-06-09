"""Block‑wise Hadamard (FWHT) pre‑rotation.

Applies a random sign flip followed by a fast Walsh‑Hadamard transform,
normalised by 1/√g, to each block independently.  ``block_size`` must be
a power of two.

--------------------------------------------------------------------
Why Hadamard?

The goal is to "gaussianise" the coordinate distribution within each block.
Before rotation, coordinates of real neural‑network activations and weights
often have heavy‑tailed or structured distributions that are hard to quantise
with a uniform scalar quantiser.  Applying a random orthogonal transform
(mixing all coordinates) makes the marginal distribution of each coordinate
converge to a Gaussian — the ideal input for Lloyd‑Max / uniform quantization.

A dense random rotation costs O(d²) per vector, which is prohibitive for
large models.  The Hadamard transform is a fast structured alternative:

    rotation(x) = D₂ · H · D₁ · x    (where D₁,D₂ are random diagonal ±1)

  - D₁, D₂: random sign flips that provide statistical randomness.
  - H: normalised Walsh‑Hadamard matrix (H/√g is orthogonal).
  - Complexity: O(g log g) per block via the butterfly algorithm (FWHT).

For d → ∞ the coordinate distribution after this structured rotation still
approaches N(0, 1/d), which is what PolarQuant and block uniform quantisers
rely on.

--------------------------------------------------------------------
Why per‑block Hadamard (not full‑dimension)?

We operate on blocks of size g (e.g. 16) rather than the full dimension D
(e.g. 4096) for three reasons:

  1. g must be a power of two for FWHT; blockifying gives us flexibility.
  2. Smaller blocks mean the Hadamard mixes fewer coordinates — this gives
     the learnable Givens rotation more room to refine local structure.
  3. Per‑block rotation is trivially parallelisable and cache‑friendly.
"""

import math

import torch
import torch.nn as nn


def _is_power_of_two(x: int) -> bool:
    """Check if x is a positive power of two (required by FWHT)."""
    return x > 0 and (x & (x - 1)) == 0


def fwht_lastdim(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh–Hadamard Transform on the last dimension.

    This implements the Cooley–Tukey–style butterfly algorithm with
    O(g log g) complexity and O(g) extra memory (the copy at entry).

    ------------------------------------------------------------------
    Mathematical definition of the normalised Hadamard matrix H_g:
        H_1 = [1]
        H_{2k} = 1/√2 · [ H_k   H_k ]
                          [ H_k  −H_k ]

    The FWHT butterfly iteratively applies the 2x2 kernel (a+b, a−b)
    for pair distances 1, 2, 4, …, g/2, and then normalises by 1/√g.

    Example for g = 4:
        Input:  [x0, x1, x2, x3]
        Step h=1: pairs (0,1), (2,3)
            → [x0+x1, x0−x1, x2+x3, x2−x3]
        Step h=2: pairs (0,2), (1,3)  [stride 2]
            → [(x0+x1)+(x2+x3), (x0−x1)+(x2−x3),
               (x0+x1)−(x2+x3), (x0−x1)−(x2−x3)]
        Normalise by 1/√4 = 0.5

    ------------------------------------------------------------------
    Important property: H/√g is orthogonal → preserves L2 norm.

    Args:
        x: Tensor of shape ``[..., g]`` where ``g`` is a power of two.

    Returns:
        Hadamard‑transformed tensor, same shape, normalised by ``1/√g``.
    """
    g = x.shape[-1]
    if not _is_power_of_two(g):
        raise ValueError(f"Last dimension must be power of two, got {g}")

    # Work on a copy to avoid mutating the input
    y = x
    h = 1
    while h < g:
        # ---- Butterfly stage with stride h ----
        # Reshape so that every consecutive 2h elements form a group,
        # then split each group into left half (a) and right half (b).
        # Groups: [..., 2h] → [..., 2, h]  (last dim split into (2, h))
        y = y.reshape(*y.shape[:-1], -1, 2 * h)
        a = y[..., :h]       # left  half of each pair
        b = y[..., h : 2 * h]  # right half of each pair
        # Kernel: (a, b) → (a+b, a−b)
        y = torch.cat([a + b, a - b], dim=-1)
        y = y.reshape(*x.shape[:-1], g)
        h *= 2

    # Normalise: H_g is orthogonal with row norm √g, so H/√g is orthogonal
    return y / math.sqrt(g)


class BlockHadamardTransform(nn.Module):
    """Per‑block random sign flip + Fast Walsh–Hadamard Transform.

    The forward pass computes:  z = H_g · (s ⊙ x) / √g
    where H_g is the Hadamard matrix, s are random ±1 signs, and ⊙ is
    element‑wise multiplication.

    The signs are randomly initialised once and never updated (registered
    as a buffer, not a parameter), because the statistical randomness of
    the Hadamard rotation comes from the signs, not from learning.

    ------------------------------------------------------------------
    Why random signs?  The pure Hadamard matrix H is a *fixed* orthogonal
    matrix.  If we always used H, the rotation would be deterministic and
    could align poorly with the data distribution.  Pre‑multiplying by a
    random diagonal ±1 matrix turns this into a random rotation from the
    family {D · H : D diagonal ±1}, providing the statistical guarantees
    needed for gaussianisation.

    Args:
        block_size: Size of each block (must be power of two).
        num_blocks: Number of independent blocks.
    """

    def __init__(self, block_size: int, num_blocks: int):
        super().__init__()
        if not _is_power_of_two(block_size):
            raise ValueError(f"block_size must be power of two, got {block_size}")
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Each block gets its own independent random sign pattern.
        # These are frozen after init (registered as buffer → not trainable).
        # Using int range [0,2) → scale to {−1,+1} avoids bias in the RNG.
        signs = torch.randint(0, 2, (num_blocks, block_size), dtype=torch.float32)
        signs = signs * 2 - 1  # {0, 1} → {−1, +1}
        self.register_buffer("signs", signs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply D ⊙ WHT to each block: block → sign_flip → hadamard → normalise.

        Args:
            x: ``[N, M, g]`` where ``M == num_blocks``, ``g == block_size``.

        Returns:
            Transformed tensor ``[N, M, g]``.
        """
        if x.shape[1] != self.num_blocks or x.shape[2] != self.block_size:
            raise ValueError(
                f"Expected shape (N, {self.num_blocks}, {self.block_size}), got {x.shape}"
            )
        # Step 1: random sign flip (D₁ matrix applied element‑wise)
        x = x * self.signs.unsqueeze(0)
        # Step 2: fast Walsh–Hadamard transform (H/√g applied)
        return fwht_lastdim(x)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse transform.

        Since D is diagonal with ±1 (D⁻¹ = D) and H/√g is orthogonal
        (H⁻¹ = Hᵀ = H), the inverse is:  x → H/√g → sign_flip.

        That is, the same operations in reverse order (the Hadamard
        transform is its own inverse up to the normalisation, which
        we handle inside fwht_lastdim).
        """
        if x.shape[1] != self.num_blocks or x.shape[2] != self.block_size:
            raise ValueError(
                f"Expected shape (N, {self.num_blocks}, {self.block_size}), got {x.shape}"
            )
        # Reverse order: H first, then D (since D⁻¹ = D)
        x = fwht_lastdim(x)
        x = x * self.signs.unsqueeze(0)
        return x
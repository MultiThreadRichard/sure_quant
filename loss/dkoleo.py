"""DKoleo (Differential KoLeo) loss for distribution uniformisation.

Encourages rotated block vectors to spread uniformly on the unit sphere
within each block, penalising small nearest‑neighbour distances.

Uses sub‑sampling to keep the pairwise distance matrix O(sample_size²)
instead of O(N²).

--------------------------------------------------------------------
What is the KoLeo regulariser?

The Kozachenko–Leonenko (KL) estimator approximates the differential
entropy of a distribution from samples:

    Ĥ(X) ≈ (d / n) Σᵢ log ρᵢ  +  const

where ρᵢ = min_{j≠i} ||xᵢ − xⱼ|| is the distance to the i‑th sample's
nearest neighbour.  Maximising entropy pushes points apart and prevents
them from collapsing into clusters.

The *differential* KoLeo loss commonly used in deep learning is:

    L_DKoleo = − (1/n) Σᵢ log(min_{j≠i} ||uᵢ − uⱼ||)

where uᵢ are vectors on the unit sphere.  Minimising L_DKoleo is
equivalent to maximising the nearest‑neighbour distance, which spreads
points uniformly over the sphere.

--------------------------------------------------------------------
Why DKoleo for rotation quantization?

After rotation, we want the coordinates within each block to be:

  1. Gaussian‑distributed (for optimal uniform quantization).
  2. Well‑spread — no two vectors should be too similar, otherwise the
     quantiser wastes representable levels on redundant patterns.

DKoleo addresses (2) by penalising small nearest‑neighbour distances
in the rotated space.  This complements the reconstruction loss (MSE),
which only cares about fidelity after the full round‑trip.

--------------------------------------------------------------------
Why per‑block?

We compute DKoleo independently per block because each block has its
own quantization scale.  Within a block, vectors should be uniformly
spread; across blocks, different energy levels are fine because each
block gets its own scale.

--------------------------------------------------------------------
Why sub‑sampling?

The pairwise distance matrix is O(N²) in the number of calibration
samples N.  For N = 2048 this is already 4M entries; for larger
calibration sets it becomes prohibitive.  We randomly sub‑sample to
``sample_size`` (default 128), reducing the cost to ~16K entries per
block — negligible overhead.

"""

import torch
import torch.nn as nn


class DKoleoLoss(nn.Module):
    """DKoleo regulariser – spreads out nearest neighbours on the sphere.

    Mathematically::

        L(z) = max( 0, − (1 / (M·N)) Σ_{m=1}^{M} Σ_{i=1}^{N}
                          log( min_{j≠i} ||u_{m,i} − u_{m,j}|| + ε ) )

    where u_{m,i} = z_i[m] / ||z_i[m]|| is the i‑th vector's m‑th block
    normalised to the unit sphere.

    Compared to the unclipped ("raw") formulation from the literature
    the upper half‑line clamp guarantees that DKoleo is always a
    non‑negative regulariser: it only penalises point sets whose
    nearest‑neighbour distances are smaller than ``exp(0) = 1`` (i.e.
    directions closer than 60°).  Point sets that are already
    uniform enough incur zero penalty and zero gradient from this
    term.

    Args:
        eps: Small constant added inside the log for numerical stability
            (prevents ``log(0)`` when two vectors are identical).
        sample_size: Maximum number of vectors used for the distance
            matrix.  If ``N > sample_size`` a random subset is drawn.
    """

    def __init__(self, eps: float = 1e-6, sample_size: int = 128):
        super().__init__()
        self.eps = eps
        self.sample_size = sample_size

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the non‑negative DKoleo loss on rotated block vectors.

        Args:
            z: Rotated block tensor ``[N, M, g]``.
                N = number of calibration vectors.
                M = number of blocks.
                g = block size (vector dimension within a block).

        Returns:
            Scalar DKoleo loss, always ``≥ 0``.  Zero means the point
            set is already spread enough that every block's mean of
            ``-log(nn_dist + ε)`` is ≤ 0.
        """
        n, _m, _g = z.shape

        # ---- Optional sub‑sampling for O(sample_size²) complexity ----
        if n > self.sample_size:
            idx = torch.randperm(n, device=z.device)[: self.sample_size]
            z = z[idx]  # [sample_size, M, g]
            n = self.sample_size

        # ---- Normalise each block‑vector to the unit sphere ----
        # DKoleo operates on directions, not magnitudes, because the
        # quantiser scale already absorbs magnitude.
        u = z / (torch.norm(z, dim=-1, keepdim=True) + self.eps)  # [N, M, g]

        # Reshape so that ``torch.cdist`` treats each block as an
        # independent batch: [B, P, D] x [B, R, D] -> [B, P, R].
        u = u.permute(1, 0, 2).contiguous()  # [M, N, g]

        # ---- Pairwise Euclidean distances per block ----
        dist = torch.cdist(u, u, p=2)  # [M, N, N]

        # ---- Mask out self‑distances (i = j) ----
        diag_mask = torch.eye(n, device=dist.device, dtype=torch.bool)
        dist = dist.masked_fill(diag_mask.unsqueeze(0), float("inf"))

        # ---- Nearest neighbour distance per vector per block ----
        nn_dist = dist.min(dim=-1).values  # [M, N]

        # ---- Clamped negative log‑likelihood ----
        raw = -torch.log(nn_dist + self.eps).mean()
        # ``clamp_min(0)`` corresponds exactly to max(0, raw).  The
        # clamp is differentiable: the gradient is 0 when raw < 0 and
        # equal to the raw gradient when raw >= 0.
        return raw.clamp_min(torch.zeros_like(raw))

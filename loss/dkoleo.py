"""DKoleo (Differential KoLeo) loss for distribution uniformisation.

Encourages rotated block vectors to spread uniformly on the unit sphere
within each block, penalising small nearest‑neighbour distances.

Uses sub‑sampling to keep the pairwise distance matrix O(sample_size²)
instead of O(N²).

--------------------------------------------------------------------
What is the KoLeo regulariser?

The Kozachenko–Leonenko (KL) estimator approximates the differential
entropy of a distribution from samples:

    Ĥ(X) ≈ (d / n) Σᵢ log ρᵢ  +  const

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
    """DKoleo regulariser – maximises nearest‑neighbour distances.

    Mathematically::

        L(z) = − (1 / (M·N)) Σ_{m=1}^{M} Σ_{i=1}^{N}
                 log( min_{j≠i} ||u_{m,i} − u_{m,j}|| + ε )

    where u_{m,i} = z_i[m] / ||z_i[m]|| is the i‑th vector's m‑th block
    normalised to the unit sphere.

    Args:
        eps: Small constant added inside the log for numerical stability
            (prevents log(0) when two vectors are identical).
        sample_size: Maximum number of vectors used for the distance
            matrix.  If N > sample_size, a random subset is drawn.
    """

    def __init__(self, eps: float = 1e-6, sample_size: int = 128):
        super().__init__()
        self.eps = eps
        self.sample_size = sample_size

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute DKoleo loss on rotated block vectors.

        Args:
            z: Rotated block tensor ``[N, M, g]``.
                N = number of calibration vectors.
                M = number of blocks.
                g = block size (vector dimension within a block).

        Returns:
            Scalar DKoleo loss (lower is better — points are more spread).
        """
        n, m, g = z.shape

        # ---- Optional sub‑sampling for O(sample_size²) complexity ----
        if n > self.sample_size:
            idx = torch.randperm(n, device=z.device)[: self.sample_size]
            z = z[idx]  # [sample_size, M, g]
            n = self.sample_size

        # ---- Normalise each block‑vector to the unit sphere ----
        # The DKoleo loss operates on directions, not magnitudes, because
        # the quantiser scale already handles magnitude.  We normalise
        # each vector's m‑th block independently.
        # z[i, m, :] has shape [g]; norm along dim=−1 gives [N, M].
        u = z / (torch.norm(z, dim=-1, keepdim=True) + self.eps)  # [N, M, g]

        # Reshape so that cdist computes per‑block distances.
        # cdist expects [B, P, D] × [B, R, D] → [B, P, R].
        # We want B = M (one batch per block), P = R = N.
        u = u.permute(1, 0, 2).contiguous()  # [M, N, g]

        # ---- Pairwise Euclidean distances per block ----
        # dist[m, i, j] = ||u[m, i, :] − u[m, j, :]||
        dist = torch.cdist(u, u, p=2)  # [M, N, N]

        # ---- Mask out self‑distances (i = j) ----
        # For the nearest‑neighbour calculation, we exclude the distance
        # to oneself by setting it to infinity.
        diag_mask = torch.eye(n, device=dist.device, dtype=torch.bool)
        dist = dist.masked_fill(diag_mask.unsqueeze(0), float("inf"))

        # ---- Nearest neighbour distance per vector per block ----
        # nn_dist[m, i] = min_{j≠i} ||u[m,i] − u[m,j]||
        nn_dist = dist.min(dim=-1).values  # [M, N]

        # ---- Negative log‑likelihood ----
        # −log(nn_dist + ε): small nn_dist → large positive loss.
        # Averaged over all blocks and vectors.
        loss = -torch.log(nn_dist + self.eps).mean()
        return loss
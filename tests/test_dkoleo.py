"""Tests for DKoleo loss module (clamped formulation)."""

import torch

from loss.dkoleo import DKoleoLoss


def test_dkoleo_runs():
    """DKoleo produces a finite scalar."""
    z = torch.randn(128, 4, 16)
    loss_fn = DKoleoLoss(sample_size=64)
    loss = loss_fn(z)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_dkoleo_reduces_with_more_dispersion():
    """Pushing points apart reduces the clamped loss when it was positive."""
    torch.manual_seed(0)
    # Use a small block size and a very clustered input; with g=4 and a
    # cluster of duplicates the raw loss is large and positive so
    # clamping still leaves room for reduction.
    g = 4
    z_cluster = torch.ones(32, 2, g) + torch.randn(32, 2, g) * 1e-3

    # z_far: random points are spread much farther apart than z_cluster.
    z_far = torch.randn(32, 2, g)

    loss_fn = DKoleoLoss(sample_size=32)
    loss_tight = loss_fn(z_cluster)
    loss_wide = loss_fn(z_far)

    assert loss_tight.item() > loss_wide.item()
    assert loss_tight.item() > 0


def test_dkoleo_small_sample():
    """Works with fewer samples than sample_size."""
    z = torch.randn(10, 3, 8)
    loss_fn = DKoleoLoss(sample_size=64)
    loss = loss_fn(z)
    assert torch.isfinite(loss).item()


def test_dkoleo_deterministic_for_small_n():
    """When n <= sample_size, no sub‑sampling → deterministic."""
    torch.manual_seed(42)
    z = torch.randn(32, 2, 8)
    loss_fn = DKoleoLoss(sample_size=64)
    l1 = loss_fn(z)
    l2 = loss_fn(z)
    assert torch.allclose(l1, l2)


def test_dkoleo_gradient():
    """Gradients flow through DKoleo whenever raw >= 0."""
    # Force a clustered configuration where the raw value is positive;
    # otherwise clamp_min(0) would give a zero gradient which is correct
    # but would make the assertion fail.
    z = torch.ones(16, 2, 4, requires_grad=True)
    z = z + torch.randn_like(z) * 1e-4  # identical cluster → huge positive raw
    z = z.detach().requires_grad_(True)
    loss_fn = DKoleoLoss(sample_size=16)
    loss = loss_fn(z)
    loss.backward()
    assert z.grad is not None
    assert z.grad.abs().sum() > 0


def test_dkoleo_zero_context():
    """Identical vectors give a finite positive clamped loss."""
    z = torch.ones(8, 1, 4)
    loss_fn = DKoleoLoss(sample_size=8)
    loss = loss_fn(z)
    assert torch.isfinite(loss).item()
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# New clamp‑specific tests – guarantee the loss is always non‑negative and
# that the clamp zeroes out high‑dim random configurations that used to give
# a negative "raw" DKoleo.
# ---------------------------------------------------------------------------


def test_dkoleo_is_always_nonnegative():
    """The clamp guarantees loss >= 0 for any finite input."""
    cases = [
        torch.randn(16, 2, 4),                      # small block dim
        torch.randn(128, 4, 16),                    # small-medium
        torch.randn(128, 4, 64),                    # medium block dim
        torch.randn(128, 4, 128),                   # large block dim (previous -ve raw)
        torch.randn(2048, 2, 128),                  # many samples, large g
        torch.ones(8, 1, 4),                        # all identical cluster
        0.001 * torch.randn(32, 2, 8) + 1.0,        # tight cluster
    ]
    loss_fn = DKoleoLoss(sample_size=256)
    for z in cases:
        loss = loss_fn(z)
        assert torch.isfinite(loss).item()
        assert loss.item() >= 0.0, (
            f"DKoleo loss must be >= 0, got {loss.item()}"
        )


def test_dkoleo_highdim_random_gives_zero():
    """Random vectors in g=128 produce a negative raw → clamped to 0."""
    torch.manual_seed(0)
    z = torch.randn(128, 4, 128)
    loss_fn = DKoleoLoss(sample_size=128)
    assert loss_fn(z).item() == 0.0


def test_dkoleo_matches_max_zero_of_raw_formula():
    """Explicitly verify that loss == max(0, -mean(log(nn_dist + eps)))."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    for z in [
        torch.randn(16, 2, 4),
        torch.ones(8, 1, 4),
        torch.randn(64, 2, 8),
        torch.randn(128, 2, 128),
    ]:
        eps = 1e-6
        loss_fn = DKoleoLoss(eps=eps, sample_size=512)
        n, m, g = z.shape
        u = z / (torch.norm(z, dim=-1, keepdim=True) + eps)
        u = u.permute(1, 0, 2).contiguous()
        dist = torch.cdist(u, u, p=2)
        diag_mask = torch.eye(n, dtype=torch.bool)
        dist = dist.masked_fill(diag_mask.unsqueeze(0), float("inf"))
        nn_dist = dist.min(dim=-1).values
        raw = -torch.log(nn_dist + eps).mean().item()
        expected = max(0.0, raw)
        actual = loss_fn(z).item()
        assert abs(expected - actual) < 1e-6, (
            f"expected max(0, raw)={expected} got {actual} for z shape {tuple(z.shape)}"
        )


def test_dkoleo_gradient_is_zero_when_clamped():
    """When the loss is clamped to zero, the gradient with respect to z
    must also be exactly zero (∂/∂z max(0, raw) = 0 when raw < 0)."""
    torch.manual_seed(0)
    z = torch.randn(128, 2, 128, requires_grad=True)  # g=128 large → raw < 0
    loss_fn = DKoleoLoss(sample_size=128)
    loss = loss_fn(z)
    assert loss.item() == 0.0
    loss.backward()
    assert z.grad is not None
    assert z.grad.abs().sum().item() == 0.0, (
        "Gradient of clamped-to-zero loss must be exactly zero"
    )

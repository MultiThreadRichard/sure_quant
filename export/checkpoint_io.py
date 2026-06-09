"""Load a trained RotationQuantizer from a .pt checkpoint."""

import torch

from model.sure_quantizer import RotationQuantizer


def load_sure_quantizer(path: str, device: str = "cpu") -> RotationQuantizer:
    """Reconstruct a ``RotationQuantizer`` from a saved checkpoint.

    Args:
        path: Path to a .pt file created by ``export_sure_quantizer``.
        device: Device to load parameters onto (``"cpu"``, ``"cuda"``, etc.).

    Returns:
        A ``RotationQuantizer`` in eval mode with all parameters restored.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)

    rq = RotationQuantizer(
        dim=ckpt["dim"],
        block_size=ckpt["block_size"],
        num_bits=ckpt["num_bits"],
        order=ckpt["order"],
    )
    rq.rotation.hadamard.signs.copy_(ckpt["signs"].to(device))
    rq.rotation.givens.theta.data.copy_(ckpt["theta"].to(device))
    rq.rotation.givens.pairs = ckpt["pairs"]
    rq.to(device)
    rq.eval()
    return rq
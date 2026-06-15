"""Load a trained SureQuantizer from a .pt checkpoint."""

import torch

from model.sure_quantizer import SureQuantizer


def load_sure_quantizer(path: str, device: str = "cpu") -> SureQuantizer:
    """Reconstruct a ``SureQuantizer`` from a saved checkpoint.

    Args:
        path: Path to a .pt file created by ``export_sure_quantizer``.
        device: Device to load parameters onto (``"cpu"``, ``"cuda"``, etc.).

    Returns:
        A ``SureQuantizer`` in eval mode with all parameters restored.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)

    strategy = ckpt.get("strategy", "rotation")
    rq = SureQuantizer(
        dim=ckpt["dim"],
        block_size=ckpt["block_size"],
        num_bits=ckpt["num_bits"],
        order=ckpt.get("order", "hadamard_givens"),
        rotation_strategy=strategy,
        stiefel_num_reflectors=ckpt.get("stiefel_num_reflectors", 8),
    )

    if strategy == "rotation":
        rq.rotation.hadamard.signs.copy_(ckpt["signs"].to(device))
        rq.rotation.givens.theta.data.copy_(ckpt["theta"].to(device))
        rq.rotation.givens.pairs = ckpt["pairs"]
    elif strategy == "stiefel":
        rq.rotation.reflectors.data.copy_(ckpt["reflectors"].to(device))
    else:
        raise ValueError(f"Unsupported rotation strategy in checkpoint: {strategy}")
    rq.to(device)
    rq.eval()
    return rq
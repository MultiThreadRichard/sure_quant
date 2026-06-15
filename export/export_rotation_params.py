"""Export rotation quantizer parameters to a portable .pt checkpoint."""

import torch

from model.sure_quantizer import SureQuantizer


def export_sure_quantizer(sure_quantizer: SureQuantizer, path: str):
    """Save all quantizer state needed for deployment to a .pt file.

    The saved dict contains:
        - ``dim``, ``block_size``, ``num_blocks``, ``num_bits``, ``order``
          (hyper‑parameters).
        - ``signs`` – Hadamard random sign buffer.
        - ``theta`` – Learned Givens angles.
        - ``pairs`` – Givens coordinate pair topology.

    Args:
        sure_quantizer: A trained ``SureQuantizer`` instance.
        path: Output file path (e.g. ``"layer_quant.pt"``).
    """
    strategy = getattr(sure_quantizer, "rotation_strategy", "rotation")
    obj = {
        "dim": sure_quantizer.dim,
        "block_size": sure_quantizer.block_size,
        "num_blocks": sure_quantizer.num_blocks,
        "num_bits": sure_quantizer.quantizer.num_bits,
        "strategy": strategy,
    }

    if strategy == "rotation":
        obj["order"] = sure_quantizer.rotation.order
        obj["signs"] = sure_quantizer.rotation.hadamard.signs.detach().cpu()
        obj["theta"] = sure_quantizer.rotation.givens.theta.detach().cpu()
        obj["pairs"] = sure_quantizer.rotation.givens.pairs
    elif strategy == "stiefel":
        obj["stiefel_num_reflectors"] = sure_quantizer.rotation.num_reflectors
        obj["reflectors"] = sure_quantizer.rotation.reflectors.detach().cpu()
    else:
        raise ValueError(f"Unsupported rotation strategy: {strategy}")

    torch.save(obj, path)
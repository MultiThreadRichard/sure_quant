"""Export rotation quantizer parameters to a portable .pt checkpoint."""

import torch

from model.sure_quantizer import RotationQuantizer


def export_sure_quantizer(sure_quantizer: RotationQuantizer, path: str):
    """Save all quantizer state needed for deployment to a .pt file.

    The saved dict contains:
        - ``dim``, ``block_size``, ``num_blocks``, ``num_bits``, ``order``
          (hyper‑parameters).
        - ``signs`` – Hadamard random sign buffer.
        - ``theta`` – Learned Givens angles.
        - ``pairs`` – Givens coordinate pair topology.

    Args:
        sure_quantizer: A trained ``RotationQuantizer`` instance.
        path: Output file path (e.g. ``"layer_quant.pt"``).
    """
    obj = {
        "dim": sure_quantizer.dim,
        "block_size": sure_quantizer.block_size,
        "num_blocks": sure_quantizer.num_blocks,
        "num_bits": sure_quantizer.quantizer.num_bits,
        "order": sure_quantizer.rotation.order,
        "signs": sure_quantizer.rotation.hadamard.signs.detach().cpu(),
        "theta": sure_quantizer.rotation.givens.theta.detach().cpu(),
        "pairs": sure_quantizer.rotation.givens.pairs,
    }
    torch.save(obj, path)
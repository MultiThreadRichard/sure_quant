"""Inference‑time wrapper that applies rotation quantization before a nn.Linear.

Wraps a linear layer so its input is first quantised through the rotation
pipeline, then passed to the original linear layer.
"""

import torch
import torch.nn as nn

from model.sure_quantizer import SureQuantizer


class SureQuantLinear(nn.Module):
    """Linear layer preceded by sure quantization of its input.

    Args:
        linear: Original ``nn.Linear`` layer.
        sure_quantizer: A pre‑trained ``SureQuantizer`` instance.
    """

    def __init__(self, linear: nn.Linear, sure_quantizer: SureQuantizer):
        super().__init__()
        if linear.in_features != sure_quantizer.dim:
            raise ValueError(
                f"Linear in_features={linear.in_features} must match "
                f"SureQuantizer dim={sure_quantizer.dim}"
            )
        self.linear = linear
        self.sure_quantizer = sure_quantizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotation quantization, then the linear layer.

        Args:
            x: Input tensor of shape ``[..., D]``.

        Returns:
            Output tensor of shape ``[..., out_features]``.
        """
        original_shape = x.shape
        x2d = x.reshape(-1, x.shape[-1])
        out_dict = self.sure_quantizer(x2d)
        x_hat = out_dict["x_hat"]  # approximated quantised input
        y = self.linear(x_hat)
        new_shape = list(original_shape[:-1]) + [y.shape[-1]]
        return y.view(*new_shape)
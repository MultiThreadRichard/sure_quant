"""Inference‑time wrapper that applies rotation quantization before a nn.Linear.

Wraps a linear layer so its input is first quantised through the rotation
pipeline, then passed to the original linear layer.
"""

import torch
import torch.nn as nn

from model.sure_quantizer import SureQuantizer


class SureQuantLinear(nn.Module):
    """Linear layer with sure quantization for both input activation and weight.

    Args:
        linear: Original ``nn.Linear`` layer.
        activation_quantizer: A ``SureQuantizer`` instance for activations.
        weight_quantizer: Optional ``SureQuantizer`` instance for weights.
    """

    def __init__(self, linear: nn.Linear, activation_quantizer: SureQuantizer, weight_quantizer: SureQuantizer = None):
        super().__init__()
        if linear.in_features != activation_quantizer.dim:
            raise ValueError(
                f"Linear in_features={linear.in_features} must match "
                f"activation SureQuantizer dim={activation_quantizer.dim}"
            )
        if weight_quantizer is not None and linear.out_features != weight_quantizer.dim:
            raise ValueError(
                f"Linear out_features={linear.out_features} must match "
                f"weight SureQuantizer dim={weight_quantizer.dim}"
            )
        
        self.linear = linear
        self.activation_quantizer = activation_quantizer
        self.weight_quantizer = weight_quantizer

    def quantize_weight(self):
        """Apply rotation quantization to weight and update linear layer."""
        if self.weight_quantizer is not None:
            weight_data = self.linear.weight.data
            orig_device = weight_data.device
            orig_dtype = weight_data.dtype
            
            weight_data_cpu = weight_data.detach().cpu()
            weight_data_t = weight_data_cpu.T.contiguous()
            
            self.weight_quantizer = self.weight_quantizer.cpu()
            self.weight_quantizer.eval()
            
            with torch.no_grad():
                out_dict = self.weight_quantizer(weight_data_t)
            
            quantized_weight_t = out_dict["x_hat"]
            quantized_weight = quantized_weight_t.T.contiguous()
            quantized_weight = quantized_weight.to(orig_device).to(orig_dtype)
            self.linear.weight.data = quantized_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotation quantization to input, then the linear layer.

        Args:
            x: Input tensor of shape ``[..., D]``.

        Returns:
            Output tensor of shape ``[..., out_features]``.
        """
        input_dtype = x.dtype
        original_shape = x.shape
        x2d = x.reshape(-1, x.shape[-1])
        
        out_dict = self.activation_quantizer(x2d)
        x_hat = out_dict["x_hat"]

        x_hat = x_hat.to(input_dtype)

        y = self.linear(x_hat)
        new_shape = list(original_shape[:-1]) + [y.shape[-1]]
        return y.view(*new_shape)

    def get_weight_rotation_params(self):
        """Get weight rotation parameters for optimization."""
        if self.weight_quantizer is not None:
            return list(self.weight_quantizer.rotation.parameters())
        return []
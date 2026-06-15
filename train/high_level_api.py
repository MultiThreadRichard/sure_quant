"""High‑level calibrator API for sure quantization.

Provides a single entry point that manages sample collection, quantizer
construction, calibration training, and parameter export.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from config.default_config import SureQuantConfig
from model.sure_quantizer import SureQuantizer
from train.calibrate_rotations import calibrate_rotation
from export.export_rotation_params import export_sure_quantizer


class SureQuantCalibrator:
    """Unified calibrator for sure quantization.

    Typical usage::

        calibrator = SureQuantCalibrator(model, cfg)
        calibrator.collect_samples_for_layer("layer.0", tensor)
        calibrator.build_quantizer_for_layer("layer.0", dim=4096)
        calibrator.calibrate_layer("layer.0")
        calibrator.export_layer("layer.0", "layer_0_quant.pt")

    Args:
        model: The model whose layers will be quantised (used for hook
            registration; sample collection is manual in v1).
        cfg: ``SureQuantConfig`` instance.      
    """

    def __init__(self, model: nn.Module, cfg: SureQuantConfig):     
        self.model = model
        self.cfg = cfg
        self.layer_samples: Dict[str, torch.Tensor] = {}
        self.layer_quantizers: Dict[str, SureQuantizer] = {}

    # ------------------------------------------------------------------
    # Sample management
    # ------------------------------------------------------------------

    def collect_samples_for_layer(
        self, layer_name: str, sample_tensor: torch.Tensor
    ):
        """Store calibration samples for a named layer.

        Args:
            layer_name: Unique name for the layer.
            sample_tensor: ``[N, D]`` tensor (can be on CPU or GPU).
        """
        self.layer_samples[layer_name] = sample_tensor

    # ------------------------------------------------------------------
    # Quantizer construction
    # ------------------------------------------------------------------

    def build_quantizer_for_layer(
        self, layer_name: str, dim: int
    ) -> SureQuantizer:
        """Create a ``SureQuantizer`` for a layer and register it.

        Args:
            layer_name: Unique name for the layer.
            dim: Input dimension ``D``.

        Returns:
            The new ``SureQuantizer`` instance.
        """
        rq = SureQuantizer(
            dim=dim,
            block_size=self.cfg.block_size,
            num_bits=self.cfg.num_bits,
            order=self.cfg.order,
        )
        rq = rq.to(self.cfg.device)
        self.layer_quantizers[layer_name] = rq
        return rq

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate_layer(self, layer_name: str):
        """Run calibration training for a single layer.

        Args:
            layer_name: The layer to calibrate (must have samples and a
                quantizer already registered).

        Returns:
            Training logs (list of per‑step dicts).
        """
        if layer_name not in self.layer_quantizers:
            raise KeyError(
                f"Quantizer for '{layer_name}' not built. Call build_quantizer_for_layer first."
            )
        if layer_name not in self.layer_samples:
            raise KeyError(
                f"Samples for '{layer_name}' not collected. Call collect_samples_for_layer first."
            )

        rq = self.layer_quantizers[layer_name]
        x = self.layer_samples[layer_name].to(self.cfg.device)
        return calibrate_rotation(rq, x, self.cfg)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_layer(self, layer_name: str, path: str):
        """Export calibrated quantizer parameters to a .pt file.

        Args:
            layer_name: The layer to export.
            path: Output file path.
        """
        if layer_name not in self.layer_quantizers:
            raise KeyError(f"Quantizer for '{layer_name}' not built.")
        export_sure_quantizer(self.layer_quantizers[layer_name], path)
"""KV-cache rotation quantization reusing the SureQuant pipeline.

The decoder (language model) produces a key/value cache per attention layer.
Each layer's cache is a tensor of shape ``[batch, num_heads, seq_len, head_dim]``.
This module quantizes those caches with the *exact same* rotation + block-uniform
quantization used for weights and activations (see :class:`SureQuantizer`).

Design notes
------------

* ``head_dim`` is the quantization dimension ``D`` — the analog of
  ``in_features`` (activation) / ``out_features`` (weight) in
  :class:`SureQuantLinear`.
* ``batch * num_heads * seq_len`` vectors are flattened into the leading ``N``
  axis, so the rotation is shared across heads and positions (just like an
  activation quantizer shares one rotation across the whole batch) while the
  ``per_vector_block`` scale remains per (head, position, block).
* Rotation parameters (Hadamard signs + Givens ``theta``) are ``nn`` state, so
  they are serialized with the model and can be calibrated with the same
  ``train.calibrate_rotations.calibrate_rotation`` trainer used for weights and
  activations.  This keeps the KV path a drop-in sibling of ``SureQuantLinear``
  rather than a separate quantizer family.
"""

from __future__ import annotations

import torch
from torch import nn

from model.sure_quantizer import SureQuantizer


class SureQuantKVCache(nn.Module):
    """Per-layer rotation quantization for a decoder's K/V cache.

    Holds at most two :class:`SureQuantizer` instances — one for the key cache
    and one for the value cache — and applies them to tensors of shape
    ``[batch, num_heads, seq_len, head_dim]``.

    Args:
        head_dim: Per-head dimension ``D`` (must be divisible by ``block_size``).
        num_bits: Quantization bit-width.
        block_size: Block size ``g`` for the rotation + uniform quantizer.
        rotation_strategy: ``"rotation"`` (Hadamard + Givens) or ``"stiefel"``.
        scale_granularity: ``"per_block"`` or ``"per_vector_block"``.
        clip_ratio: Absmax clipping ratio in ``(0, 1]``.
        quantize_k: Build and apply a key-cache quantizer.
        quantize_v: Build and apply a value-cache quantizer.
    """

    def __init__(
        self,
        head_dim: int,
        num_bits: int,
        block_size: int,
        rotation_strategy: str = "rotation",
        scale_granularity: str = "per_vector_block",
        clip_ratio: float = 1.0,
        quantize_k: bool = True,
        quantize_v: bool = True,
    ):
        super().__init__()
        if head_dim % block_size != 0:
            raise ValueError(
                f"head_dim={head_dim} must be divisible by block_size={block_size}"
            )

        self.head_dim = head_dim
        self.block_size = block_size
        # Stored under private names so they do not shadow the quantize_k /
        # quantize_v methods below.
        self._quantize_k = quantize_k
        self._quantize_v = quantize_v

        common = dict(
            dim=head_dim,
            block_size=block_size,
            num_bits=num_bits,
            rotation_strategy=rotation_strategy,
            scale_granularity=scale_granularity,
            clip_ratio=clip_ratio,
        )
        self.k_quantizer = SureQuantizer(**common) if quantize_k else None
        self.v_quantizer = SureQuantizer(**common) if quantize_v else None

    @staticmethod
    def _flatten(x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
        """Flatten ``[batch, heads, seq, head_dim]`` to ``[N, head_dim]``."""
        if x.dim() != 4:
            raise ValueError(f"Expected 4D KV tensor, got shape {x.shape}")
        original_shape = x.shape
        return x.reshape(-1, x.shape[-1]), original_shape

    def _quantize(
        self,
        x: torch.Tensor,
        quantizer: SureQuantizer | None,
    ) -> torch.Tensor:
        """Quantize a 4D K or V tensor, preserving shape and input dtype."""
        if quantizer is None:
            return x
        input_dtype = x.dtype
        x2d, original_shape = self._flatten(x)
        x_hat = quantizer(x2d)["x_hat"]
        return x_hat.view(original_shape).to(dtype=input_dtype)

    def _quantize_detailed(
        self,
        x: torch.Tensor,
        quantizer: SureQuantizer | None,
    ) -> dict[str, torch.Tensor] | None:
        """Quantize and return the full pipeline states (for calibration/analysis)."""
        if quantizer is None:
            return None
        x2d, _ = self._flatten(x)
        return quantizer(x2d)

    def quantize_k(self, k: torch.Tensor) -> torch.Tensor:
        return self._quantize(k, self.k_quantizer)

    def quantize_v(self, v: torch.Tensor) -> torch.Tensor:
        return self._quantize(v, self.v_quantizer)

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize both caches.

        Args:
            k: Key cache ``[batch, num_heads, seq_len, head_dim]``.
            v: Value cache ``[batch, num_heads, seq_len, head_dim]``.

        Returns:
            ``(k_hat, v_hat)`` — dequantized reconstructions in the input dtype.
        """
        return self.quantize_k(k), self.quantize_v(v)

    @torch.inference_mode()
    def evaluate_mse(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> dict[str, float]:
        """Mean squared error between the native and quantized caches.

        Args:
            k: Native (unquantized) key cache.
            v: Native (unquantized) value cache.

        Returns:
            Dict with ``k_mse`` and ``v_mse`` computed in float32.
        """
        k_hat = self.quantize_k(k)
        v_hat = self.quantize_v(v)
        return {
            "k_mse": float((k.float() - k_hat.float()).square().mean()),
            "v_mse": float((v.float() - v_hat.float()).square().mean()),
        }

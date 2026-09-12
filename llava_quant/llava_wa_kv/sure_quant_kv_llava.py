"""LLaVA KV-cache rotation quantization (SureQuant) integration.

Extends the weight/activation rotation-quantization pipeline (``modeling.py``)
to the decoder's KV cache.  The core tensor-level quantization lives in
:class:`model.sure_quant_kv.SureQuantKVCache`; this module only adds the
LLaVA-specific glue:

* :func:`build_kv_quantizers` — build one ``SureQuantKVCache`` per decoder layer.
* :func:`quantize_kv_cache` — attach those quantizers to the model (entry point
  analogous to ``quantize_llava_model``, kept separate so it can be folded into
  ``llava_quant_calib_wa_grid_search.py`` later without touching the core).
* :class:`LLaVAKVSureQuantizer` — operate on a transformers ``Cache``
  (prefill / incremental decode / MSE evaluation).
* :func:`calibrate_kv_layer` — train the per-layer Givens rotations on collected
  KV vectors, reusing the exact ``train.calibrate_rotations`` trainer.

The KV cache is produced by the decoder (``model.language_model``); each layer
yields ``[batch, num_heads, seq_len, head_dim]`` tensors, so ``head_dim`` is the
quantization dimension.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn

from config.default_config import SureQuantConfig
from model.sure_quant_kv import SureQuantKVCache


def _language_config(model: nn.Module) -> Any:
    """Return the language-model config (LlamaConfig) of a LLaVA model."""
    language_model = getattr(model, "language_model", None)
    if language_model is not None and hasattr(language_model, "config"):
        return language_model.config
    text_config = getattr(getattr(model, "config", None), "text_config", None)
    if text_config is not None:
        return text_config
    raise ValueError("Could not locate the language-model config on the model")


def build_kv_quantizers(
    model: nn.Module,
    *,
    num_bits: int = 4,
    block_size: int = 16,
    rotation_strategy: str = "rotation",
    scale_granularity: str = "per_vector_block",
    clip_ratio: float = 1.0,
    quantize_k: bool = True,
    quantize_v: bool = True,
) -> nn.ModuleList:
    """Build one :class:`SureQuantKVCache` per decoder layer.

    The per-head dimension is ``hidden_size // num_attention_heads`` and the
    number of quantizers is ``num_hidden_layers``, both read from the language
    model config so the builder is robust to different LLaVA backbones.
    """
    config = _language_config(model)
    num_layers = config.num_hidden_layers
    head_dim = config.hidden_size // config.num_attention_heads

    return nn.ModuleList(
        [
            SureQuantKVCache(
                head_dim=head_dim,
                num_bits=num_bits,
                block_size=block_size,
                rotation_strategy=rotation_strategy,
                scale_granularity=scale_granularity,
                clip_ratio=clip_ratio,
                quantize_k=quantize_k,
                quantize_v=quantize_v,
            )
            for _ in range(num_layers)
        ]
    )


def quantize_kv_cache(
    model: nn.Module,
    **kwargs: Any,
) -> nn.Module:
    """Attach per-layer KV-cache quantizers to the model and return it.

    This is the KV analog of :func:`llava_quant.llava_wa.modeling.quantize_llava_model`
    and mirrors its signature style.  The quantizers are registered as the
    ``model.sure_quant_kv`` submodule, so their rotation parameters participate
    in ``state_dict`` / ``.to(...)`` automatically — the hook future grid-search
    or persistence integration needs.  Calling this does *not* alter the linear
    wrappers or the calibration grid; it only adds the KV quantizer module.
    """
    model.sure_quant_kv = build_kv_quantizers(model, **kwargs)
    return model


def _is_dynamic_cache(past_kv: Any) -> bool:
    """Whether ``past_kv`` is a transformers ``Cache`` (mutable ``key_cache`` lists)."""
    return hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache")


def _iter_cache_layers(past_kv: Any) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(k, v)`` tensors ``[batch, heads, seq, head_dim]`` per layer."""
    if _is_dynamic_cache(past_kv):
        return zip(past_kv.key_cache, past_kv.value_cache)
    # Legacy tuple-of-tuples format: ((k0, v0), (k1, v1), ...).
    return ((layer[0], layer[1]) for layer in past_kv)


def _get_cache_layer(past_kv: Any, i: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Read layer ``i``'s ``(k, v)`` from either cache representation."""
    if _is_dynamic_cache(past_kv):
        return past_kv.key_cache[i], past_kv.value_cache[i]
    layer = past_kv[i]
    return layer[0], layer[1]


def _set_cache_layers(
    past_kv: Any, layers_kv: list[tuple[torch.Tensor, torch.Tensor]]
) -> Any:
    """Write per-layer ``(k, v)`` back, returning a cache of the same type."""
    if _is_dynamic_cache(past_kv):
        for i, (k, v) in enumerate(layers_kv):
            past_kv.key_cache[i] = k
            past_kv.value_cache[i] = v
        return past_kv
    # Legacy tuple-of-tuples is immutable → rebuild a fresh tuple.
    return tuple(layers_kv)


class LLaVAKVSureQuantizer(nn.Module):
    """Quantize a LLaVA decoder's KV cache during prefill and decode.

    Mirrors the role of ``LLaVAKVOptimizedQuantizer`` (TurboQuant) but uses the
    SureQuant rotation pipeline.  It can be constructed from a model or from
    explicit cache geometry; the model reference is not retained.

    Args:
        model: Optional model used only to read ``num_layers`` / ``head_dim``.
        num_layers: Number of decoder layers (mutually exclusive with ``model``).
        head_dim: Per-head dimension (mutually exclusive with ``model``).
        **kv_kwargs: Forwarded to :func:`build_kv_quantizers` / ``SureQuantKVCache``
            (``num_bits``, ``block_size``, ``rotation_strategy``, ...).
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        *,
        num_layers: int | None = None,
        head_dim: int | None = None,
        **kv_kwargs: Any,
    ):
        super().__init__()
        if model is not None:
            config = _language_config(model)
            num_layers = config.num_hidden_layers
            head_dim = config.hidden_size // config.num_attention_heads
        if num_layers is None or head_dim is None:
            raise ValueError("Provide either `model` or both `num_layers` and `head_dim`")

        self.num_layers = num_layers
        self.head_dim = head_dim
        self.layers = nn.ModuleList(
            [
                SureQuantKVCache(head_dim=head_dim, **kv_kwargs)
                for _ in range(num_layers)
            ]
        )
        self.native_past_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None

    # ------------------------------------------------------------------
    # Cache quantization
    # ------------------------------------------------------------------
    @torch.no_grad()
    def quantize_prefill(self, past_kv: Any) -> Any:
        """Quantize the full cache and stash native copies for MSE evaluation.

        Works for both a transformers ``Cache`` (mutated in place) and the
        legacy tuple-of-tuples ``past_key_values`` (returns a new tuple).
        """
        self.native_past_kv = []
        quantized_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, (k, v) in enumerate(_iter_cache_layers(past_kv)):
            self.native_past_kv.append((k.detach().clone(), v.detach().clone()))
            quantized_layers.append(
                (self.layers[i].quantize_k(k), self.layers[i].quantize_v(v))
            )
        return _set_cache_layers(past_kv, quantized_layers)

    @torch.no_grad()
    def quantize_decode_incremental(self, past_kv: Any, seq_len_before: int) -> Any:
        """Quantize only the newly appended positions (``seq_len_before:``)."""
        quantized_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, (k, v) in enumerate(_iter_cache_layers(past_kv)):
            k_hist = k[:, :, :seq_len_before, :]
            v_hist = v[:, :, :seq_len_before, :]
            k_new = k[:, :, seq_len_before:, :]
            v_new = v[:, :, seq_len_before:, :]
            quantized_layers.append(
                (
                    torch.cat([k_hist, self.layers[i].quantize_k(k_new)], dim=2),
                    torch.cat([v_hist, self.layers[i].quantize_v(v_new)], dim=2),
                )
            )
        return _set_cache_layers(past_kv, quantized_layers)

    @torch.no_grad()
    def quantize_decode_with_native_update(self, past_kv: Any, seq_len_before: int) -> Any:
        """Decode-step quantization that also grows ``native_past_kv`` for eval."""
        if self.native_past_kv is None:
            raise RuntimeError("native_past_kv unset; call quantize_prefill first")
        for i, (k, v) in enumerate(_iter_cache_layers(past_kv)):
            k_new = k[:, :, seq_len_before:, :]
            v_new = v[:, :, seq_len_before:, :]
            k_native, v_native = self.native_past_kv[i]
            self.native_past_kv[i] = (
                torch.cat([k_native, k_new.detach().clone()], dim=2),
                torch.cat([v_native, v_new.detach().clone()], dim=2),
            )
        return self.quantize_decode_incremental(past_kv, seq_len_before)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_metrics(self, past_kv: Any) -> dict[str, Any]:
        """MSE between the stashed native cache and the current quantized cache."""
        if self.native_past_kv is None:
            raise RuntimeError("native_past_kv unset; call quantize_prefill first")

        layer_scores: dict[str, dict[str, float]] = {}
        for i, (k_native, v_native) in enumerate(self.native_past_kv):
            k_hat, v_hat = _get_cache_layer(past_kv, i)
            layer_scores[f"layer_{i}"] = {
                "k_mse": float((k_native.float() - k_hat.float()).square().mean()),
                "v_mse": float((v_native.float() - v_hat.float()).square().mean()),
            }

        k_mse = float(sum(s["k_mse"] for s in layer_scores.values()) / len(layer_scores))
        v_mse = float(sum(s["v_mse"] for s in layer_scores.values()) / len(layer_scores))
        return {
            "mean_k_mse": k_mse,
            "mean_v_mse": v_mse,
            "layer_scores": layer_scores,
        }


# ---------------------------------------------------------------------------
# Calibration (extensibility hook for future grid-search integration)
# ---------------------------------------------------------------------------
def calibrate_kv_layer(
    layer: SureQuantKVCache,
    k_vectors: torch.Tensor,
    v_vectors: torch.Tensor,
    cfg: SureQuantConfig,
) -> dict[str, list[dict[str, Any]]]:
    """Calibrate one layer's K/V Givens rotations on collected ``[N, head_dim]`` vectors.

    Reuses ``train.calibrate_rotations.calibrate_rotation`` so the KV path trains
    exactly like activations/weights do.  ``k_vectors`` / ``v_vectors`` are the
    flattened per-head vectors for a single decoder layer.
    """
    from train.calibrate_rotations import calibrate_rotation

    device = k_vectors.device
    logs: dict[str, list[dict[str, Any]]] = {}
    if layer.k_quantizer is not None:
        logs["k"] = calibrate_rotation(
            layer.k_quantizer.to(device), k_vectors.to(device), cfg
        )
    if layer.v_quantizer is not None:
        logs["v"] = calibrate_rotation(
            layer.v_quantizer.to(device), v_vectors.to(device), cfg
        )
    return logs

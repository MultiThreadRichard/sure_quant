from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from config.default_config import SureQuantConfig
from model.sure_quant_linear import SureQuantLinear
from ops.block_ops import blockify, deblockify
from scripts.llava_wa.modeling import quantize_llava_model


INT4_WEIGHTS_NAME = "surequant_int4_weights.pt"
INT4_FORMAT_VERSION = 1


def _jsonable_config(cfg: SureQuantConfig) -> dict[str, Any]:
    value = asdict(cfg)
    value["target_types"] = list(value["target_types"])
    return value


def _pack_signed_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack two signed values in [-8, 7] into each uint8."""
    values = values.detach().to(device="cpu", dtype=torch.int8).flatten()
    if values.numel() and (values.min() < -8 or values.max() > 7):
        raise ValueError("INT4 values must be in the range [-8, 7]")
    nibbles = torch.bitwise_and(values.to(torch.int16), 0x0F).to(torch.uint8)
    if nibbles.numel() % 2:
        nibbles = torch.cat((nibbles, torch.zeros(1, dtype=torch.uint8)))
    return nibbles[0::2] | (nibbles[1::2] << 4)


def _unpack_signed_int4(packed: torch.Tensor, value_count: int) -> torch.Tensor:
    """Unpack two's-complement nibbles into an int8 vector."""
    packed = packed.detach().to(device="cpu", dtype=torch.uint8).flatten()
    nibbles = torch.empty(packed.numel() * 2, dtype=torch.uint8)
    nibbles[0::2] = packed & 0x0F
    nibbles[1::2] = (packed >> 4) & 0x0F
    signed = nibbles.to(torch.int8)
    signed[signed >= 8] -= 16
    if value_count < 0 or value_count > signed.numel():
        raise ValueError("Invalid unpacked INT4 value count")
    return signed[:value_count]


@torch.inference_mode()
def _compress_int4_weights(
    model: nn.Module,
) -> tuple[dict[str, Any], set[str]]:
    """Encode SureQuant weights in their calibrated rotation domain."""
    layers: dict[str, dict[str, Any]] = {}
    weight_keys: set[str] = set()
    for name, module in model.named_modules():
        quantizer = module.weight_quantizer if isinstance(module, SureQuantLinear) else None
        if quantizer is None:
            continue
        if quantizer.quantizer.num_bits != 4:
            raise ValueError(
                f"{name}: packed persistence requires num_bits=4, got "
                f"{quantizer.quantizer.num_bits}"
            )

        device = module.linear.weight.device
        quantizer.to(device=device).eval()
        weight_t = module.linear.weight.detach().T.contiguous()
        rotated = quantizer.rotation(blockify(weight_t, quantizer.block_size))
        _, scale = quantizer.quantizer(rotated)
        scale_bc = quantizer.quantizer.broadcast_scale(scale)
        codes = torch.round(rotated / scale_bc).clamp(-8, 7).to(torch.int8)
        layers[name] = {
            "packed_weight": _pack_signed_int4(codes),
            "scale": scale.detach().cpu(),
            "rotated_shape": list(codes.shape),
            "weight_shape": list(module.linear.weight.shape),
            "scale_granularity": quantizer.quantizer.scale_granularity,
            "clip_ratio": quantizer.quantizer.clip_ratio,
        }
        weight_keys.add(f"{name}.linear.weight")

    artifact = {
        "format_version": INT4_FORMAT_VERSION,
        "num_bits": 4,
        "packing": "signed_twos_complement_two_values_per_uint8",
        "layers": layers,
    }
    return artifact, weight_keys


@torch.inference_mode()
def _restore_int4_weights(model: nn.Module, artifact: dict[str, Any]) -> None:
    """Decode packed weights and apply each quantizer's inverse rotation."""
    if artifact.get("format_version") != INT4_FORMAT_VERSION:
        raise ValueError(f"Unsupported INT4 format version: {artifact.get('format_version')}")
    if artifact.get("num_bits") != 4:
        raise ValueError(f"Expected a 4-bit artifact, got {artifact.get('num_bits')}")

    modules = dict(model.named_modules())
    for name, state in artifact["layers"].items():
        module = modules.get(name)
        if not isinstance(module, SureQuantLinear) or module.weight_quantizer is None:
            raise KeyError(f"Packed weight has no matching SureQuantLinear module: {name}")
        rotated_shape = tuple(int(value) for value in state["rotated_shape"])
        value_count = 1
        for dimension in rotated_shape:
            value_count *= dimension
        codes = _unpack_signed_int4(state["packed_weight"], value_count)
        device = module.linear.weight.device
        rotated = codes.reshape(rotated_shape).to(device=device)
        scale = state["scale"].to(device=device)
        scale_bc = module.weight_quantizer.quantizer.broadcast_scale(scale)
        rotated = rotated.to(dtype=scale.dtype) * scale_bc
        weight_t = deblockify(module.weight_quantizer.rotation.inverse(rotated))
        restored = weight_t.T.to(dtype=module.linear.weight.dtype)
        if list(restored.shape) != state["weight_shape"]:
            raise ValueError(
                f"{name}: restored shape {list(restored.shape)} does not match "
                f"saved shape {state['weight_shape']}"
            )
        module.linear.weight.copy_(restored)


def save_quantized_model(
    model: nn.Module,
    processor: Any,
    output_dir: Path,
    metadata: dict[str, Any],
    *,
    max_shard_size: str,
) -> None:
    """Save non-weight state plus genuinely packed INT4 quantized weights."""
    output_dir.mkdir(parents=True, exist_ok=True)
    int4_artifact, packed_weight_keys = _compress_int4_weights(model)
    state_dict = model.state_dict()
    if packed_weight_keys:
        state_dict = {
            name: value
            for name, value in state_dict.items()
            if name not in packed_weight_keys
        }
    model.save_pretrained(
        output_dir,
        state_dict=state_dict,
        safe_serialization=False,
        max_shard_size=max_shard_size,
    )
    int4_path = output_dir / INT4_WEIGHTS_NAME
    if packed_weight_keys:
        torch.save(int4_artifact, int4_path)
    else:
        int4_path.unlink(missing_ok=True)
    processor.save_pretrained(output_dir)
    metadata = dict(metadata)
    if packed_weight_keys:
        metadata["weight_storage"] = {
            "format": "surequant_packed_int4",
            "format_version": INT4_FORMAT_VERSION,
            "filename": INT4_WEIGHTS_NAME,
            "num_bits": 4,
            "packed_layers": len(int4_artifact["layers"]),
        }
    else:
        metadata["weight_storage"] = {
            "format": "floating_point",
            "packed_layers": 0,
        }
    (output_dir / "surequant_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _relocate_custom_module_params_to_ref_device(model: nn.Module) -> int:
    """Align late-created quantizer state with each wrapped linear's device."""
    moved = 0
    for module in model.modules():
        if not isinstance(module, SureQuantLinear):
            continue
        ref_device = module.linear.weight.device
        for quantizer in (module.activation_quantizer, module.weight_quantizer):
            if quantizer is None:
                continue
            for parameter in quantizer.parameters():
                if parameter.device != ref_device:
                    parameter.data = parameter.data.to(device=ref_device, dtype=parameter.dtype)
                    moved += 1
            for buffer in quantizer.buffers():
                if buffer.device != ref_device:
                    buffer.data = buffer.data.to(device=ref_device, dtype=buffer.dtype)
                    moved += 1
    return moved


def load_quantized_model(
    output_dir: str | Path,
    *,
    device_map: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """Rebuild custom wrappers before loading saved SureQuant state."""
    from transformers import LlavaForConditionalGeneration
    from transformers.modeling_utils import load_sharded_checkpoint

    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / "surequant_config.json").read_text(encoding="utf-8"))
    quant_cfg = metadata["surequant"]
    model_cfg = metadata["model_quantization"]
    model = LlavaForConditionalGeneration.from_pretrained(
        metadata["base_checkpoint"], device_map=device_map, torch_dtype=torch_dtype
    )
    quantize_llava_model(
        model,
        num_bits=quant_cfg["num_bits"],
        block_size=quant_cfg["block_size"],
        rotation_strategy=model_cfg["rotation_strategy"],
        quantize_vision=model_cfg["quantize_vision"],
        quantize_mm_proj=model_cfg["quantize_mm_proj"],
        quantize_language=model_cfg["quantize_language"],
        quantize_weight=model_cfg["quantize_weight"],
        clip_ratio=quant_cfg.get("clip_ratio", 1.0),
        activation_scale_granularity=quant_cfg.get(
            "activation_scale_granularity", "per_block"
        ),
        weight_scale_granularity=quant_cfg.get(
            "weight_scale_granularity", "per_block"
        ),
    )
    storage = metadata.get("weight_storage", {})
    is_int4_checkpoint = storage.get("format") == "surequant_packed_int4"
    int4_path = output_dir / storage.get("filename", INT4_WEIGHTS_NAME)
    if is_int4_checkpoint and not int4_path.exists():
        raise FileNotFoundError(f"Packed INT4 weight file is missing: {int4_path}")
    if (output_dir / "pytorch_model.bin.index.json").exists():
        load_sharded_checkpoint(model, output_dir, strict=not is_int4_checkpoint)
    else:
        state_dict = torch.load(
            output_dir / "pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict, strict=not is_int4_checkpoint)

    relocated = _relocate_custom_module_params_to_ref_device(model)
    if relocated:
        print(
            f"[load_quantized_model] relocated {relocated} SureQuant "
            "param/buffer tensors to match linear.weight device."
        )

    if is_int4_checkpoint:
        artifact = torch.load(int4_path, map_location="cpu", weights_only=True)
        _restore_int4_weights(model, artifact)

    model.eval()
    return model

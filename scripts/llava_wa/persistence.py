from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from config.default_config import SureQuantConfig
from model.sure_quant_linear import SureQuantLinear
from scripts.llava_wa.modeling import quantize_llava_model


def _jsonable_config(cfg: SureQuantConfig) -> dict[str, Any]:
    value = asdict(cfg)
    value["target_types"] = list(value["target_types"])
    return value


def save_quantized_model(
    model: nn.Module,
    processor: Any,
    output_dir: Path,
    metadata: dict[str, Any],
    *,
    max_shard_size: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=False, max_shard_size=max_shard_size)
    processor.save_pretrained(output_dir)
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
    )
    if (output_dir / "pytorch_model.bin.index.json").exists():
        load_sharded_checkpoint(model, output_dir, strict=True)
    else:
        state_dict = torch.load(
            output_dir / "pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict, strict=True)

    relocated = _relocate_custom_module_params_to_ref_device(model)
    if relocated:
        print(
            f"[load_quantized_model] relocated {relocated} SureQuant "
            "param/buffer tensors to match linear.weight device."
        )
    model.eval()
    return model

from __future__ import annotations

from torch import nn

from model.sure_quant_linear import SureQuantLinear
from model.sure_quantizer import SureQuantizer


def quantize_linear_layer(
    linear: nn.Linear,
    *,
    num_bits: int,
    block_size: int,
    rotation_strategy: str,
    quantize_weight: bool,
) -> SureQuantLinear:
    activation_quantizer = SureQuantizer(
        dim=linear.in_features,
        block_size=block_size,
        num_bits=num_bits,
        rotation_strategy=rotation_strategy,
    )
    weight_quantizer = None
    if quantize_weight and linear.out_features % block_size == 0:
        weight_quantizer = SureQuantizer(
            dim=linear.out_features,
            block_size=block_size,
            num_bits=num_bits,
            rotation_strategy=rotation_strategy,
        )
    return SureQuantLinear(linear, activation_quantizer, weight_quantizer)


def _replace_linears(
    root: nn.Module,
    *,
    num_bits: int,
    block_size: int,
    rotation_strategy: str,
    quantize_weight: bool,
) -> int:
    linears = [
        (name, module)
        for name, module in root.named_modules()
        if name and isinstance(module, nn.Linear) and "lm_head" not in name
    ]
    replaced = 0
    for name, linear in linears:
        if linear.in_features % block_size != 0:
            print(
                f"Skipping {name}: in_features={linear.in_features} is not "
                f"divisible by block_size={block_size}"
            )
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        setattr(
            parent,
            child_name,
            quantize_linear_layer(
                linear,
                num_bits=num_bits,
                block_size=block_size,
                rotation_strategy=rotation_strategy,
                quantize_weight=quantize_weight,
            ),
        )
        replaced += 1
    return replaced


def quantize_llava_model(
    model: nn.Module,
    *,
    num_bits: int = 4,
    block_size: int = 128,
    rotation_strategy: str = "rotation",
    quantize_vision: bool = True,
    quantize_mm_proj: bool = True,
    quantize_language: bool = True,
    quantize_weight: bool = True,
) -> nn.Module:
    targets: list[tuple[str, nn.Module]] = []
    if quantize_vision:
        targets.append(("vision", model.vision_tower.vision_model.encoder.layers))
    if quantize_mm_proj:
        targets.append(("multimodal projector", model.multi_modal_projector))
    if quantize_language:
        targets.append(("language", model.language_model.model.layers))
    for label, root in targets:
        count = _replace_linears(
            root,
            num_bits=num_bits,
            block_size=block_size,
            rotation_strategy=rotation_strategy,
            quantize_weight=quantize_weight,
        )
        print(f"Wrapped {count} {label} linear layers")
    return model


def selected_linear_names(
    model: nn.Module,
    *,
    block_size: int,
    quantize_vision: bool,
    quantize_mm_proj: bool,
    quantize_language: bool,
) -> set[str]:
    roots: list[nn.Module] = []
    if quantize_vision:
        roots.append(model.vision_tower.vision_model.encoder.layers)
    if quantize_mm_proj:
        roots.append(model.multi_modal_projector)
    if quantize_language:
        roots.append(model.language_model.model.layers)
    selected_ids = {
        id(module)
        for root in roots
        for name, module in root.named_modules()
        if name and isinstance(module, nn.Linear) and "lm_head" not in name
        and module.in_features % block_size == 0
    }
    return {name for name, module in model.named_modules() if id(module) in selected_ids}

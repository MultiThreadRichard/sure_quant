"""FP4 Quantization for LLaVA models.

- BlockFP4Quantizer: FP4 E2M1 per-group fake quantizer with STE
- SureFP4Quantizer: Rotation + FP4 quantization pipeline (drop-in replacement for SureQuantizer)
- quantize_linear_layer_fp4: Replace nn.Linear with FP4-aware SureQuantLinear
- quantize_llava_model_fp4: Full LLaVA model FP4 quantization
- FP4 persistence: save/load with FP4-packed uint8 weights

The FP4 quantizer uses NVFP4 E2M1 format (sign + 2 exponent + 1 mantissa = 4 bit)
with per-group float32 scaling and no global_scale.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from model.sure_quant_linear import SureQuantLinear
from model.wrappers import CompositeBlockRotation
from ops.block_ops import blockify, deblockify
from ops.hadamard import BlockHadamardTransform
from ops.givens import BlockGivensRotation
from quant.int4_fp4_quantizer import (
    FP4_E2M1_DATA,
    calculate_qparams,
    pack_fp4_to_uint8,
    unpack_fp4_from_uint8,
)


# ---------------------------------------------------------------------------
# FP4 persistence constants
# ---------------------------------------------------------------------------
FP4_WEIGHTS_NAME = "surequant_fp4_weights.pt"
FP4_FORMAT_VERSION = 1



class BlockFP4Quantizer(nn.Module):
    """Per-group FP4 E2M1 fake quantizer with Straight-Through Estimator.

    Args:
        num_bits: Must be 4 for FP4 E2M1.
        group_size: Per-group scaling granularity (default 16).
    """

    def __init__(
        self,
        num_bits: int = 4,
        group_size: int = 16,
        clip_ratio: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        if num_bits != 4:
            raise ValueError(f"BlockFP4Quantizer only supports 4 bits, got {num_bits}")
        if not 0.0 < clip_ratio <= 1.0:
            raise ValueError(f"clip_ratio must be in (0, 1], got {clip_ratio}")
        self.num_bits = 4
        self.q_type = "float"
        self.group_size = group_size
        self.qmax = FP4_E2M1_DATA.max  # 6.0
        self.qmin = FP4_E2M1_DATA.min  # -6.0
        self.clip_ratio = float(clip_ratio)
        self.eps = eps

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """FP4-quantize a block-partitioned tensor.

        Algorithm:
            1. Partition each block into groups of ``group_size``.
            2. Compute per-group symmetric scale = max(|z_group|) / 6.0.
            3. Quantize to FP4 E2M1 values via direct LUT cast and dequantize back.
            4. Apply STE for differentiable calibration.

        Args:
            z: Tensor of shape ``[N, M, g]`` where g is divisible by group_size.

        Returns:
            ``(z_hat, scale)`` where:
            - ``z_hat``: ``[N, M, g]`` — FP4 quantized + dequantized with STE.
            - ``scale``: ``[N, M, num_groups]`` — per-group scale for persistence.
        """
        if z.dim() != 3:
            raise ValueError(f"z must be 3D [N, M, g], got shape {z.shape}")

        N, M, g = z.shape
        if g % self.group_size != 0:
            raise ValueError(
                f"block dimension g={g} must be divisible by group_size={self.group_size}"
            )
        num_groups = g // self.group_size

        # [N, M, g] → [N*M, num_groups, group_size]
        z_grouped = z.reshape(-1, num_groups, self.group_size)

        # Per-group scale (symmetric, no global_scale).
        # Apply clip_ratio to shrink the effective range, trading clipping of
        # outliers for a finer quantization step (mirrors INT4 BlockUniformQuantizer).
        min_vals = z_grouped.amin(dim=-1)   # [N*M, num_groups]
        max_vals = z_grouped.amax(dim=-1)
        center = (min_vals + max_vals) / 2
        half_range = (max_vals - min_vals) / 2
        clipped_half_range = half_range * self.clip_ratio
        clipped_min = center - clipped_half_range
        clipped_max = center + clipped_half_range
        scale, _ = calculate_qparams(
            clipped_min, clipped_max,
            num_bits=4, q_type="float", symmetric=True,
        )  # scale: [N*M, num_groups]

        scale_bc = scale.unsqueeze(-1)  # [N*M, num_groups, 1]
        codes = FP4_E2M1_DATA.cast_to_fp4(z_grouped / scale_bc)
        z_dequant = codes * scale_bc

        # Reshape back to [N, M, g]
        z_hat = z_dequant.reshape(N, M, g)

        # Straight-Through Estimator
        z_hat = z + (z_hat - z).detach()

        # Scale: [N*M, num_groups] → [N, M, num_groups]
        scale = scale.reshape(N, M, num_groups)

        return z_hat, scale


# ---------------------------------------------------------------------------
# SureFP4Quantizer — rotation + FP4 quantization pipeline
# ---------------------------------------------------------------------------
class SureFP4Quantizer(nn.Module):
    """Rotation-quantization pipeline using FP4 E2M1 inner quantizer.

    Per-block FP4 quantization: each rotation block (of size ``block_size``)
    receives a single FP4 E2M1 scale, matching the INT4 ``BlockUniformQuantizer``
    per-block scaling design.

    Args:
        dim: Input dimension ``D``.
        block_size: Block size ``g`` (must be power of two).
        num_bits: Quantization bit-width (must be 4).
        order: Order for rotation strategy.
        rotation_strategy: ``"rotation"``.
        rotation_module: Optional pre-built strategy module.
        stiefel_num_reflectors: Reflector count for stiefel strategy.
        TODO scale_granularity: Unused; accepted for API parity with ``SureQuantizer``
            (FP4 always uses per-group scale).
        clip_ratio: Absmax clipping ratio in ``(0, 1]``.
    """

    def __init__(
        self,
        dim: int,
        block_size: int,
        num_bits: int = 4,
        order: str = "hadamard_givens",
        rotation_strategy: str = "rotation",
        rotation_module: nn.Module | None = None,
        stiefel_num_reflectors: int = 8,
        scale_granularity: str = "per_vector_block",
        clip_ratio: float = 1.0,
    ):
        super().__init__()
        if dim % block_size != 0:
            raise ValueError(f"dim={dim} must be divisible by block_size={block_size}")
        if num_bits != 4:
            raise ValueError(f"SureFP4Quantizer only supports 4 bits, got {num_bits}")
        if rotation_strategy not in ("rotation",):
            raise ValueError(f"Unknown rotation_strategy: {rotation_strategy}")

        self.dim = dim
        self.block_size = block_size
        self.num_blocks = dim // block_size
        self.num_bits = 4
        self.rotation_strategy = rotation_strategy
        self.order = order
        self.stiefel_num_reflectors = int(stiefel_num_reflectors)
        self.clip_ratio = float(clip_ratio)

        if rotation_module is not None:
            self.rotation = rotation_module
        else:
            self.rotation = self._build_rotation_strategy()

        self.quantizer = BlockFP4Quantizer(
            num_bits=4, group_size=block_size, clip_ratio=clip_ratio,
        )

    def _build_rotation_strategy(self) -> nn.Module:
        if self.rotation_strategy == "rotation":
            hadamard = BlockHadamardTransform(self.block_size, self.num_blocks)
            givens = BlockGivensRotation(self.block_size, self.num_blocks)
            return CompositeBlockRotation(hadamard, givens, order=self.order)
        return nn.Identity()  # placeholder for stiefel if needed later

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 2 or x.shape[1] != self.dim:
            raise ValueError(f"Expected x of shape [N, {self.dim}], got {x.shape}")

        x_blk = blockify(x, self.block_size)
        z = self.rotation(x_blk)
        z_hat, scale = self.quantizer(z)
        x_hat_blk = self.rotation.inverse(z_hat)
        x_hat = deblockify(x_hat_blk)

        return {
            "x_blk": x_blk,
            "z": z,
            "z_hat": z_hat,
            "x_hat_blk": x_hat_blk,
            "x_hat": x_hat,
            "scale": scale,
        }


# ---------------------------------------------------------------------------
# FP4 model quantization functions
# ---------------------------------------------------------------------------
def quantize_linear_layer_fp4(
    linear: nn.Linear,
    *,
    num_bits: int,
    block_size: int,
    rotation_strategy: str,
    quantize_weight: bool,
    clip_ratio: float = 1.0,
    activation_scale_granularity: str = "per_vector_block",
    weight_scale_granularity: str = "per_vector_block",
) -> SureQuantLinear:
    """Replace a single nn.Linear with an FP4-aware SureQuantLinear."""
    activation_quantizer = SureFP4Quantizer(
        dim=linear.in_features,
        block_size=block_size,
        num_bits=num_bits,
        rotation_strategy=rotation_strategy,
        scale_granularity=activation_scale_granularity,
        clip_ratio=clip_ratio,
    )
    weight_quantizer = None
    if quantize_weight and linear.out_features % block_size == 0:
        weight_quantizer = SureFP4Quantizer(
            dim=linear.out_features,
            block_size=block_size,
            num_bits=num_bits,
            rotation_strategy=rotation_strategy,
            scale_granularity=weight_scale_granularity,
            clip_ratio=clip_ratio,
        )
    return SureQuantLinear(linear, activation_quantizer, weight_quantizer)


def _replace_linears_fp4(
    root: nn.Module,
    *,
    num_bits: int,
    block_size: int,
    rotation_strategy: str,
    quantize_weight: bool,
    clip_ratio: float = 1.0,
    activation_scale_granularity: str = "per_vector_block",
    weight_scale_granularity: str = "per_vector_block",
) -> int:
    """Recursively replace nn.Linear children with FP4-wrapped versions."""
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
            quantize_linear_layer_fp4(
                linear,
                num_bits=num_bits,
                block_size=block_size,
                rotation_strategy=rotation_strategy,
                quantize_weight=quantize_weight,
                clip_ratio=clip_ratio,
                activation_scale_granularity=activation_scale_granularity,
                weight_scale_granularity=weight_scale_granularity,
            ),
        )
        replaced += 1
    return replaced


def quantize_llava_model_fp4(
    model: nn.Module,
    *,
    num_bits: int = 4,
    block_size: int = 128,
    rotation_strategy: str = "rotation",
    quantize_vision: bool = True,
    quantize_mm_proj: bool = True,
    quantize_language: bool = True,
    quantize_weight: bool = True,
    clip_ratio: float = 1.0,
    activation_scale_granularity: str = "per_block",
    weight_scale_granularity: str = "per_block",
) -> nn.Module:
    """Quantize a LLaVA model using FP4 E2M1 quantization.

    Same target selection as ``quantize_llava_model`` but wraps with
    ``SureFP4Quantizer`` instead of ``SureQuantizer``.
    """
    targets: list[tuple[str, nn.Module]] = []
    if quantize_vision:
        targets.append(("vision", model.vision_tower.vision_model.encoder.layers))
    if quantize_mm_proj:
        targets.append(("multimodal projector", model.multi_modal_projector))
    if quantize_language:
        targets.append(("language", model.language_model.model.layers))
    for label, root in targets:
        count = _replace_linears_fp4(
            root,
            num_bits=num_bits,
            block_size=block_size,
            rotation_strategy=rotation_strategy,
            quantize_weight=quantize_weight,
            clip_ratio=clip_ratio,
            activation_scale_granularity=activation_scale_granularity,
            weight_scale_granularity=weight_scale_granularity,
        )
        print(f"Wrapped {count} {label} linear layers (FP4)")
    return model


# def selected_fp4_linear_names(
#     model: nn.Module,
#     *,
#     block_size: int,
#     quantize_vision: bool,
#     quantize_mm_proj: bool,
#     quantize_language: bool,
# ) -> set[str]:
#     """Identify linear modules eligible for FP4 quantization."""
#     roots: list[nn.Module] = []
#     if quantize_vision:
#         roots.append(model.vision_tower.vision_model.encoder.layers)
#     if quantize_mm_proj:
#         roots.append(model.multi_modal_projector)
#     if quantize_language:
#         roots.append(model.language_model.model.layers)
#     selected_ids = {
#         id(module)
#         for root in roots
#         for name, module in root.named_modules()
#         if name and isinstance(module, nn.Linear) and "lm_head" not in name
#         and module.in_features % block_size == 0
#     }
#     return {name for name, module in model.named_modules() if id(module) in selected_ids}


# ---------------------------------------------------------------------------
# FP4 Pack / Unpack 
# ---------------------------------------------------------------------------
def _pack_fp4_signed(values: torch.Tensor) -> torch.Tensor:
    """Pack FP4 values in [-6.0, 6.0] into uint8 (2 values per uint8)."""
    values = values.detach().to(device="cpu", dtype=torch.float32)
    if values.numel() and (values.min() < -6.0 or values.max() > 6.0):
        raise ValueError("FP4 values must be in the range [-6.0, 6.0]")
    return pack_fp4_to_uint8(values)


def _unpack_fp4_signed(
    packed: torch.Tensor,
    original_shape: tuple,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Unpack uint8 back to FP4 float tensor."""
    return unpack_fp4_from_uint8(packed, original_shape, dtype)


# ---------------------------------------------------------------------------
# FP4 Weight Compression / Decompression
# ---------------------------------------------------------------------------
@torch.inference_mode()
def _compress_fp4_weights(
    model: nn.Module,
) -> tuple[dict[str, Any], set[str]]:
    """Encode FP4-quantized weights in their calibrated rotation domain.

    For each ``SureQuantLinear`` with an FP4 weight quantizer:
        1. Rotate weight to the calibrated rotation domain.
        2. Compute per-group FP4 scale.
        3. Quantize to FP4 E2M1 codes and pack into uint8.

    Returns:
        (artifact_dict, packed_weight_keys)
    """
    layers: dict[str, Any] = {}
    weight_keys: set[str] = set()

    for name, module in model.named_modules():
        if not isinstance(module, SureQuantLinear) or module.weight_quantizer is None:
            continue

        wq = module.weight_quantizer
        # Verify it is an FP4 quantizer
        if not isinstance(wq.quantizer, BlockFP4Quantizer):
            raise ValueError(
                f"{name}: weight_quantizer is not FP4; "
                f"got {type(wq.quantizer).__name__}"
            )
        if wq.num_bits != 4:
            raise ValueError(
                f"{name}: FP4 persistence requires num_bits=4, got {wq.num_bits}"
            )

        device = module.linear.weight.device
        wq.to(device=device).eval()
        weight_t = module.linear.weight.detach().T.contiguous()
        rotated = wq.rotation(blockify(weight_t, wq.block_size))
        # rotated: [in_features, M, g]

        # --- 与 BlockFP4Quantizer.forward() 保持一致的每组分块 FP4 量化 ---
        N, M, g = rotated.shape
        group_size = wq.quantizer.group_size
        clip_ratio = wq.quantizer.clip_ratio
        if g % group_size != 0:
            raise ValueError(
                f"block dimension g={g} must be divisible by group_size={group_size}"
            )
        num_groups = g // group_size

        # [N, M, g] -> [N*M, num_groups, group_size]
        z_grouped = rotated.reshape(-1, num_groups, group_size)

        # 每组分块对称 scale (应用 clip_ratio 缩小有效范围)
        min_vals = z_grouped.amin(dim=-1)   # [N*M, num_groups]
        max_vals = z_grouped.amax(dim=-1)
        center = (min_vals + max_vals) / 2
        half_range = (max_vals - min_vals) / 2
        clipped_half_range = half_range * clip_ratio
        clipped_min = center - clipped_half_range
        clipped_max = center + clipped_half_range
        scale, _ = calculate_qparams(
            clipped_min, clipped_max,
            num_bits=4, q_type="float", symmetric=True,
        )  # scale: [N*M, num_groups]

        # 计算 FP4 码值 (cast_to_fp4 返回 float32, 值在 [-6.0, 6.0])
        scale_bc = scale.unsqueeze(-1)  # [N*M, num_groups, 1]
        codes = FP4_E2M1_DATA.cast_to_fp4(z_grouped / scale_bc) # [N*M, num_groups, group_size]

        # 还原形状到 [N, M, g]
        codes = codes.reshape(N, M, g)

        # 打包为 uint8 (每两个 FP4 值共用一个字节)
        packed = _pack_fp4_signed(codes)

        # 保存 scale: [N*M, num_groups] -> [N, M, num_groups]
        scale = scale.reshape(N, M, num_groups)

        layers[name] = {
            "packed_weight": packed,
            "scale": scale.detach().cpu(),
            "rotated_shape": list(codes.shape),
            "weight_shape": list(module.linear.weight.shape),
            "group_size": group_size,
            "num_groups": num_groups,
            "clip_ratio": clip_ratio,
        }
        weight_keys.add(f"{name}.linear.weight")

    artifact = {
        "format_version": FP4_FORMAT_VERSION,
        "num_bits": 4,
        "quant_type": "fp4_e2m1",
        "packing": "fp4_two_values_per_uint8",
        "layers": layers,
    }
    return artifact, weight_keys


@torch.inference_mode()
def _restore_fp4_weights(model: nn.Module, artifact: dict[str, Any]) -> None:
    """Decode FP4 packed weights and apply each quantizer's inverse rotation.

    Args:
        model: Model with FP4-quantized ``SureQuantLinear`` layers.
        artifact: Dictionary produced by ``_compress_fp4_weights``.
    """
    if artifact.get("format_version") != FP4_FORMAT_VERSION:
        raise ValueError(f"Unsupported FP4 format version: {artifact.get('format_version')}")
    if artifact.get("num_bits") != 4:
        raise ValueError(f"Expected a 4-bit FP4 artifact, got {artifact.get('num_bits')}")

    modules = dict(model.named_modules())
    for name, state in artifact["layers"].items():
        module = modules.get(name)
        if not isinstance(module, SureQuantLinear) or module.weight_quantizer is None:
            raise KeyError(f"FP4 packed weight has no matching SureQuantLinear module: {name}")

        rotated_shape = tuple(int(v) for v in state["rotated_shape"])
        device = module.linear.weight.device

        # Unpack uint8 → FP4 float codes
        codes = _unpack_fp4_signed(
            state["packed_weight"], rotated_shape, dtype=torch.float32,
        ).to(device=device)

        # Dequantize using stored per-group scale
        scale = state["scale"].to(device=device)
        N, M, g = rotated_shape
        group_size = int(state["group_size"])
        num_groups = int(state["num_groups"])

        # Broadcast scale
        scale_bc = scale.reshape(N, M, num_groups, 1)
        scale_bc = scale_bc.expand(-1, -1, -1, group_size).reshape(N, M, g)

        rotated = codes * scale_bc.to(codes.dtype)

        # Inverse rotation to recover weight
        weight_t = deblockify(module.weight_quantizer.rotation.inverse(rotated))
        restored = weight_t.T.to(dtype=module.linear.weight.dtype)

        if list(restored.shape) != state["weight_shape"]:
            raise ValueError(
                f"{name}: restored shape {list(restored.shape)} does not match "
                f"saved shape {state['weight_shape']}"
            )
        module.linear.weight.copy_(restored)


# ---------------------------------------------------------------------------
# FP4 save / load
# ---------------------------------------------------------------------------
def save_quantized_model_fp4(
    model: nn.Module,
    processor: Any,
    output_dir: str | Path,
    metadata: dict[str, Any],
    *,
    max_shard_size: str = "5GB",
) -> None:
    """Save FP4-quantized model with packed uint8 weights.

    Same structure as ``save_quantized_model`` but uses FP4 compression.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fp4_artifact, packed_weight_keys = _compress_fp4_weights(model)

    # Remove packed weights from float state dict
    state_dict = model.state_dict()
    if packed_weight_keys:
        state_dict = {
            k: v for k, v in state_dict.items() if k not in packed_weight_keys
        }

    model.save_pretrained(
        output_dir,
        state_dict=state_dict,
        safe_serialization=False,
        max_shard_size=max_shard_size,
    )

    # Save FP4 weights
    fp4_path = output_dir / FP4_WEIGHTS_NAME
    if packed_weight_keys:
        torch.save(fp4_artifact, fp4_path)
    else:
        fp4_path.unlink(missing_ok=True)

    processor.save_pretrained(output_dir)

    # Build metadata
    metadata = dict(metadata)
    if packed_weight_keys:
        metadata["weight_storage"] = {
            "format": "surequant_packed_fp4",
            "format_version": FP4_FORMAT_VERSION,
            "filename": FP4_WEIGHTS_NAME,
            "num_bits": 4,
            "quant_type": "fp4_e2m1",
            "packed_layers": len(fp4_artifact["layers"]),
        }
    else:
        metadata["weight_storage"] = {
            "format": "floating_point",
            "packed_layers": 0,
        }

    (output_dir / "surequant_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"FP4 quantized model saved to {output_dir}")


def load_quantized_model_fp4(
    output_dir: str | Path,
    *,
    device_map: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """Load an FP4-quantized model saved with ``save_quantized_model_fp4``."""
    from transformers import LlavaForConditionalGeneration
    from transformers.modeling_utils import load_sharded_checkpoint

    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / "surequant_config.json").read_text(encoding="utf-8"))

    quant_cfg = metadata["surequant"]
    model_cfg = metadata["model_quantization"]
    storage = metadata.get("weight_storage", {})
    is_fp4_checkpoint = storage.get("format") == "surequant_packed_fp4"
    fp4_path = output_dir / storage.get("filename", FP4_WEIGHTS_NAME)

    if is_fp4_checkpoint and not fp4_path.exists():
        raise FileNotFoundError(f"FP4 packed weight file is missing: {fp4_path}")

    # Load base model
    model = LlavaForConditionalGeneration.from_pretrained(
        metadata["base_checkpoint"], device_map=device_map, torch_dtype=torch_dtype,
    )

    # Apply FP4 quantization wrappers
    quantize_llava_model_fp4(
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

    # Load float state dict
    if (output_dir / "pytorch_model.bin.index.json").exists():
        load_sharded_checkpoint(model, output_dir, strict=not is_fp4_checkpoint)
    else:
        state_dict = torch.load(
            output_dir / "pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict, strict=not is_fp4_checkpoint)

    # Relocate quantizer params to correct devices
    _relocate_custom_module_params_to_ref_device(model)

    # Restore FP4 packed weights
    if is_fp4_checkpoint:
        artifact = torch.load(fp4_path, map_location="cpu", weights_only=True)
        _restore_fp4_weights(model, artifact)

    model.eval()
    return model


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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# def _relocate_fp4_module_params(model: nn.Module) -> int:
#     """Align late-created FP4 quantizer state with each wrapped linear's device."""
#     moved = 0
#     for module in model.modules():
#         if not isinstance(module, SureQuantLinear):
#             continue
#         ref_device = module.linear.weight.device
#         for quantizer in (module.activation_quantizer, module.weight_quantizer):
#             if quantizer is None:
#                 continue
#             for parameter in quantizer.parameters():
#                 if parameter.device != ref_device:
#                     parameter.data = parameter.data.to(device=ref_device, dtype=parameter.dtype)
#                     moved += 1
#             for buffer in quantizer.buffers():
#                 if buffer.device != ref_device:
#                     buffer.data = buffer.data.to(device=ref_device, dtype=buffer.dtype)
#                     moved += 1
#     if moved:
#         print(f"[FP4 load] relocated {moved} quantizer param/buffer tensors")
#     return moved
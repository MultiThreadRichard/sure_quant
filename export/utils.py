import os
import time
import torch
import torch.nn as nn
from transformers import AutoProcessor, LlavaForConditionalGeneration

from model.sure_quantizer import SureQuantizer
from model.sure_quant_linear import SureQuantLinear

from PIL import Image
from datasets import load_dataset

from train.calibrate_rotations import calibrate_rotation
from config.default_config import SureQuantConfig
from loss.reconstruction import reconstruction_loss


def quantize_linear_layer(
    linear: nn.Linear,
    num_bits: int = 4,
    block_size: int = 128,
    rotation_strategy: str = "rotation",
    quantize_weight: bool = True
) -> SureQuantLinear:
    """将普通 Linear 层替换为量化 Linear 层

    Args:
        linear: 原始 nn.Linear 层
        num_bits: 量化位宽
        block_size: 分块大小
        rotation_strategy: 旋转策略 ("rotation" 或 "stiefel")
        quantize_weight: 是否对权重应用旋转量化

    Returns:
        SureQuantLinear: 量化后的线性层
    """
    activation_quantizer = SureQuantizer(
        dim=linear.in_features,
        block_size=block_size,
        num_bits=num_bits,
        rotation_strategy=rotation_strategy
    )

    weight_quantizer = None
    if quantize_weight and linear.out_features % block_size == 0:
        weight_quantizer = SureQuantizer(
            dim=linear.out_features,
            block_size=block_size,
            num_bits=num_bits,
            rotation_strategy=rotation_strategy
        )

    return SureQuantLinear(linear, activation_quantizer, weight_quantizer)


def quantize_llava_model(
    model: LlavaForConditionalGeneration,
    num_bits: int = 4,
    block_size: int = 128,
    rotation_strategy: str = "rotation",
    quantize_vision: bool = True,
    quantize_mm_proj: bool = True,
    quantize_language: bool = True,
    quantize_weight: bool = True
) -> LlavaForConditionalGeneration:
    """量化 LLaVA 模型的激活和权重

    Args:
        model: 原始 LLaVA 模型
        num_bits: 量化位宽
        block_size: 分块大小
        rotation_strategy: 旋转策略
        quantize_vision: 是否量化视觉编码器
        quantize_mm_proj: 是否量化多模态投影层
        quantize_language: 是否量化语言解码器
        quantize_weight: 是否对权重应用旋转量化

    Returns:
        LlavaForConditionalGeneration: 量化后的模型
    """
    def quantize_module(submodule):
        for name, module in submodule.named_modules():
            if isinstance(module, nn.Linear):
                if 'lm_head' in name:
                    continue

                quantized_linear = quantize_linear_layer(
                    module, num_bits, block_size, rotation_strategy, quantize_weight
                )
                parent_module = get_parent_module(submodule, name)
                set_attr_by_name(parent_module, name.split('.')[-1], quantized_linear)

    if quantize_vision:
        print(">>>>> Quantizing vision model...")
        quantize_module(model.vision_tower.vision_model.encoder.layers)

    if quantize_mm_proj:
        print(">>>>> Quantizing multimodal projection...")
        quantize_module(model.multi_modal_projector)

    if quantize_language:
        print(">>>>> Quantizing language model...")
        quantize_module(model.language_model.model.layers)

    return model


def get_parent_module(module: nn.Module, name: str) -> nn.Module:
    """获取模块的父模块"""
    parts = name.split('.')
    if len(parts) == 1:
        return module
    parent_name = '.'.join(parts[:-1])
    return module.get_submodule(parent_name)


def set_attr_by_name(module: nn.Module, attr_name: str, value):
    """通过名称设置模块属性"""
    setattr(module, attr_name, value)


def save_quantized_model(
    quantized_model: LlavaForConditionalGeneration,
    save_path: str,
    cfg: SureQuantConfig = None
) -> None:
    """保存量化后的模型及其配置

    Args:
        quantized_model: 已校准的量化模型
        save_path: 保存路径（目录）
        cfg: 量化配置（可选）
    """
    os.makedirs(save_path, exist_ok=True)

    # 保存模型权重
    model_path = os.path.join(save_path, "pytorch_model.bin")
    torch.save(quantized_model.state_dict(), model_path)
    print(f"Model weights saved to {model_path}")

    # 保存配置
    if cfg is not None:
        cfg_path = os.path.join(save_path, "quant_config.json")
        import json
        cfg_dict = {
            "num_bits": cfg.num_bits,
            "block_size": cfg.block_size,
            "rotation_strategy": cfg.rotation_strategy,
        }
        with open(cfg_path, 'w') as f:
            json.dump(cfg_dict, f, indent=2)
        print(f"Quantization config saved to {cfg_path}")

    # 保存处理器配置（用于推理）
    processor_config_path = os.path.join(save_path, "processor_config.json")
    processor_config = {
        "model_name_or_path": "/home/ccwan/stu_Jiangtp/model_repo/llava-7b-hf",
    }
    with open(processor_config_path, 'w') as f:
        json.dump(processor_config, f, indent=2)
    print(f"Processor config saved to {processor_config_path}")

    print(f"Quantized model saved successfully to {save_path}")


def load_quantized_model(
    save_path: str,
    device: str = "cuda"
) -> LlavaForConditionalGeneration:
    """加载已保存的量化模型

    Args:
        save_path: 模型保存路径（目录）
        device: 加载设备

    Returns:
        LlavaForConditionalGeneration: 加载后的量化模型
    """
    import json

    # 加载配置
    cfg_path = os.path.join(save_path, "quant_config.json")
    with open(cfg_path, 'r') as f:
        cfg_dict = json.load(f)

    num_bits = cfg_dict.get("num_bits", 4)
    block_size = cfg_dict.get("block_size", 128)
    rotation_strategy = cfg_dict.get("rotation_strategy", "rotation")

    # 加载处理器配置
    processor_config_path = os.path.join(save_path, "processor_config.json")
    with open(processor_config_path, 'r') as f:
        processor_config = json.load(f)
    checkpoint = processor_config["model_name_or_path"]

    # 先加载原始模型
    model = LlavaForConditionalGeneration.from_pretrained(
        checkpoint,
        device_map='cuda',
        torch_dtype=torch.float16,
    )

    # 应用量化（替换为 SureQuantLinear）
    quantized_model = quantize_llava_model(
        model,
        num_bits=num_bits,
        block_size=block_size,
        rotation_strategy=rotation_strategy,
        quantize_weight=True
    )

    # 加载量化后的权重
    model_path = os.path.join(save_path, "pytorch_model.bin")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    quantized_model.load_state_dict(state_dict)

    # 移动到目标设备
    quantized_model.to(device)
    quantized_model.eval()

    print(f"Quantized model loaded successfully from {save_path}")
    return quantized_model
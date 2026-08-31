"""LLaVA Model Quantization with SureQuant."""
import sys
sys.path.append("/home/ecnu01/workspace/sure_quant")

import torch
import torch.nn as nn
from transformers import AutoProcessor, LlavaForConditionalGeneration

from model.sure_quantizer import SureQuantizer
from model.sure_quant_linear import SureQuantLinear

from PIL import Image

from train.calibrate_rotations import calibrate_rotation
from config.default_config import SureQuantConfig


class WeightQuantizer(nn.Module):
    """对称均匀权重量化器（补充现有激活量化器）"""
    
    def __init__(self, num_bits: int, eps: float = 1e-8):
        super().__init__()
        self.num_bits = num_bits
        self.eps = eps
        self.qmax = 2 ** (num_bits - 1) - 1
        self.qmin = -(2 ** (num_bits - 1))
    
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """量化权重并返回量化值"""
        scale = weight.abs().amax() / max(self.qmax, 1)
        scale = torch.clamp(scale, min=self.eps)
        q = torch.round(weight / scale)
        q = torch.clamp(q, self.qmin, self.qmax)
        return q * scale


def quantize_linear_layer(
    linear: nn.Linear,
    num_bits: int = 4,
    block_size: int = 128,
    rotation_strategy: str = "rotation"
) -> SureQuantLinear:
    """将普通 Linear 层替换为量化 Linear 层
    
    Args:
        linear: 原始 nn.Linear 层
        num_bits: 量化位宽
        block_size: 分块大小
        rotation_strategy: 旋转策略 ("rotation" 或 "stiefel")
    
    Returns:
        SureQuantLinear: 量化后的线性层
    """
    # 创建量化器
    quantizer = SureQuantizer(
        dim=linear.in_features,
        block_size=block_size,
        num_bits=num_bits,
        rotation_strategy=rotation_strategy
    )
    
    # 创建量化线性层
    return SureQuantLinear(linear, quantizer)



def quantize_llava_model(
    model: LlavaForConditionalGeneration,
    num_bits: int = 4,
    block_size: int = 128,
    rotation_strategy: str = "rotation",
    quantize_vision: bool = True,
    quantize_mm_proj: bool = True,
    quantize_language: bool = True
) -> LlavaForConditionalGeneration:
    """量化 LLaVA 模型的激活和权重
    
    Args:
        model: 原始 LLaVA 模型
        num_bits: 量化位宽
        block_size: 分块大小
        rotation_strategy: 旋转策略
        quantize_vision: 是否量化视觉编码器
        quantize_language: 是否量化语言解码器
    
    Returns:
        LlavaForConditionalGeneration: 量化后的模型
    """
    weight_quantizer = WeightQuantizer(num_bits)

    if quantize_vision:
        print(">>>>> Quantizing vision model...")

        vision_model = model.vision_tower.vision_model.encoder.layers
        for name, module in vision_model.named_modules():
            if isinstance(module, nn.Linear):
                print(f"name: {name}, module: {module}")
                # 量化激活：替换为 SureQuantLinear
                quantized_linear = quantize_linear_layer(
                    module, num_bits, block_size, rotation_strategy
                )
                parent_module = get_parent_module(vision_model, name)
                set_attr_by_name(parent_module, name.split('.')[-1], quantized_linear)
                
                # 量化权重
                if hasattr(module, 'weight'):
                    d_type = module.weight.dtype
                    module.weight.data = weight_quantizer(module.weight.data).to(d_type)
    
    if quantize_mm_proj:
        print(">>>>> Quantizing multimodal projection...")
        mm_proj = model.multi_modal_projector
        for name, module in mm_proj.named_modules():
            if isinstance(module, nn.Linear):
                print(f"name: {name}, module: {module}")
                # 量化激活：替换为 SureQuantLinear
                quantized_linear = quantize_linear_layer(
                    module, num_bits, block_size, rotation_strategy
                )
                parent_module = get_parent_module(mm_proj, name)
                set_attr_by_name(parent_module, name.split('.')[-1], quantized_linear)
                
                # 量化权重
                if hasattr(module, 'weight'):
                    d_type = module.weight.dtype
                    module.weight.data = weight_quantizer(module.weight.data).to(d_type)


    if quantize_language:
        print(">>>>> Quantizing language model...")
        decoder = model.language_model.model.layers
        for name, module in decoder.named_modules():
            if isinstance(module, nn.Linear):
                # 跳过 lm_head（输出层通常不量化）
                if 'lm_head' in name:
                    continue

                print(f"name: {name}, module: {module}")
                
                # 量化激活：替换为 SureQuantLinear
                quantized_linear = quantize_linear_layer(
                    module, num_bits, block_size, rotation_strategy
                )
                parent_module = get_parent_module(decoder, name)
                set_attr_by_name(parent_module, name.split('.')[-1], quantized_linear)
                
                # 量化权重
                if hasattr(module, 'weight'):
                    d_type = module.weight.dtype
                    module.weight.data = weight_quantizer(module.weight.data).to(d_type)
    
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




def infer(model, processor, img_path):
    # Confirm generations of the quantized model look sane.
    print("========== SAMPLE GENERATION ==============")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please describe the animal in this image\n"},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    raw_image = Image.open(img_path)


    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(model.device)
    print(inputs.keys())
    print(f"inputs['input_ids'].shape: {inputs['input_ids'].shape}")
    # print(f"inputs['input_ids']: {inputs['input_ids']}")


    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128)
    print("output: ", output)
    print("output.shape: ", output.shape)
    
    print(processor.decode(output[0], skip_special_tokens=True))
    print("==========================================")
    return output[0]

# ------------------------------
# 无校准sure-quant
# ------------------------------
def example_usage():
    # 加载原始模型
    checkpoint = "/home/ecnu01/workspace/models/llava-1.5-7b-hf"

    model = LlavaForConditionalGeneration.from_pretrained(
        checkpoint,
        device_map='cuda',
        torch_dtype=torch.float16,
    )
    
    # 量化模型
    quantized_model = quantize_llava_model(
        model,
        num_bits=8,
        block_size=128,
        rotation_strategy="rotation"
    )
    quantized_model.to("cuda")

    # print(quantized_model)
    processor = AutoProcessor.from_pretrained(checkpoint)
    infer(quantized_model, processor, "/home/ecnu01/workspace/awq_learn/sample_img/sample1.jpg")
    infer(quantized_model, processor, "/home/ecnu01/workspace/awq_learn/sample_img/sample2.jpg")

    
    # # 保存量化模型
    # quantized_model.save_pretrained("./llava-surequant-4bit")
    
    # # 推理示例
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # quantized_model = quantized_model.to(device)
    # quantized_model.eval()
    
    # # 准备输入（图像 + 文本）
    # pixel_values = torch.randn(1, 3, 224, 224).to(device)  # 示例图像
    # input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(device)  # 示例文本
    
    # with torch.no_grad():
    #     outputs = quantized_model(
    #         pixel_values=pixel_values,
    #         input_ids=input_ids
    #     )
    
    # print("量化模型推理成功！")
    # print(f"输出 logits 形状: {outputs.logits.shape}")


if __name__ == "__main__":
    example_usage()
    print(">>>>>>>>>>>> done")
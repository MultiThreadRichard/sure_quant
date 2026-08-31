"""LLaVA Model Quantization with SureQuant."""
import sys
sys.path.append("/home/ecnu01/workspace/sure_quant")

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
                # print(f"name: {name}, module: {module}")
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
                # print(f"name: {name}, module: {module}")
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

                # print(f"name: {name}, module: {module}")
                
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


def collect_calib_data_from_full_model(
    model: LlavaForConditionalGeneration,
    processor,
    image_paths: list,
    prompts: list,
    device: torch.device
) -> dict:
    """收集模型各层的激活数据作为校准数据
    
    Args:
        model: LLaVA 模型
        processor: 图像/文本处理器
        image_paths: 校准图像路径列表
        prompts: 校准提示词列表
        device: 设备
    
    Returns:
        dict: 各层的激活数据，key为层名，value为tensor
    """
    model.eval()
    activation_dict = {}
    
    # 注册钩子收集激活
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(module, nn.Linear):
                # 收集量化前的输入激活
                if input[0] is not None:
                    if name not in activation_dict:
                        activation_dict[name] = []
                    activation_dict[name].append(input[0].detach().cpu())
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # 运行校准数据
    with torch.no_grad():
        for img_path, prompt in zip(image_paths, prompts):
            raw_image = Image.open(img_path)
            inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)
            # print(inputs.keys())
            # print(f"inputs['input_ids'].shape: {inputs['input_ids'].shape}")
            # print(f"inputs['pixel_values'].shape: {inputs['pixel_values'].shape}")
            # print(f"inputs['attention_mask'].shape: {inputs['attention_mask'].shape}")
            model(**inputs)
    
    # 移除钩子
    for h in hooks:
        h.remove()
    
    # 合并收集的数据
    for name in activation_dict:
        print(f"name: {name}, collected {len(activation_dict[name])}")
        # act_list = activation_dict[name]
        # for act in act_list:
        #     print(f"act.shape: {act.shape}")
        activation_dict[name] = torch.cat(activation_dict[name], dim=0)
    
    return activation_dict


def collect_calib_data_from_inputs(
    model: LlavaForConditionalGeneration,
    inputs_list: list,
) -> dict:
    """从预处理好的输入列表收集模型各层的激活数据作为校准数据
    
    Args:
        model: LLaVA 模型
        processor: 图像/文本处理器（保留参数以保持接口一致性）
        inputs_list: 预处理好的输入列表，每个元素是 processor 返回的字典
        device: 设备
    
    Returns:
        dict: 各层的激活数据，key为层名，value为tensor
    """
    model.eval()
    activation_dict = {}
    
    # 注册钩子收集激活
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(module, nn.Linear):
                # 收集量化前的输入激活
                if input[0] is not None:
                    if name not in activation_dict:
                        activation_dict[name] = []
                    activation_dict[name].append(input[0].detach().cpu())
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # 运行校准数据
    with torch.no_grad():
        for inputs in inputs_list:
            model(**inputs)
    
    # 移除钩子
    for h in hooks:
        h.remove()
    
    # 合并收集的数据
    for name in activation_dict:
        print(f"name: {name}, collected {len(activation_dict[name])}")
        activation_dict[name] = torch.cat(activation_dict[name], dim=0)
    
    return activation_dict


def eval_reconstruction_mse(rq: SureQuantizer, x: torch.Tensor):
    with torch.no_grad():
        out = rq(x)
        mse = reconstruction_loss(out["x_blk"], out["x_hat_blk"]).item()
    return mse


def calibrate_all_quantizers(
    quantized_model: LlavaForConditionalGeneration,
    calibration_data: dict,
    cfg: SureQuantConfig = None
) -> dict:
    """校准量化模型中所有的 SureQuantizer
    
    Args:
        quantized_model: 已量化的 LLaVA 模型
        calibration_data: 各层的校准数据
        cfg: 校准配置，默认使用默认配置
    
    Returns:
        dict: 各层的校准日志
    """
    if cfg is None:
        cfg = SureQuantConfig()
    
    device = next(quantized_model.parameters()).device
    logs_dict = {}
    
    # 遍历所有模块
    for name, module in quantized_model.named_modules():
        # if isinstance(module, nn.Linear):
        #     print(f"Linear name: {name}")
        if isinstance(module, SureQuantLinear):
            print(f"SureQuantLinear name: {name}")
            quantizer = module.sure_quantizer
            
            # 获取对应层的校准数据
            if name in calibration_data:
                print(f"\n===== Calibrating {name} =====")
                sample_data = calibration_data[name].to(device)
                
                # print(f"sample_data.shape: {sample_data.shape}, quantizer.dim: {quantizer.dim}")
                # break
                
                # 确保数据维度匹配
                if sample_data.shape[-1] != quantizer.dim:
                    raise ValueError(f"Calibration data dimension {sample_data.shape[-1]} does not match quantizer dimension {quantizer.dim} for layer {name}")
                
                sample_data = sample_data.view(-1, quantizer.dim)

                # 校准
                logs = calibrate_rotation(quantizer, sample_data, cfg)

                # mse_before = eval_reconstruction_mse(quantizer, sample_data)  # 评估初始MSE
                # logs = calibrate_rotation(quantizer, sample_data, cfg)
                # mse_after = eval_reconstruction_mse(quantizer, sample_data)  # 评估后MSE
                # print(f"  MSE before calibration: {mse_before:.6f}, after calibration: {mse_after:.6f}")

                logs_dict[name] = logs
                
                # break
            else:
                print(f"\n===== {name} has no calibration data, skipping =====")
    
    print(f">>>>>>>>  Calibration completed")
    return logs_dict


def prepare_calib_data():
    checkpoint = "/home/ecnu01/workspace/models/llava-1.5-7b-hf"

    # original model
    model = LlavaForConditionalGeneration.from_pretrained(
        checkpoint,
        device_map='cuda',
        torch_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(checkpoint)

    device = next(model.parameters()).device

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

    calib_data_dict = collect_calib_data_from_full_model(
        model,
        processor,
        image_paths=["/home/ecnu01/workspace/awq_learn/sample_img/sample1.jpg"],
        prompts=[prompt],
        device=device,
    )

    return calib_data_dict



def load_calib_data(calib_sample_num=128):
    checkpoint = "/home/ecnu01/workspace/models/llava-1.5-7b-hf"

    # original model
    model = LlavaForConditionalGeneration.from_pretrained(
        checkpoint,
        device_map='cuda',
        torch_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(checkpoint)

    device = next(model.parameters()).device


    data_path_list = [
        '/home/ecnu01/workspace/data/flickr30k/data/test-00000-of-00009.parquet',
    ]

    calib_dataset = load_dataset('parquet', data_files=data_path_list[0], split='train')
    print(f">>>>>>>> load dataset path: {data_path_list[0]}")
    # print(f'calib_dataset.column_names: {calib_dataset.column_names}')
    # print(f'len(calib_dataset): {len(calib_dataset)}')


    # 只取前 128 个样本
    calib_dataset = calib_dataset.select(range(calib_sample_num))

    # 只保留 image 列，移除其他所有列
    # columns_to_remove = [col for col in calib_dataset.column_names if col != 'image']
    # calib_dataset = calib_dataset.remove_columns(columns_to_remove)
    # print(f"保留 image 列后，剩余列: {calib_dataset.column_names}")

    print(f'len(calib_dataset): {len(calib_dataset)}')
    # print(type(calib_dataset))
    # print(calib_dataset[0])
    # print(calib_dataset[0]['image'])

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

    inputs_list = []
    for item in calib_dataset:
        raw_image = item['image']
        inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)
        # print(inputs.keys())
        # print(f"inputs['input_ids'].shape: {inputs['input_ids'].shape}")
        # print(f"inputs['pixel_values'].shape: {inputs['pixel_values'].shape}")
        # print(f"inputs['attention_mask'].shape: {inputs['attention_mask'].shape}")

        inputs_list.append(inputs)
        # break
    
    return collect_calib_data_from_inputs(
        model,
        inputs_list,
    )



    






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


def make_cfg() -> SureQuantConfig:
    cfg = SureQuantConfig()
    cfg.num_bits = 4
    cfg.block_size = 128
    cfg.calibration_steps = 10
    cfg.calibration_batch_size = 128
    cfg.calibration_lr = 0.01
    # cfg.lambda_dk = 0.0
    cfg.device = "cuda"
    return cfg



# ------------------------------
# 使用示例
# ------------------------------
def example_usage():
    calib_data_dict = prepare_calib_data()

    cfg = make_cfg()

    # 获取量化模型
    checkpoint = "/home/ecnu01/workspace/models/llava-1.5-7b-hf"

    model = LlavaForConditionalGeneration.from_pretrained(
        checkpoint,
        device_map='cuda',
        torch_dtype=torch.float16,
    )
    
    quantized_model = quantize_llava_model(
        model,
        num_bits=4,
        block_size=128,
        rotation_strategy="rotation"
    )
    quantized_model.to("cuda")
    # print(quantized_model)

    # 校准quantized_model
    logs_dict = calibrate_all_quantizers(quantized_model, calib_data_dict, cfg)


    processor = AutoProcessor.from_pretrained(checkpoint)
    infer(quantized_model, processor, "/home/ecnu01/workspace/awq_learn/sample_img/sample1.jpg")
    infer(quantized_model, processor, "/home/ecnu01/workspace/awq_learn/sample_img/sample2.jpg")


def example_calib():
    cfg = make_cfg()
    cfg.num_bits = 8

    calib_data_dict = load_calib_data(cfg.calibration_batch_size)

    checkpoint = "/home/ecnu01/workspace/models/llava-1.5-7b-hf"

    model = LlavaForConditionalGeneration.from_pretrained(
        checkpoint,
        device_map='cuda',
        torch_dtype=torch.float16,
    )
    
    quantized_model = quantize_llava_model(
        model,
        num_bits=8,
        block_size=128,
        rotation_strategy="rotation"
    )
    quantized_model.to("cuda")
    # print(quantized_model)

    # 校准quantized_model
    logs_dict = calibrate_all_quantizers(quantized_model, calib_data_dict, cfg)

    processor = AutoProcessor.from_pretrained(checkpoint)
    infer(quantized_model, processor, "/home/ecnu01/workspace/awq_learn/sample_img/sample1.jpg")
    infer(quantized_model, processor, "/home/ecnu01/workspace/awq_learn/sample_img/sample2.jpg")




if __name__ == "__main__":
    # example_usage()

    example_calib()
    print(">>>>>>>>>>>> done")
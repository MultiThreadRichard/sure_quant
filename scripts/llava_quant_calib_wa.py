"""LLaVA Model Quantization with SureQuant."""
import sys
import os
sys.path.append("/home/ecnu03/workspace/sure_quant")

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


def collect_calib_data_from_full_model(
    model: LlavaForConditionalGeneration,
    processor,
    image_paths: list,
    prompts: list,
    device: torch.device,
    max_samples_per_layer: int = 512,
) -> dict:
    """收集模型各层的激活数据作为校准数据"""
    model.eval()
    activation_dict = {}

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(module, nn.Linear):
                if input[0] is not None:
                    if name not in activation_dict:
                        activation_dict[name] = []
                    act = input[0].detach().cpu()
                    if act.dim() > 2:
                        act = act.view(-1, act.shape[-1])
                    activation_dict[name].append(act)
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        for img_path, prompt in zip(image_paths, prompts):
            raw_image = Image.open(img_path)
            inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)
            model(**inputs)

    for h in hooks:
        h.remove()

    for name in activation_dict:
        activation_dict[name] = torch.cat(activation_dict[name], dim=0)
        total_samples = activation_dict[name].shape[0]
        if total_samples > max_samples_per_layer:
            idx = torch.randperm(total_samples)[:max_samples_per_layer]
            activation_dict[name] = activation_dict[name][idx]
        print(f"name: {name}, collected {total_samples}, kept {activation_dict[name].shape[0]}")

    return activation_dict


def collect_calib_data_from_inputs(
    model: LlavaForConditionalGeneration,
    inputs_list: list,
    max_samples_per_layer: int = 512,
) -> dict:
    """从预处理好的输入列表收集模型各层的激活数据作为校准数据"""
    model.eval()
    activation_dict = {}

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(module, nn.Linear):
                if input[0] is not None:
                    if name not in activation_dict:
                        activation_dict[name] = []
                    act = input[0].detach().cpu()
                    if act.dim() > 2:
                        act = act.view(-1, act.shape[-1])
                    activation_dict[name].append(act)
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        for inputs in inputs_list:
            model(**inputs)

    for h in hooks:
        h.remove()

    for name in activation_dict:
        activation_dict[name] = torch.cat(activation_dict[name], dim=0)
        total_samples = activation_dict[name].shape[0]
        if total_samples > max_samples_per_layer:
            idx = torch.randperm(total_samples)[:max_samples_per_layer]
            activation_dict[name] = activation_dict[name][idx]
        print(f"name: {name}, collected {total_samples}, kept {activation_dict[name].shape[0]}")

    return activation_dict


def eval_reconstruction_mse(rq: SureQuantizer, x: torch.Tensor):
    with torch.no_grad():
        out = rq(x)
        mse = reconstruction_loss(out["x_blk"], out["x_hat_blk"]).item()
    return mse


def calibrate_weight_rotation(weight_quantizer, weight_data, cfg):
    """校准权重旋转量化器（使用 SureQuantizer，在 CPU 上执行以节省 GPU 内存）

    特别注意：该步骤在 GPU 上容易因显存不足触发 OOM，因此当前临时改为 CPU 执行。
    代价是整个量化校准流程会非常慢，完整流程可能超过 10+ 小时。

    Args:
        weight_quantizer: SureQuantizer 实例
        weight_data: 原始权重数据 [out_features, in_features]
        cfg: 校准配置

    Returns:
        dict: 校准日志
    """
    # 特别注意：GPU 显存不足会导致量化校准阶段 OOM，故此处强制使用 CPU 训练。
    # 该临时方案会显著增加总耗时，完整量化校准流程可能超过 10+ 小时。
    cpu_device = torch.device("cpu")
    weight_quantizer = weight_quantizer.to(cpu_device)
    weight_quantizer.train()

    optimizer = torch.optim.AdamW(weight_quantizer.rotation.parameters(), lr=cfg.calibration_lr)

    logs = {"weight_rec": []}

    weight_data_t = weight_data.cpu().T.contiguous()

    for step in range(cfg.calibration_steps):
        optimizer.zero_grad()

        out_dict = weight_quantizer(weight_data_t)
        w_hat = out_dict["x_hat"]
        w_blk = out_dict["x_blk"]
        w_hat_blk = out_dict["x_hat_blk"]

        rec_loss = reconstruction_loss(w_blk, w_hat_blk)

        rec_loss.backward()
        optimizer.step()

        logs["weight_rec"].append(rec_loss.item())

        if step % max(1, cfg.calibration_steps // 5) == 0:
            print(f"    Weight calibration step {step+1}/{cfg.calibration_steps}, rec={rec_loss.item():.6f}")

    weight_quantizer.eval()
    return logs


def calibrate_all_quantizers(
    quantized_model: LlavaForConditionalGeneration,
    calibration_data: dict,
    cfg: SureQuantConfig = None
) -> dict:
    """校准量化模型中所有的 SureQuantizer（激活和权重）"""
    if cfg is None:
        cfg = SureQuantConfig()

    device = next(quantized_model.parameters()).device
    logs_dict = {}

    import psutil

    layer_count = 0
    total_layers = sum(1 for _, m in quantized_model.named_modules() if isinstance(m, SureQuantLinear))

    for name, module in quantized_model.named_modules():
        if isinstance(module, SureQuantLinear):
            layer_count += 1
            print(f"SureQuantLinear name: {name} ({layer_count}/{total_layers})")

            cpu_mem = psutil.Process().memory_info().rss / 1e9
            print(f"  CPU memory: {cpu_mem:.2f}GB")

            if device.type == 'cuda':
                mem_allocated = torch.cuda.memory_allocated(device) / 1e9
                mem_reserved = torch.cuda.memory_reserved(device) / 1e9
                print(f"  GPU memory: allocated={mem_allocated:.2f}GB, reserved={mem_reserved:.2f}GB")

            layer_logs = {}

            activation_quantizer = module.activation_quantizer
            if name in calibration_data:
                print(f"\n===== Calibrating activation quantizer for {name} =====")
                try:
                    sample_data = calibration_data[name].to(device)

                    if sample_data.shape[-1] != activation_quantizer.dim:
                        raise ValueError(f"Calibration data dimension {sample_data.shape[-1]} does not match quantizer dimension {activation_quantizer.dim} for layer {name}")

                    sample_data = sample_data.view(-1, activation_quantizer.dim)

                    activation_logs = calibrate_rotation(activation_quantizer, sample_data, cfg)
                    layer_logs["activation"] = activation_logs

                except Exception as e:
                    print(f"  ERROR calibrating activation for {name}: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            else:
                print(f"\n===== {name} has no activation calibration data, skipping =====")

            if module.weight_quantizer is not None:
                print(f"\n===== Calibrating weight quantizer for {name} =====")
                try:
                    weight_data = module.linear.weight.data
                    weight_logs = calibrate_weight_rotation(module.weight_quantizer, weight_data, cfg)
                    layer_logs["weight"] = weight_logs

                    module.quantize_weight()

                except Exception as e:
                    print(f"  ERROR calibrating weight for {name}: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            else:
                print(f"\n===== {name} has no weight quantizer, skipping weight calibration =====")

            logs_dict[name] = layer_logs

            if device.type == 'cuda':
                torch.cuda.empty_cache()
                mem_allocated_after = torch.cuda.memory_allocated(device) / 1e9
                mem_reserved_after = torch.cuda.memory_reserved(device) / 1e9
                print(f"  GPU memory after empty_cache: allocated={mem_allocated_after:.2f}GB, reserved={mem_reserved_after:.2f}GB")

    print(f">>>>>>>>  Calibration completed")
    return logs_dict


def prepare_calib_data():
    checkpoint = "/home/ecnu03/workspace/models/llava-1.5-7b-hf"

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
        image_paths=["/home/ecnu03/workspace/awq_learn/sample_img/sample1.jpg"],
        prompts=[prompt],
        device=device,
    )

    return calib_data_dict


def load_calib_data(calib_sample_num=128):
    checkpoint = "/home/ecnu03/workspace/models/llava-1.5-7b-hf"

    model = LlavaForConditionalGeneration.from_pretrained(
        checkpoint,
        device_map='cuda',
        torch_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(checkpoint)

    device = next(model.parameters()).device

    data_path_list = [
        '/home/ecnu03/workspace/data/flickr30k/data/test-00000-of-00009.parquet',
    ]

    calib_dataset = load_dataset('parquet', data_files=data_path_list[0], split='train')
    print(f">>>>>>>> load dataset path: {data_path_list[0]}")

    calib_dataset = calib_dataset.select(range(calib_sample_num))

    print(f'len(calib_dataset): {len(calib_dataset)}')

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
        inputs_list.append(inputs)

    calib_data_dict = collect_calib_data_from_inputs(model, inputs_list)

    del model
    del inputs_list
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print(f">>>>>>>> released model memory after collecting calibration data")

    return calib_data_dict


def infer(model, processor, img_path):
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
    cfg.device = "cuda"
    return cfg


def example_usage():
    calib_data_dict = prepare_calib_data()

    cfg = make_cfg()
    cfg.num_bits = 4
    cfg.block_size = 128
    cfg.calibration_steps = 10
    cfg.calibration_batch_size = 128
    cfg.calibration_lr = 0.01
    cfg.rotation_strategy = "rotation"

    checkpoint = "/home/ecnu03/workspace/models/llava-1.5-7b-hf"

    model = LlavaForConditionalGeneration.from_pretrained(
        checkpoint,
        device_map='cuda',
        torch_dtype=torch.float16,
    )

    quantized_model = quantize_llava_model(
        model,
        num_bits=cfg.num_bits,
        block_size=cfg.block_size,
        rotation_strategy=cfg.rotation_strategy,
        quantize_weight=True
    )
    quantized_model.to("cuda")

    logs_dict = calibrate_all_quantizers(quantized_model, calib_data_dict, cfg)

    processor = AutoProcessor.from_pretrained(checkpoint)
    infer(quantized_model, processor, "/home/ecnu03/workspace/awq_learn/sample_img/sample1.jpg")
    infer(quantized_model, processor, "/home/ecnu03/workspace/awq_learn/sample_img/sample2.jpg")


def example_calib():
    cfg = make_cfg()
    cfg.num_bits = 4
    cfg.block_size = 128
    cfg.calibration_steps = 3
    cfg.calibration_batch_size = 128
    cfg.calibration_lr = 0.005
    cfg.rotation_strategy = "rotation"
    print(f"entry file:{os.path.abspath(__file__)}\n cfg: {cfg}")

    calib_data_dict = load_calib_data(cfg.calibration_batch_size)

    checkpoint = "/home/ecnu03/workspace/models/llava-1.5-7b-hf"

    model = LlavaForConditionalGeneration.from_pretrained(
        checkpoint,
        device_map='cuda',
        torch_dtype=torch.float16,
    )

    quantized_model = quantize_llava_model(
        model,
        num_bits=cfg.num_bits,
        block_size=cfg.block_size,
        rotation_strategy=cfg.rotation_strategy,
        quantize_weight=True
    )
    quantized_model.to("cuda")

    logs_dict = calibrate_all_quantizers(quantized_model, calib_data_dict, cfg)

    processor = AutoProcessor.from_pretrained(checkpoint)
    infer(quantized_model, processor, "/home/ecnu03/workspace/awq_learn/sample_img/sample1.jpg")
    infer(quantized_model, processor, "/home/ecnu03/workspace/awq_learn/sample_img/sample2.jpg")


if __name__ == "__main__":
    start_time = time.time()
    example_calib()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f">>>>>>>>>>>> done, elapsed time: {elapsed_time:.2f} seconds")
"""LLaVA Model Quantization with SureQuant."""
import sys
import os
sys.path.append("/home/ccwan/stu_Jiangtp/sure_quant")

import time
import torch
import torch.nn as nn
from transformers import AutoProcessor, LlavaForConditionalGeneration

from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

from model.sure_quantizer import SureQuantizer
from model.sure_quant_linear import SureQuantLinear

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
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": "A cat is in the image. Please answer yes or no."},
    #             {"type": "image"},
    #         ],
    #     },
    # ]
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": "Two dogs are in the image. Please answer yes or no."},
    #             {"type": "image"},
    #         ],
    #     },
    # ]
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



def load_dataset_from_local(path):
    trainset = load_dataset('parquet', data_files=path, split='train')
    # testset = load_dataset('parquet', data_files=path, split='test')
    print(f'len(trainset): {len(trainset)}')
    # print(type(trainset))
    # print(f'len(testset): {len(testset)}')
    # print(type(testset))

    messages = []
    for item in trainset:
        # print(item)
        # break
        # mme_data = {
        #     'question_id': 'code_reasoning/0020.png',
        #     'image': Image.open('path_to_image/code_reasoning/0020.png'),
        #     'question': 'Is a python code shown in the picture? Please answer yes or no.',
        #     'answer': 'Yes',
        #     'category': 'code_reasoning'
        # }

        msg_item = [{
            "role": "user",
            "content": [
                {"type": "image", "image": item['image']},
                {"type": "text", "text": item['question']}
            ]
        }]
        messages.append(msg_item)

    return trainset, messages


def speed_compute(input_len, generate_len, t_elapsed) -> str:
    new_generated_tokens = generate_len - input_len
    return new_generated_tokens / t_elapsed


def average_data_list(float_list):
    if len(float_list) == 0:
        return 0
    return sum(float_list) / len(float_list)


def mme_test(model, processor):
    # TO MOD
    data_path_list = [
        '/home/ccwan/stu_Jiangtp/data/MME/data/test-00000-of-00004-a25dbe3b44c4fda6.parquet',
        '/home/ccwan/stu_Jiangtp/data/MME/data/test-00001-of-00004-7d22c7f1aba6fca4.parquet',
        '/home/ccwan/stu_Jiangtp/data/MME/data/test-00002-of-00004-594798fd3f5b029c.parquet',
        '/home/ccwan/stu_Jiangtp/data/MME/data/test-00003-of-00004-53ae1794f93b1e35.parquet',
    ]


    # TO MOD
    output_path = '/home/ccwan/stu_Jiangtp/sure_quant/logs/mme_eval_res'
    os.makedirs(output_path, exist_ok=True)

    turn = 0
    speed_list = []

    t_benchmark_start = time.perf_counter()
    for data_path in data_path_list:
        t_data, messages = load_dataset_from_local(data_path)
        print(f'>>>>>>>>> load {data_path}')
        # break

        print('>>>>>>>>> start eval')
        mode = 'a'
        sp_list = []
        with open(os.path.join(output_path, f'eval_results0{turn}.txt'), mode, encoding="utf-8") as fout:
            for item, msg_item in tqdm(zip(t_data, messages)):
                # torch.cuda.empty_cache()

                # 使用 processor 处理输入
                text = processor.apply_chat_template(msg_item, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(msg_item)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to("cuda")
                
                # print(inputs)
                # print(type(inputs)) # <class 'transformers.feature_extraction_utils.BatchFeature'>
                # print(inputs.keys())
                # print(f"inputs['input_ids'].shape: {inputs['input_ids'].shape}")
                # print(f"inputs['attention_mask'].shape: {inputs['attention_mask'].shape}")
                # print(f"inputs['pixel_values'].shape: {inputs['pixel_values'].shape}")
                # print(f"inputs['image_grid_thw'].shape: {inputs['image_grid_thw'].shape}")

                start = time.perf_counter()

                generated_ids = model.generate(**inputs, max_new_tokens=128)
                # generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.1)

                
                # print(f"generated_ids.shape: {generated_ids.shape}")

                t_elapsed = time.perf_counter() - start

                sp_list.append(speed_compute(inputs['input_ids'].shape[-1], generated_ids.shape[-1], t_elapsed))

                response = processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                # 打印结果
                # print("Generated Response:", response)

                print(item['category'], item['question_id'], item['question'], item['answer'], response, sep='\t', file=fout)
                # break
        
        speed_list.append(average_data_list(sp_list))

        print(f'>>>>>>>>> end eval')
        torch.cuda.empty_cache()
        turn += 1

        # break

    t_benchmark_end = time.perf_counter()

    print(f'>>>>>>>>> complete turn: {turn}')
    print(f'>>>>>>>>> total elapsed time: {t_benchmark_end - t_benchmark_start} s')
    print(f'average infer speed: {average_data_list(speed_list):.2f} token/s')



if __name__ == "__main__":
    save_path = "/home/ccwan/stu_Jiangtp/sure_quant/model_saved/llava_7b_sure_calib_4bit_blk128"

    loaded_model = load_quantized_model(save_path, device="cuda")
    processor = AutoProcessor.from_pretrained("/home/ccwan/stu_Jiangtp/model_repo/llava-7b-hf")

    # infer(loaded_model, processor, "/home/ccwan/stu_Jiangtp/MQuant/assert/sample1.jpg")
    # infer(loaded_model, processor, "/home/ccwan/stu_Jiangtp/MQuant/assert/sample2.jpg")

    mme_test(loaded_model, processor)

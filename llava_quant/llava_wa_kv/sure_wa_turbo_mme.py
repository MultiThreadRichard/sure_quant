#!/usr/bin/env python3
"""
quantized inference using SureQuant (weight) + TurboQuant (KV cache).
mme test
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
import time

import torch
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
print(f"PROJECT_ROOT: {_PROJECT_ROOT}")
sys.path.insert(0, str(_PROJECT_ROOT))
# sys.path.insert(0, str(_PROJECT_ROOT / "llava_quant"))
# sys.path.insert(0, str(_PROJECT_ROOT / "mme"))

from llava_quant.llava_wa.persistence import load_quantized_model
from llava_quant.llava_wa.config import (
    PATH_PREFIX,
)




MME_DATA_PATH_LIST = [
    f'{PATH_PREFIX}/workspace/data/MME/data/test-00000-of-00004-a25dbe3b44c4fda6.parquet',
    f'{PATH_PREFIX}/workspace/data/MME/data/test-00001-of-00004-7d22c7f1aba6fca4.parquet',
    f'{PATH_PREFIX}/workspace/data/MME/data/test-00002-of-00004-594798fd3f5b029c.parquet',
    f'{PATH_PREFIX}/workspace/data/MME/data/test-00003-of-00004-53ae1794f93b1e35.parquet',
]

# MME_OUTPUT_PATH = f"{PATH_PREFIX}/workspace/sure_quant/logs/mme_eval_res"
MME_OUTPUT_PATH = f"{PATH_PREFIX}/workspace/sure_quant/logs/mme_eval_res_sure_llm_w4a16_turbo_kv4"


# DEFAULT_MODEL_DIR = Path("/home/ecnu01/sure_quant_models/w4a16_language_only/20260823/best_quantized_model")
DEFAULT_MODEL_DIR = Path("/home/ecnu01/workspace/sure_quant/model_saved/llava_7b_surequant_llm_w4a16/best_quantized_model")
# DEFAULT_MODEL_DIR = Path("/home/ecnu01/workspace/sure_quant/model_saved/llava_7b_surequant_w4a16/best_quantized_model")




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="W4A16KV4 inference: SureQuant W4A16 weights + TurboQuant KV4 cache"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Path to saved W4A16 quantized model directory",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image (optional, uses test image if not provided)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Please describe this image in detail.",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--k-bits",
        type=int,
        default=4,
        help="KV cache K quantization bits (default: 4)",
    )
    parser.add_argument(
        "--v-bits",
        type=int,
        default=4,
        help="KV cache V quantization bits (default: 4)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Enable KV cache quantization error evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda/cpu)",
    )
    return parser.parse_args()


def load_w4a16_model(model_dir: str, device: str):
    """Load W4A16 quantized model from saved checkpoint."""

    model_path = Path(model_dir)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Quantized model directory not found: {model_path}")

    config_file = model_path / "surequant_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"surequant_config.json not found in {model_path}")

    print(f"[W4A16] Loading quantized model from: {model_path}")
    model = load_quantized_model(
        model_path,
        device_map=device,
        torch_dtype=torch.float16,
    )
    model.eval()
    print(f"[W4A16] Model loaded successfully on {device}")
    return model


def load_processor(model_dir: str):
    """Load processor for the quantized model."""
    from transformers import AutoProcessor

    print(f"[Processor] Loading from: {model_dir}")
    processor = AutoProcessor.from_pretrained(model_dir)
    return processor


def run_inference(
    model,
    processor,
    image_path: str | None,
    prompt: str,
    max_new_tokens: int,
    k_bits: int,
    v_bits: int,
    evaluate: bool,
):
    """Run W4A16KV4 inference using LLaVAInferEngine."""
    from mme.llava_kv_quant_turbo import LLaVAInferEngine

    # Override KV quantizer bits if different from default
    engine = LLaVAInferEngine(model, processor)
    if hasattr(engine.kv_quant, "k_bits"):
        engine.kv_quant.k_bits = k_bits
        engine.kv_quant.v_bits = v_bits
        print(f"[KV4] KV quantizer configured: K={k_bits}-bit, V={v_bits}-bit")

    # Load image
    sample_image_path = _PROJECT_ROOT / "sample_img" / "cat2.jpg"
    if image_path:
        raw_image = Image.open(image_path).convert("RGB")
        print(f"[Input] Image: {image_path}")
    elif sample_image_path.exists():
        raw_image = Image.open(str(sample_image_path)).convert("RGB")
        print(f"[Input] Using sample image: {sample_image_path}")
    else:
        raw_image = Image.new("RGB", (336, 336), color=(128, 128, 128))
        print("[Input] Using dummy image (no image provided)")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        },
    ]
    print(f"[Input] Prompt: {prompt}")
    print(f"[Inference] max_new_tokens={max_new_tokens}, evaluate={evaluate}")
    print("-" * 60)

    output_text = engine.generate(
        raw_image=raw_image,
        messages=messages,
        max_new_tokens=max_new_tokens,
        need_eval=evaluate,
    )

    print("-" * 60)
    print(f"[Output] {output_text}")
    return output_text


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


def mme_test(model, processor, args):
    from mme.llava_kv_quant_turbo import LLaVAInferEngine

    engine = LLaVAInferEngine(model, processor)

    # TO MOD
    output_path = MME_OUTPUT_PATH
    os.makedirs(output_path, exist_ok=True)

    turn = 0

    for data_path in MME_DATA_PATH_LIST:
        t_data, messages = load_dataset_from_local(data_path)
        print(f'>>>>>>>>> load {data_path}')
        # break

        print('>>>>>>>>> start eval')
        mode = 'a'
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

                generated_ids = engine.generate_for_mme(
                    inputs=inputs,
                    max_new_tokens=args.max_new_tokens,
                )
                # generated_ids = model.generate(**inputs, max_new_tokens=128)
                # generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.1)

                
                # print(f"generated_ids.shape: {generated_ids.shape}")

                response = processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                # 打印结果
                # print("Generated Response:", response)

                print(item['category'], item['question_id'], item['question'], item['answer'], response, sep='\t', file=fout)
                # break


        print(f'>>>>>>>>> end eval')
        torch.cuda.empty_cache()
        turn += 1

        # break

    print(f'>>>>>>>>> mme complete turn: {turn}')


def main():
    args = parse_args()

    print("=" * 60)
    print("W4A16KV4 Quantized Inference")
    print("=" * 60)
    print(f"  Model: {args.model_dir}")
    print(f"  Device: {args.device}")
    print(f"  Weight: 4-bit (SureQuant INT4)")
    print(f"  Activation: 16-bit (FP16)")
    print(f"  KV Cache: K={args.k_bits}-bit, V={args.v_bits}-bit (TurboQuant)")
    print("=" * 60)

    # Step 1: Load W4A16 quantized model
    model = load_w4a16_model(args.model_dir, args.device)

    # Step 2: Load processor
    processor = load_processor(args.model_dir)

    # Step 3: mme test
    mme_test(model, processor, args)


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f">>>>>>>>>>>>> Done, elapsed time: {elapsed:.2f} seconds")
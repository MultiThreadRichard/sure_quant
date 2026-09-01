import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
print(f"REPO_ROOT: {REPO_ROOT}")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time
import gc
import json
import argparse
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

from llava_quant.llava_wa.config import (
    PATH_PREFIX,
    DEFAULT_INFERENCE_PROMPT,
    build_parser,
)
from llava_quant.llava_wa.utils import (
    compute_kl_for_quantization,
    compute_cos_similarity,
    compute_pearson_correlation,
)
from llava_quant.llava_wa.data import (
    make_prompt,
)

from llava_quant.llava_wa.modeling_fp4 import load_quantized_model_fp4

"""
sample test
加载后fp4模型, val_small评估

CUDA_VISIBLE_DEVICES=1 nohup python scripts/test_val_small_fp4.py > logs/val_small_res/flickr_fp4_01.log 2>&1 &

"""


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
CHECKPOINT = f"{PATH_PREFIX}/workspace/models/llava-1.5-7b-hf"
# SAMPLE_IMG_DIR = f"{PATH_PREFIX}/workspace/awq_learn/sample_img"
CALIB_DATA_PATH = f"{PATH_PREFIX}/workspace/data/flickr30k/data/test-00000-of-00009.parquet"
# SAVE_ID = "02"
# DEFAULT_SAVE_DIR = f"{PATH_PREFIX}/workspace/sure_quant/model_saved/llava_7b_sure_fp4_{SAVE_ID}"

MME_DATA_PATH_LIST = [
    f'{PATH_PREFIX}/workspace/data/MME/data/test-00000-of-00004-a25dbe3b44c4fda6.parquet',
    f'{PATH_PREFIX}/workspace/data/MME/data/test-00001-of-00004-7d22c7f1aba6fca4.parquet',
    f'{PATH_PREFIX}/workspace/data/MME/data/test-00002-of-00004-594798fd3f5b029c.parquet',
    f'{PATH_PREFIX}/workspace/data/MME/data/test-00003-of-00004-53ae1794f93b1e35.parquet',
]

MME_OUTPUT_PATH = f"{PATH_PREFIX}/workspace/sure_quant/logs/mme_eval_res"

# TO TEST
# SAMPLE_IMG_DIR = f"{PATH_PREFIX}/workspace/sure_quant/sample_img"

# SAMPLE_PATH_LIST = [
#     f"{SAMPLE_IMG_DIR}/two_dogs.jpg",
#     f"{SAMPLE_IMG_DIR}/cat1.jpg",
#     f"{SAMPLE_IMG_DIR}/cat2.jpg",
#     f"{SAMPLE_IMG_DIR}/car.jpg",
#     f"{SAMPLE_IMG_DIR}/backyard.png",
#     f"{SAMPLE_IMG_DIR}/men.png",
# ]

# fp4
QMODEL_PATH = "/home/ecnu01/workspace/sure_quant/logs/search_fp4_mse02/best_quantized_model"

# VAL_DIR = Path("/home/ecnu01/workspace/sure_quant/val_small")
VAL_DIR = Path("/home/ecnu01/workspace/sure_quant/logs/flick_figs")

SAMPLE_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".webp", ".png"}


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
def infer(
    model: LlavaForConditionalGeneration,
    processor: Any,
    img_path,
    prompt_text: str = "Please describe this image\n",
    max_new_tokens: int = 128,
) -> torch.Tensor:
    """Run inference on a single image and print the result."""
    # print("========== SAMPLE GENERATION ============")
    prompt = make_prompt(processor, prompt_text)
    raw_image = Image.open(img_path)
    device = next(model.parameters()).device

    inputs = processor(
        images=raw_image, text=prompt, return_tensors="pt",
    ).to(device)
    # print(f"inputs['input_ids'].shape: {inputs['input_ids'].shape}")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = processor.decode(output[0], skip_special_tokens=True)
    # print(f"Generated: {decoded}")
    # print("==========================================")
    return output[0], decoded


def run_saved_model_fp4() -> None:
    """Load a saved quantized model and run inference."""
    save_path = QMODEL_PATH
    print(f"\n========== Loading saved quantized model from {save_path} ==========")

    loaded_model = load_quantized_model_fp4(
        save_path, device_map="cuda", torch_dtype=torch.float16,
    )

    processor = AutoProcessor.from_pretrained(save_path)

    image_paths: list[Path] = sorted(
        p for p in VAL_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in SAMPLE_IMAGE_SUFFIXES
    )

    out_list = []
    for img_path in image_paths:
        output_ids, out_txt = infer(loaded_model, processor, img_path)
        out_list.append({'img_path': img_path, 'output_ids': output_ids, 'out_txt': out_txt})
    return out_list


def compare_with_full_model(out_list):
    model = LlavaForConditionalGeneration.from_pretrained(
        CHECKPOINT, device_map="cuda", torch_dtype=torch.float16,
    )

    processor = AutoProcessor.from_pretrained(CHECKPOINT)

    for item in out_list:
        img_path = item['img_path']
        output_ids = item['output_ids']
        out_txt = item['out_txt']
        output_ids_full, out_txt_full = infer(model, processor, img_path)

        print(f">>>>>>>> img_path: {img_path}")
        print(f"before quant: {out_txt_full}")
        print(f"after quant: {out_txt}")
        
        kl = compute_kl_for_quantization(output_ids_full, output_ids)
        print(f"kl: {kl}")
        cos_sim = compute_cos_similarity(output_ids_full, output_ids)
        print(f"cos_sim: {cos_sim}")
        pcc = compute_pearson_correlation(output_ids_full, output_ids)
        print(f"pearson corr: {pcc}")
        print()




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



def mme_test(model, processor):
    # TO MOD
    # data_path_list = [
    #     '/home/ccwan/stu_Jiangtp/data/MME/data/test-00000-of-00004-a25dbe3b44c4fda6.parquet',
    #     '/home/ccwan/stu_Jiangtp/data/MME/data/test-00001-of-00004-7d22c7f1aba6fca4.parquet',
    #     '/home/ccwan/stu_Jiangtp/data/MME/data/test-00002-of-00004-594798fd3f5b029c.parquet',
    #     '/home/ccwan/stu_Jiangtp/data/MME/data/test-00003-of-00004-53ae1794f93b1e35.parquet',
    # ]

    # TO MOD
    # output_path = '/home/ccwan/stu_Jiangtp/sure_quant/logs/mme_eval_res'
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

                generated_ids = model.generate(**inputs, max_new_tokens=128)
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



def run_mme_fp4():
    save_path = QMODEL_PATH
    print(f"\n========== Loading saved quantized model from {save_path} ==========")

    loaded_model = load_quantized_model_fp4(
        save_path, device_map="cuda", torch_dtype=torch.float16,
    )

    processor = AutoProcessor.from_pretrained(save_path)

    mme_test(loaded_model, processor)



# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()
    print(args)

    out_list = run_saved_model_fp4()
    compare_with_full_model(out_list)




if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f">>>>>>>>>>>>> Done, elapsed time: {elapsed:.2f} seconds")
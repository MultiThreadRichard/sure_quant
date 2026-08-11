import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
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

from scripts.llava_wa.config import (
    PATH_PREFIX,
    DEFAULT_INFERENCE_PROMPT,
    build_parser,
)
from scripts.llava_wa.modeling import (
    quantize_linear_layer,
    quantize_llava_model,
    selected_linear_names,
)
from scripts.llava_wa.calibration import (
    calibrate_all_quantizers,
    reconstruction_score,
)
from scripts.llava_wa.data import (
    collect_calibration_data,
    split_calibration_data,
    make_prompt,
    generate_assistant_outputs,
)
from scripts.llava_wa.persistence import save_quantized_model, load_quantized_model, _jsonable_config
from scripts.llava_wa.search import seed_everything


"""
加载后int4模型, mme评估
"""


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
CHECKPOINT = f"{PATH_PREFIX}/workspace/models/llava-1.5-7b-hf"
SAMPLE_IMG_DIR = f"{PATH_PREFIX}/workspace/awq_learn/sample_img"
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
SAMPLE_PATH_LIST = [
    f"{PATH_PREFIX}/workspace/sure_quant/sample_img/two_dogs.jpg",
    f"{PATH_PREFIX}/workspace/sure_quant/sample_img/cat1.jpg",
]


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
def infer(
    model: LlavaForConditionalGeneration,
    processor: Any,
    img_path: str,
    prompt_text: str = DEFAULT_INFERENCE_PROMPT,
    max_new_tokens: int = 128,
) -> torch.Tensor:
    """Run inference on a single image and print the result."""
    print("========== SAMPLE GENERATION ============")
    prompt = make_prompt(processor, prompt_text)
    raw_image = Image.open(img_path)
    device = next(model.parameters()).device

    inputs = processor(
        images=raw_image, text=prompt, return_tensors="pt",
    ).to(device)
    print(f"inputs['input_ids'].shape: {inputs['input_ids'].shape}")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = processor.decode(output[0], skip_special_tokens=True)
    print(f"Generated: {decoded}")
    print("==========================================")
    return output[0]


def run_saved_model_int4() -> None:
    """Load a saved quantized model and run inference."""
    save_path = "/home/ecnu01/sure_quant_models/20260803/best_quantized_model"
    print(f"\n========== Loading saved quantized model from {save_path} ==========")

    loaded_model = load_quantized_model(
        save_path, device_map="cuda", torch_dtype=torch.float16,
    )

    processor = AutoProcessor.from_pretrained(save_path)

    # infer(loaded_model, processor, os.path.join(SAMPLE_IMG_DIR, "sample1.jpg"))
    # infer(loaded_model, processor, os.path.join(SAMPLE_IMG_DIR, "sample2.jpg"))
    for img_path in SAMPLE_PATH_LIST:
        infer(loaded_model, processor, img_path)




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



def run_mme_int4():
    save_path = "/home/ecnu01/sure_quant_models/20260803/best_quantized_model"
    print(f"\n========== Loading saved quantized model from {save_path} ==========")

    loaded_model = load_quantized_model(
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

    run_saved_model_int4()

    # run_mme_int4()



if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f">>>>>>>>>>>>> Done, elapsed time: {elapsed:.2f} seconds")
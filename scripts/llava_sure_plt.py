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

from scripts.plt_tools import collect_llava_vision_input_activations, plt_llava_vision_activation, plt_llava_vision_weight


"""
加载后surequant模型, 可视化
"""


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
CHECKPOINT = f"{PATH_PREFIX}/workspace/models/llava-1.5-7b-hf"
SAMPLE_IMG_DIR = f"{PATH_PREFIX}/workspace/awq_learn/sample_img"
# CALIB_DATA_PATH = f"{PATH_PREFIX}/workspace/data/flickr30k/data/test-00000-of-00009.parquet"
# SAVE_ID = "02"
# DEFAULT_SAVE_DIR = f"{PATH_PREFIX}/workspace/sure_quant/model_saved/llava_7b_sure_fp4_{SAVE_ID}"




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

    infer(loaded_model, processor, os.path.join(SAMPLE_IMG_DIR, "sample1.jpg"))
    infer(loaded_model, processor, os.path.join(SAMPLE_IMG_DIR, "sample2.jpg"))



def analyze_llava_activation_after_surequant(
    model: LlavaForConditionalGeneration,
    processor: Any,
    output_path: str,
    img_path: str,
    prompt_text: str = DEFAULT_INFERENCE_PROMPT,
):
    prompt = make_prompt(processor, prompt_text)
    raw_image = Image.open(img_path)
    device = next(model.parameters()).device

    inputs = processor(
        images=raw_image, text=prompt, return_tensors="pt",
    ).to(device)
    print(f"inputs['input_ids'].shape: {inputs['input_ids'].shape}")


    # paint activations vision
    activations = collect_llava_vision_input_activations(model, inputs)
    print(f"activations len: {len(activations)}")
    print(f"activations[0].shape: {activations[0].shape}")

    plt_llava_vision_activation(activations, output_path=output_path)

    print("==========================================")


def analyze_llava_weight_after_surequant(
    model: LlavaForConditionalGeneration,
    output_path: str,
):
    plt_llava_vision_weight(model, output_path=output_path)



# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    save_path = "/home/ecnu01/sure_quant_models/20260803/best_quantized_model"
    print(f"\n========== Loading saved quantized model from {save_path} ==========")

    loaded_model = load_quantized_model(
        save_path, device_map="cuda", torch_dtype=torch.float16,
    )

    processor = AutoProcessor.from_pretrained(save_path)

    # activation
    # sure_a_vis_int4_path = "/home/ecnu01/workspace/sure_quant/logs/sure_a_vis_int4"
    # os.makedirs(sure_a_vis_int4_path, exist_ok=True)

    # analyze_llava_activation_after_surequant(
    #     loaded_model,
    #     processor,
    #     output_path=sure_a_vis_int4_path,
    #     img_path=os.path.join(SAMPLE_IMG_DIR, "sample1.jpg")
    # )

    # weight
    sure_w_vis_int4_path = "/home/ecnu01/workspace/sure_quant/logs/sure_w_vis_int4"
    os.makedirs(sure_w_vis_int4_path, exist_ok=True)

    analyze_llava_weight_after_surequant(
        loaded_model,
        output_path=sure_w_vis_int4_path,
    )





if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f">>>>>>>>>>>>> Done, elapsed time: {elapsed:.2f} seconds")
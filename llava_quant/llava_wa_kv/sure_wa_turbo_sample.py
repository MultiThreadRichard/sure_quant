#!/usr/bin/env python3
"""
SureQuant (weight) W4A16 + TurboQuant (KV cache) quantized inference, sample test.

Mirrors the sample-test logic in tests/test_mme_int4.py (run_saved_model_int4 +
compare_with_full_model), but drives inference through LLaVAInferEngine so the
KV cache is quantized by TurboQuant during generation.

CUDA_VISIBLE_DEVICES=0 nohup python llava_quant/llava_wa_kv/sure_wa_turbo_sample.py \
    > logs/sure_wa_turbo_sample.log 2>&1 &
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

from llava_quant.llava_wa.config import (
    PATH_PREFIX,
    DEFAULT_INFERENCE_PROMPT,
)
from llava_quant.llava_wa.utils import (
    compute_kl_for_quantization,
    compute_cos_similarity,
    compute_pearson_correlation,
)
from llava_quant.llava_wa.data import make_prompt
from llava_quant.llava_wa.persistence import load_quantized_model
from llava_quant.llava_wa.search import seed_everything
from mme.llava_kv_quant_turbo import LLaVAInferEngine


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
CHECKPOINT = f"{PATH_PREFIX}/workspace/models/llava-1.5-7b-hf"
CALIB_DATA_PATH = f"{PATH_PREFIX}/workspace/data/flickr30k/data/test-00000-of-00009.parquet"

MME_DATA_PATH_LIST = [
    f'{PATH_PREFIX}/workspace/data/MME/data/test-00000-of-00004-a25dbe3b44c4fda6.parquet',
    f'{PATH_PREFIX}/workspace/data/MME/data/test-00001-of-00004-7d22c7f1aba6fca4.parquet',
    f'{PATH_PREFIX}/workspace/data/MME/data/test-00002-of-00004-594798fd3f5b029c.parquet',
    f'{PATH_PREFIX}/workspace/data/MME/data/test-00003-of-00004-53ae1794f93b1e35.parquet',
]

MME_OUTPUT_PATH = f"{PATH_PREFIX}/workspace/sure_quant/logs/mme_eval_res"

# TO TEST
SAMPLE_IMG_DIR = f"{PATH_PREFIX}/workspace/sure_quant/sample_img"

SAMPLE_PATH_LIST = [
    f"{SAMPLE_IMG_DIR}/two_dogs.jpg",
    f"{SAMPLE_IMG_DIR}/cat1.jpg",
    f"{SAMPLE_IMG_DIR}/cat2.jpg",
    f"{SAMPLE_IMG_DIR}/car.jpg",
    f"{SAMPLE_IMG_DIR}/backyard.png",
    f"{SAMPLE_IMG_DIR}/men.png",
]

# int4
# QMODEL_PATH = "/home/ecnu01/sure_quant_models/20260808/best_quantized_model"
# QMODEL_PATH = "/home/ecnu01/sure_quant_models/w4a16_language_only/best_quantized_model"
# QMODEL_PATH = "/home/ecnu01/sure_quant_models/w4a16_language_only/20260824/best_quantized_model"
# QMODEL_PATH = "/home/ecnu01/workspace/sure_quant/model_saved/llava_7b_surequant_w4a16/best_quantized_model"
QMODEL_PATH = "/home/ecnu01/workspace/sure_quant/model_saved/llava_7b_surequant_llm_w4a16/best_quantized_model"

# Sampling seeds for do_sample=True inference
SEED_LIST = [0, 1, 2, 3, 42]


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def infer(
    model: LlavaForConditionalGeneration,
    processor: Any,
    img_path: str,
    prompt_text: str = "Please describe this image\n",
    max_new_tokens: int = 128,
    do_sample: bool = False,
) -> tuple[torch.Tensor, str]:
    """Full-precision baseline inference (unchanged from tests/test_mme_int4.py)."""
    prompt = make_prompt(processor, prompt_text)
    raw_image = Image.open(img_path)
    device = next(model.parameters()).device

    inputs = processor(
        images=raw_image, text=prompt, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
    decoded = processor.decode(output[0], skip_special_tokens=True)
    return output[0], decoded


def infer_with_engine(
    engine: LLaVAInferEngine,
    processor: Any,
    img_path: str,
    prompt_text: str = "Please describe this image\n",
    max_new_tokens: int = 128,
    do_sample: bool = False,
) -> tuple[torch.Tensor, str]:
    """KV-cache-quantized inference via LLaVAInferEngine (TurboQuant KV4)."""
    prompt = make_prompt(processor, prompt_text)
    raw_image = Image.open(img_path)
    device = next(engine.model.parameters()).device

    inputs = processor(
        images=raw_image, text=prompt, return_tensors="pt",
    ).to(device)

    generated_ids = engine.generate_for_mme(inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
    decoded = processor.decode(generated_ids[0], skip_special_tokens=True)
    return generated_ids[0], decoded


def run_saved_model_kv(do_sample: bool = False) -> list[dict[str, Any]]:
    """Load a saved W4A16 quantized model and run KV-quantized inference."""
    save_path = QMODEL_PATH
    print(f"\n========== Loading saved quantized model from {save_path} ==========")

    loaded_model = load_quantized_model(
        save_path, device_map="cuda", torch_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(save_path)

    engine = LLaVAInferEngine(loaded_model, processor)

    out_list = []
    for img_path in SAMPLE_PATH_LIST:
        output_ids, out_txt = infer_with_engine(engine, processor, img_path, do_sample=do_sample)
        out_list.append({'img_path': img_path, 'output_ids': output_ids, 'out_txt': out_txt})
    return out_list


def compare_with_full_model(out_list: list[dict[str, Any]], do_sample: bool = False) -> None:
    model = LlavaForConditionalGeneration.from_pretrained(
        CHECKPOINT, device_map="cuda", torch_dtype=torch.float16,
    )

    processor = AutoProcessor.from_pretrained(CHECKPOINT)

    for item in out_list:
        img_path = item['img_path']
        output_ids = item['output_ids']
        out_txt = item['out_txt']
        output_ids_full, out_txt_full = infer(model, processor, img_path, do_sample=do_sample)

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


# def make_messages(prompt_text: str) -> list[dict[str, Any]]:
#     """Build a LLaVA chat messages list (image placeholder resolved by the engine)."""
#     return [{"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image"}]}]


# def run_saved_model_kv_sample(
#     seed_list: list[int] | None = None,
#     prompt_text: str = "Please describe this image\n",
#     max_new_tokens: int = 128,
# ) -> None:
#     """Load a saved W4A16 model and run sampling (do_sample=True) inference under different seeds."""
#     save_path = QMODEL_PATH
#     print(f"\n========== Loading saved quantized model from {save_path} ==========")

#     loaded_model = load_quantized_model(
#         save_path, device_map="cuda", torch_dtype=torch.float16,
#     )
#     processor = AutoProcessor.from_pretrained(save_path)
#     engine = LLaVAInferEngine(loaded_model, processor)

#     seeds = seed_list if seed_list is not None else SEED_LIST
#     for img_path in SAMPLE_PATH_LIST:
#         raw_image = Image.open(img_path).convert("RGB")
#         print(f"\n===== img_path: {img_path} =====")
#         for seed in seeds:
#             seed_everything(seed)
#             out_txt = engine.generate_sample(
#                 raw_image=raw_image,
#                 messages=make_messages(prompt_text),
#                 max_new_tokens=max_new_tokens,
#             )
#             print(f"[seed {seed}] {out_txt}")

def run_saved_model_kv_sample(
    seed_list: list[int] | None = None,
) -> None:
    for seed_id in seed_list:
        print(f">>>>>>>>>> seed: {seed_id}")
        seed_everything(seed_id)

        out_list = run_saved_model_kv(do_sample=True)
        compare_with_full_model(out_list, do_sample=True)



# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    # out_list = run_saved_model_kv()
    # compare_with_full_model(out_list)

    run_saved_model_kv_sample([10, 20, 30, 40, 50])


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f">>>>>>>>>>>>> Done, elapsed time: {elapsed:.2f} seconds")

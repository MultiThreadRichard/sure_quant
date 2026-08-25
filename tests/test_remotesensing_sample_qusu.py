import sys
import os
from pathlib import Path
import torch.nn.functional as F
import getpass

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




"""
remotesensing sample test
CUDA_VISIBLE_DEVICES=0 python tests/test_remotesensing_sample_qusu.py

CUDA_VISIBLE_DEVICES=0 nohup python -u tests/test_remotesensing_sample_qusu.py > tests/remotesensing_sample01.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u tests/test_remotesensing_sample_qusu.py > tests/remotesensing_sample02.log 2>&1 &
"""

PATH_PREFIX = f"/home/{getpass.getuser()}"

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
# CHECKPOINT = "/home/ccwan/stu_Jiangtp/model_repo/llava-7b-hf"
CHECKPOINT = f"{PATH_PREFIX}/workspace/models/llava-1.5-7b-hf"

#QMODEL_PATH = "/home/ccwan/stu_Jiangtp/spinquant-test/lab_models/llava-1.5-7b-spinquant-learned-w4a16"
# BEST_MODEL_DIR = "/home/ecnu01/sure_quant_models/20260808/best_quantized_model"
BEST_MODEL_DIR = "/home/ecnu01/sure_quant_models/w4a16_language_only/20260823/best_quantized_model"


#VAL_DIR = Path("/home/ccwan/stu_Jiangtp/spinquant-test/sample_img/remotesensing_sample")
VAL_DIR = Path("/home/ecnu01/workspace/remotesensing_sample")

SAMPLE_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".webp", ".png"}

# print(f"CHECKPOINT: {CHECKPOINT}")




def make_prompt(processor: Any, text: str) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": text}, {"type": "image"}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True)




def compute_kl_for_quantization(
    fp_logits: torch.Tensor,  # original output logits
    q_logits: torch.Tensor,   # quantized output logits
    bins: int = 256,          # 直方图分箱数(bin)
    eps: float = 1e-10,        # 防止 log(0)
) -> float:
    # 1. 展平
    fp_flat = fp_logits.to(torch.float32).detach().cpu().flatten()
    q_flat = q_logits.to(torch.float32).detach().cpu().flatten()

    # 2. 统一取值范围（必须用相同的 min/max 分箱，否则 KL 无意义）
    min_val = min(fp_flat.min(), q_flat.min())
    max_val = max(fp_flat.max(), q_flat.max())

    # 3. 把一个一维张量里的数字，分成若干区间，统计每个区间有多少个数，返回每个区间的数量
    fp_hist = torch.histc(fp_flat, bins=bins, min=min_val, max=max_val)
    q_hist = torch.histc(q_flat, bins=bins, min=min_val, max=max_val)

    # 4. 转化为频率分布，norm到[0,1] 防止后续KL计算发生nan
    p = fp_hist / (fp_hist.sum())
    q = q_hist / (q_hist.sum())

    # 5. 数值安全保护
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)

    # 6. 计算 KL(P || Q)：用量化分布 Q 近似真实分布 P
    kl = torch.sum(p * torch.log(p / q))

    return kl.item()


def compute_cos_similarity(fp_weight: torch.Tensor, q_weight: torch.Tensor):
    fp_weight = fp_weight.detach().cpu().flatten().float()
    q_weight = q_weight.detach().cpu().flatten().float()
    eval_len = min(len(fp_weight), len(q_weight))
    fp_weight = fp_weight[:eval_len]
    q_weight = q_weight[:eval_len]
    return F.cosine_similarity(fp_weight, q_weight, dim=0).item()


def compute_pearson_correlation(x: torch.Tensor, y: torch.Tensor):
    """
    计算两个张量的皮尔逊相关系数 PCC
    x, y: 任意形状的张量（会自动展平）
    返回: PCC 值，范围 [-1,1]
    """
    # 展平成一维
    x = x.detach().cpu().flatten().float()
    y = y.detach().cpu().flatten().float()

    eval_len = min(len(x), len(y))
    x = x[:eval_len]
    y = y[:eval_len]

    # 减去均值
    x_mean = x - x.mean()
    y_mean = y - y.mean()

    # 计算分子（协方差部分）
    numerator = (x_mean * y_mean).sum()
    
    # 计算分母（标准差乘积）
    denominator = torch.sqrt(torch.sum(x_mean ** 2)) * torch.sqrt(torch.sum(y_mean ** 2))
    
    # 防止除 0
    eps = 1e-8
    pcc = numerator / (denominator + eps)
    
    return pcc.item()


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

def loaded_best_model():
    """Load the saved best quantized model once per test module."""
    # if not BEST_MODEL_DIR.is_dir():
    #     pytest.skip(f"Best-model directory not found: {BEST_MODEL_DIR}")
    # if not torch.cuda.is_available():
    #     pytest.skip("CUDA device is required to load this saved checkpoint")

    from scripts.llava_quant_calib_wa import load_quantized_model

    model = load_quantized_model(
        BEST_MODEL_DIR, device_map="cuda", torch_dtype=torch.float16
    )

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(BEST_MODEL_DIR)
    model.eval()
    return model, processor

def run_saved_model_int4() -> None:
    """Load a saved quantized model and run inference."""
    save_path = BEST_MODEL_DIR
    print(f"\n========== Loading saved quantized model from {save_path} ==========")

    loaded_model, processor = loaded_best_model()

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





# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    out_list = run_saved_model_int4()
    compare_with_full_model(out_list)



if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f">>>>>>>>>>>>> Done, elapsed time: {elapsed:.2f} seconds")
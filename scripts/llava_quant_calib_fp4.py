"""LLaVA FP4 Quantization Calibration Script.


TODO Usage:
    python llava_quant_calib_fp4.py              # run saved model inference
    python llava_quant_calib_fp4.py --calib       # calibrate and save
    python llava_quant_calib_fp4.py --quick       # quick single-image test
"""

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

from config.default_config import SureQuantConfig
from model.sure_quantizer import SureQuantizer
from model.sure_quant_linear import SureQuantLinear

from scripts.llava_wa.modeling_fp4 import (
    quantize_llava_model_fp4, 
    save_quantized_model_fp4,
    load_quantized_model_fp4,
)


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
CHECKPOINT = f"{PATH_PREFIX}/workspace/models/llava-1.5-7b-hf"
SAMPLE_IMG_DIR = f"{PATH_PREFIX}/workspace/awq_learn/sample_img"
CALIB_DATA_PATH = f"{PATH_PREFIX}/workspace/data/flickr30k/data/test-00000-of-00009.parquet"
SAVE_ID = "1"
DEFAULT_SAVE_DIR = f"{PATH_PREFIX}/workspace/sure_quant/model_saved/llava_7b_sure_fp4_{SAVE_ID}"



# ---------------------------------------------------------------------------
# Configuration helper
# ---------------------------------------------------------------------------
def make_cfg(
    num_bits: int = 4,
    block_size: int = 128,
    calibration_steps: int = 100,
    calibration_batch_size: int = 128,
    calibration_lr: float = 0.005,
    device: str = "cuda",
    **kwargs: Any,
) -> SureQuantConfig:
    """Build a SureQuantConfig with FP4-friendly defaults."""
    return SureQuantConfig(
        num_bits=num_bits,
        block_size=block_size,
        calibration_steps=calibration_steps,
        calibration_batch_size=calibration_batch_size,
        calibration_lr=calibration_lr,
        device=device,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_calib_data(
    calibration_sample_num: int = 128,
    max_samples_per_layer: int = 512,
    prompt_text: str = "Please describe the animal in this image\n",
    image_column: str = "image",
    block_size: int = 128,
    device_map: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
    seed: int = 42,
    quantize_vision: bool = True,
    quantize_mm_proj: bool = True,
    quantize_language: bool = True,
) -> tuple:
    """Load calibration data using llava_wa modular components.

    Returns:
        (processor, calibration_data) tuple where calibration_data maps
        layer names to activation tensors.
    """
    processor = AutoProcessor.from_pretrained(CHECKPOINT)
    model = LlavaForConditionalGeneration.from_pretrained(
        CHECKPOINT, device_map=device_map, torch_dtype=torch_dtype,
    )

    dataset = load_dataset("parquet", data_files=CALIB_DATA_PATH, split="train")
    sample_count = min(calibration_sample_num, len(dataset))
    dataset = dataset.select(range(sample_count))
    print(f"Loaded {sample_count} calibration samples from {CALIB_DATA_PATH}")

    layer_names = selected_linear_names(
        model,
        block_size=block_size,
        quantize_vision=quantize_vision,
        quantize_mm_proj=quantize_mm_proj,
        quantize_language=quantize_language,
    )
    samples = ({"image": item[image_column]} for item in dataset)
    calibration_data = collect_calibration_data(
        model,
        processor,
        samples,
        prompt=make_prompt(processor, prompt_text),
        max_samples_per_layer=max_samples_per_layer,
        seed=seed,
        layer_names=layer_names,
    )

    del model, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Released full-precision calibration model memory")

    return processor, calibration_data


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


# ---------------------------------------------------------------------------
# Quick calibration (single image, for testing)
# ---------------------------------------------------------------------------
# def _quick_calibrate(
#     cfg: SureQuantConfig,
#     image_path: str,
#     quantize_vision: bool = True,
#     quantize_mm_proj: bool = True,
#     quantize_language: bool = True,
# ) -> tuple:
#     """Minimal single-image calibration for fast testing."""
#     processor = AutoProcessor.from_pretrained(CHECKPOINT)
#     model = LlavaForConditionalGeneration.from_pretrained(
#         CHECKPOINT, device_map="cuda", torch_dtype=torch.float16,
#     )

#     device = next(model.parameters()).device
#     prompt = make_prompt(processor, DEFAULT_INFERENCE_PROMPT)
#     raw_image = Image.open(image_path)
#     inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)

#     # Collect activations via hooks
#     calib_data: dict[str, torch.Tensor] = {}

#     def make_hook(name: str):
#         def hook(_module, inp, _out):
#             if inp and inp[0] is not None:
#                 act = inp[0].detach().cpu()
#                 if act.dim() > 2:
#                     act = act.view(-1, act.shape[-1])
#                 if name in calib_data:
#                     calib_data[name] = torch.cat((calib_data[name], act), dim=0)
#                 else:
#                     calib_data[name] = act
#         return hook

#     hooks = []
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Linear):
#             hooks.append(module.register_forward_hook(make_hook(name)))

#     with torch.no_grad():
#         model(**inputs)
#     for h in hooks:
#         h.remove()

#     # Subsample if needed
#     for name in calib_data:
#         total = calib_data[name].shape[0]
#         if total > 512:
#             idx = torch.randperm(total)[:512]
#             calib_data[name] = calib_data[name][idx]
#         print(f"Collected {calib_data[name].shape[0]} vectors for {name}")

#     # Quantize
#     quantized_model = quantize_llava_model(
#         model,
#         num_bits=cfg.num_bits,
#         block_size=cfg.block_size,
#         rotation_strategy="rotation",
#         quantize_vision=quantize_vision,
#         quantize_mm_proj=quantize_mm_proj,
#         quantize_language=quantize_language,
#         quantize_weight=True,
#     )

#     # Calibrate
#     logs = calibrate_all_quantizers(quantized_model, calib_data, cfg)

#     return quantized_model, processor, logs



def example_calib(args) -> None:
    """Full calibration pipeline: collect data, quantize, calibrate, save."""

    cfg = make_cfg(
        num_bits=4,
        block_size=128,
        calibration_steps=100,
        calibration_batch_size=128,
        calibration_lr=0.005,
    )
    print(f"Entry file: {os.path.abspath(__file__)}\nConfig: {cfg}")

    seed_everything(42)

    # 1. Load calibration data
    processor, calib_data = load_calib_data(
        calibration_sample_num=cfg.calibration_batch_size,
        max_samples_per_layer=512,
        block_size=cfg.block_size,
    )

    # 2. Load model and quantize
    model = LlavaForConditionalGeneration.from_pretrained(
        CHECKPOINT, device_map="cuda", torch_dtype=torch.float16,
    )
    # quantized_model = quantize_llava_model(
    #     model,
    #     num_bits=cfg.num_bits,
    #     block_size=cfg.block_size,
    #     rotation_strategy="rotation",
    #     quantize_vision=True,
    #     quantize_mm_proj=True,
    #     quantize_language=True,
    #     quantize_weight=True,
    # )

    quantized_model = quantize_llava_model_fp4(
        model,
        num_bits=cfg.num_bits,
        block_size=cfg.block_size,
        rotation_strategy="rotation",
        quantize_vision=True,
        quantize_mm_proj=True,
        quantize_language=True,
        quantize_weight=True,
    )
    quantized_model.to("cuda")

    # 3. Calibrate
    logs_dict = calibrate_all_quantizers(quantized_model, calib_data, cfg)

    # 4. Save with llava_wa persistence (INT4 compression)
    metadata = {
        "format_version": 1,
        "base_checkpoint": CHECKPOINT,
        # "selection_metric": "mean_layer_validation_reconstruction_kl",
        # "best_trial": best_trial,
        # "best_score": best_score,
        # "assistant_outputs_file": str(Path("..") / inference_path.name),
        "surequant": _jsonable_config(cfg),
        "model_quantization": {
            "rotation_strategy": args.rotation_strategy,
            "quantize_vision": args.quantize_vision,
            "quantize_mm_proj": args.quantize_mm_proj,
            "quantize_language": args.quantize_language,
            "quantize_weight": args.quantize_weight,
        },
    }
    save_quantized_model_fp4(
        quantized_model, processor, DEFAULT_SAVE_DIR, metadata,
        max_shard_size="5GB",
    )
    
    # metadata = {
    #     "format_version": 1,
    #     "base_checkpoint": CHECKPOINT,
    #     "selection_metric": "mean_layer_validation_reconstruction_kl",
    #     "surequant": {
    #         "num_bits": cfg.num_bits,
    #         "block_size": cfg.block_size,
    #         "calibration_steps": cfg.calibration_steps,
    #         "calibration_lr": cfg.calibration_lr,
    #         "rotation_strategy": "rotation",
    #         "lambda_rec": cfg.lambda_rec,
    #         "lambda_dk": cfg.lambda_dk,
    #         "lambda_bal": cfg.lambda_bal,
    #         "lambda_range": cfg.lambda_range,
    #     },
    #     "model_quantization": {
    #         "rotation_strategy": "rotation",
    #         "quantize_vision": True,
    #         "quantize_mm_proj": True,
    #         "quantize_language": True,
    #         "quantize_weight": True,
    #     },
    # }
    # save_quantized_model(
    #     quantized_model, processor, save_path, metadata,
    #     max_shard_size="5GB",
    # )

    # 5. Inference
    infer(quantized_model, processor, os.path.join(SAMPLE_IMG_DIR, "sample1.jpg"))
    infer(quantized_model, processor, os.path.join(SAMPLE_IMG_DIR, "sample2.jpg"))


def run_saved_model() -> None:
    """Load a saved quantized model and run inference."""
    save_path = DEFAULT_SAVE_DIR
    print(f"\n========== Loading saved quantized model from {save_path} ==========")

    loaded_model = load_quantized_model_fp4(
        save_path, device_map="cuda", torch_dtype=torch.float16,
    )

    processor = AutoProcessor.from_pretrained(CHECKPOINT)

    infer(loaded_model, processor, os.path.join(SAMPLE_IMG_DIR, "sample1.jpg"))
    infer(loaded_model, processor, os.path.join(SAMPLE_IMG_DIR, "sample2.jpg"))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()
    print(args)

    # example_calib(args)

    run_saved_model()


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f">>>>>>>>>>>>> Done, elapsed time: {elapsed:.2f} seconds")
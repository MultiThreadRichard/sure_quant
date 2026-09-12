#!/usr/bin/env python3
"""W4A16KV4 quantized inference using SureQuant (weight) + TurboQuant (KV cache).

Pipeline:
  1. Load W4A16 quantized model from saved checkpoint (SureQuant INT4 weights)
  2. Wrap with LLaVAInferEngine (TurboQuant KV cache quantizer)
  3. Run inference with quantized weights + quantized KV cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
print(f"PROJECT_ROOT: {_PROJECT_ROOT}")
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "llava_quant"))
sys.path.insert(0, str(_PROJECT_ROOT / "mme"))

# DEFAULT_MODEL_DIR = _PROJECT_ROOT / "runs" / "w4a16_language_only" / "best_quantized_model"
# DEFAULT_MODEL_DIR = Path("/home/ecnu01/sure_quant_models/w4a16_language_only/20260823/best_quantized_model")
DEFAULT_MODEL_DIR = Path("/home/ecnu01/workspace/sure_quant/model_saved/llava_7b_surequant_w4a16/best_quantized_model")




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
    from llava_wa.persistence import load_quantized_model

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
    from llava_kv_quant_turbo import LLaVAInferEngine

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

    sample_image_path = _PROJECT_ROOT / "sample_img/cat2.jpg"
    # Step 3: Run W4A16KV4 inference
    output = run_inference(
        model=model,
        processor=processor,
        image_path=sample_image_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        evaluate=args.evaluate,
    )

    print("\n[Done] W4A16KV4 inference completed")
    return output


if __name__ == "__main__":
    main()

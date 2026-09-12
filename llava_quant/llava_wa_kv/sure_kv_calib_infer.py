#!/usr/bin/env python3
"""SureQuant KV-cache: calibrate -> quantized inference -> verification.

End-to-end KV-only quantization of a LLaVA decoder's KV cache using the
SureQuant rotation + block-uniform pipeline.  Weights and activations stay
full-precision here — the weight/activation SureQuant path is intentionally out
of scope for this script.

Pipeline (three stages, run in sequence):

1. **Calibrate** — load the calibration dataset, collect per-layer native KV
   vectors, and train each layer's Givens rotation (reusing
   ``calibrate_kv_layer`` / ``train.calibrate_rotations.calibrate_rotation``).
2. **Infer** — run greedy decoding with a manual prefill/decode loop that
   quantizes the KV cache after every step (``LLaVAKVSureQuantizer``).
3. **Verify** — compare the stashed native cache against the quantized cache
   (MSE > 0 and not bitwise-equal) to confirm the cache was really quantized.

Usage:
    CUDA_VISIBLE_DEVICES=0 python llava_quant/llava_wa_kv/sure_kv_calib_infer.py
    # skip calibration (Hadamard-only identity Givens baseline)
    CUDA_VISIBLE_DEVICES=0 python llava_quant/llava_wa_kv/sure_kv_calib_infer.py --skip-calibrate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from config.default_config import SureQuantConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SureQuant KV-cache: calibrate -> quantized inference -> verify"
    )
    # Quantization
    parser.add_argument("--num-bits", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--rotation-strategy", choices=("rotation", "stiefel"),
                        default="rotation")
    parser.add_argument("--scale-granularity", choices=("per_block", "per_vector_block"),
                        default="per_vector_block")
    parser.add_argument("--clip-ratio", type=float, default=1.0)
    parser.add_argument("--quantize-k", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quantize-v", action=argparse.BooleanOptionalAction, default=True)
    # Calibration
    parser.add_argument("--skip-calibrate", action="store_true",
                        help="Skip Givens calibration (use Hadamard-only identity Givens)")
    parser.add_argument("--calibration-steps", type=int, default=100)
    parser.add_argument("--calibration-lr", type=float, default=0.005)
    parser.add_argument("--calibration-sample-num", type=int, default=128,
                        help="Number of calibration dataset samples (truncates the dataset)")
    parser.add_argument("--validation-fraction", type=float, default=0.2,
                        help="Fraction of the calibration dataset used as validation")
    # Inference
    parser.add_argument("--image", type=str, default=None,
                        help="Test image path (defaults to sample_img/cat2.jpg)")
    parser.add_argument("--prompt", type=str, default="Please describe this image.",
                        help="Text prompt for generation")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    # Misc
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    if requested != "cpu" and not torch.cuda.is_available():
        print("[warn] CUDA not available; falling back to CPU")
        return "cpu"
    return requested


def _build_config(args: argparse.Namespace) -> SureQuantConfig:
    return SureQuantConfig(
        num_bits=args.num_bits,
        block_size=args.block_size,
        clip_ratio=args.clip_ratio,
        activation_scale_granularity=args.scale_granularity,
        weight_scale_granularity=args.scale_granularity,
        calibration_steps=args.calibration_steps,
        calibration_lr=args.calibration_lr,
        # Training minibatch size for ``calibrate_rotation``. This is independent
        # of the dataset size, which is controlled by --calibration-sample-num.
        calibration_batch_size=128,
        device=args.device,
    )


def calibrate_kv_quantizers(
    model,
    processor,
    kv_quant,
    cfg: SureQuantConfig,
    args: argparse.Namespace,
    device: str,
) -> None:
    """Collect native KV vectors and train each layer's Givens rotation.

    Reuses ``collect_kv_cache`` (from ``test_kv_mse``) to gather per-layer
    ``[N, head_dim]`` K/V vectors and ``calibrate_kv_layer`` to run the exact
    ``train.calibrate_rotations.calibrate_rotation`` trainer on them.
    """
    from datasets import load_dataset

    from llava_quant.llava_wa.config import CALIBRATION_DATA_PATHS, DEFAULT_PROMPT
    from llava_quant.llava_wa.data import make_prompt
    from llava_quant.llava_wa.search import _release_cuda_memory
    from llava_quant.llava_wa_kv.sure_quant_kv_llava import calibrate_kv_layer
    from llava_quant.llava_wa_kv.test_kv_mse import collect_kv_cache

    # 1. Load the calibration dataset (parquet of images) and truncate it.
    data_path = CALIBRATION_DATA_PATHS[0]
    dataset = load_dataset("parquet", data_files=data_path, split="train")
    sample_count = min(args.calibration_sample_num, len(dataset))
    dataset = dataset.select(range(sample_count))
    print(f"Loaded {sample_count} calibration samples from {data_path}")

    # 2. Split train / validation by validation_fraction (validation is unused
    #    here — kept for parity with the MSE test's calibration path).
    num_val = min(max(1, round(sample_count * args.validation_fraction)), sample_count - 1)
    order = torch.randperm(sample_count, generator=torch.Generator().manual_seed(args.seed))
    train_indices = order[num_val:].tolist()
    train_samples = [dataset[i] for i in train_indices]
    print(f"Split dataset into {len(train_samples)} train / {num_val} validation samples")

    # 3. Collect native KV-cache vectors for the training split.
    prompt = make_prompt(processor, DEFAULT_PROMPT)
    print("[calibrate] collecting native KV cache ...")
    train_kv = collect_kv_cache(model, processor, train_samples, prompt, device)
    _release_cuda_memory()

    # 4. Calibrate per-layer Givens rotations on the training KV vectors.
    print("[calibrate] training per-layer Givens rotations ...")
    for i in range(kv_quant.num_layers):
        calibrate_kv_layer(
            kv_quant.layers[i],
            train_kv[i][0].to(device),
            train_kv[i][1].to(device),
            cfg,
        )
        _release_cuda_memory()
    del train_kv
    _release_cuda_memory()
    print("[calibrate] done")


@torch.no_grad()
def generate_with_quantized_kv(
    model,
    processor,
    kv_quant,
    raw_image,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> tuple[torch.Tensor, object]:
    """Greedy decode with the KV cache quantized after every prefill/decode step.

    Mirrors ``LLaVAInferEngine`` (TurboQuant) but drives the SureQuant
    ``LLaVAKVSureQuantizer`` instead.  Decode steps use
    ``quantize_decode_with_native_update`` so the full native cache is retained
    for the post-hoc verification stage.

    Returns:
        ``(generated_ids, final_past_kv)``.
    """
    from transformers import DynamicCache

    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]

    past_kv = DynamicCache()
    generated = input_ids
    eos_token_id = processor.tokenizer.eos_token_id

    for step in range(max_new_tokens):
        if step == 0:
            # Prefill: process the whole prompt+image, then quantize the full cache.
            outputs = model(
                input_ids=generated,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = kv_quant.quantize_prefill(outputs.past_key_values)
        else:
            # Decode: only the latest token is new; quantize just that increment.
            seq_len_before = generated.shape[1] - 1
            outputs = model(
                input_ids=generated[:, -1:],
                attention_mask=attention_mask,
                pixel_values=None,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = kv_quant.quantize_decode_with_native_update(
                outputs.past_key_values, seq_len_before=seq_len_before,
            )

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

        if next_token.item() == eos_token_id:
            break

    return generated, past_kv


def verify_kv_quantized(kv_quant, past_kv) -> tuple[bool, dict]:
    """Confirm the KV cache was actually quantized during inference.

    Quantization is lossy, so a correctly quantized cache must differ from the
    native cache: MSE > 0 and not bitwise-equal on at least one layer.
    """
    from llava_quant.llava_wa_kv.sure_quant_kv_llava import _get_cache_layer

    metrics = kv_quant.evaluate_metrics(past_kv)
    mean_k = metrics["mean_k_mse"]
    mean_v = metrics["mean_v_mse"]

    bitwise_equal = True
    for i, (k_native, v_native) in enumerate(kv_quant.native_past_kv):
        k_hat, v_hat = _get_cache_layer(past_kv, i)
        if not torch.equal(k_native, k_hat) or not torch.equal(v_native, v_hat):
            bitwise_equal = False
            break

    confirmed = (mean_k > 0.0 or mean_v > 0.0) and not bitwise_equal

    print("=" * 70)
    print("KV quantization verification")
    print("=" * 70)
    print(f"  mean K MSE (native vs quantized): {mean_k:.8f}")
    print(f"  mean V MSE (native vs quantized): {mean_v:.8f}")
    print(f"  cache bitwise-identical to native : {bitwise_equal}")
    print(f"  => KV cache was quantized: {confirmed}")
    return confirmed, metrics


def main() -> None:
    from PIL import Image
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    from llava_quant.llava_wa.config import CHECKPOINT
    from llava_quant.llava_wa.data import make_prompt
    from llava_quant.llava_wa.search import _release_cuda_memory
    from llava_quant.llava_wa_kv.sure_quant_kv_llava import LLaVAKVSureQuantizer

    args = parse_args()
    device = _resolve_device(args.device)
    if device == "cpu":
        print("[warn] running on CPU will be very slow; consider --device cuda")

    print("=" * 70)
    print("SureQuant KV-cache: calibrate -> quantized inference -> verify")
    print("=" * 70)

    # --- Load full-precision model (weights/activations NOT quantized). ---
    print(f"[load] model from {CHECKPOINT}")
    model = LlavaForConditionalGeneration.from_pretrained(
        CHECKPOINT, device_map=device, torch_dtype=torch.float16,
    ).eval()
    processor = AutoProcessor.from_pretrained(CHECKPOINT)

    kv_quant = LLaVAKVSureQuantizer(
        model,
        num_bits=args.num_bits,
        block_size=args.block_size,
        rotation_strategy=args.rotation_strategy,
        scale_granularity=args.scale_granularity,
        clip_ratio=args.clip_ratio,
        quantize_k=args.quantize_k,
        quantize_v=args.quantize_v,
    ).to(device)
    print(f"[build] {len(kv_quant.layers)} KV quantizers (head_dim={kv_quant.head_dim})")
    _release_cuda_memory()

    # --- Stage 1: calibrate the KV Givens rotations. ---
    if args.skip_calibrate:
        print("[calibrate] skipped (using Hadamard-only identity Givens)")
    else:
        calibrate_kv_quantizers(model, processor, kv_quant, _build_config(args), args, device)

    # --- Stage 2: run quantized inference. ---
    image_path = args.image
    if image_path is None:
        default_image = _REPO_ROOT / "sample_img" / "cat2.jpg"
        image_path = str(default_image) if default_image.exists() else None
    if image_path is None:
        raise ValueError("No test image available; pass --image")

    raw_image = Image.open(image_path).convert("RGB")
    prompt = make_prompt(processor, args.prompt)
    print(f"[infer] generating with quantized KV cache (image={image_path}) ...")
    generated, past_kv = generate_with_quantized_kv(
        model, processor, kv_quant, raw_image, prompt, args.max_new_tokens, device,
    )

    out_text = processor.decode(generated[0], skip_special_tokens=True)
    print("-" * 70)
    print(f"[output] {out_text}")
    print("-" * 70)

    # --- Stage 3: confirm the KV cache was actually quantized. ---
    confirmed, _ = verify_kv_quantized(kv_quant, past_kv)
    if not confirmed:
        raise RuntimeError("KV cache was NOT quantized during inference — check the pipeline")


if __name__ == "__main__":
    main()

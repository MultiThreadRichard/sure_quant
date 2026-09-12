#!/usr/bin/env python3
"""Standalone KV-cache rotation-quantization MSE test.

Measures the reconstruction MSE of the decoder KV cache under the same
rotation + block-uniform quantization used for weights/activations
(:class:`model.sure_quant_kv.SureQuantKVCache`).

Two modes:

* ``synthetic`` (default) — no model / no data needed.  Builds a single
  :class:`SureQuantKVCache` and quantizes random Gaussian K/V tensors of shape
  ``[batch, heads, seq, head_dim]``.  Runs on CPU by default.
* ``model`` — loads LLaVA-1.5-7b, runs one prefill forward on an image, collects
  the decoder's native KV cache, and quantizes it with one ``SureQuantKVCache``
  per layer via :class:`LLaVAKVSureQuantizer`.

``--calibrate`` (model mode only) loads the calibration dataset from
``CALIBRATION_DATA_PATHS``, truncates it to ``--calibration-sample-num`` samples,
splits it into train/validation by ``--validation-fraction``, then trains the
per-layer Givens rotations on the training split (reusing
``train.calibrate_rotations.calibrate_rotation``) and reports MSE on the held-out
validation split — comparable against the Hadamard-only baseline (Givens
initialized to identity).

Usage:
    python llava_quant/llava_wa_kv/test_kv_mse.py
    python llava_quant/llava_wa_kv/test_kv_mse.py --mode model
    python llava_quant/llava_wa_kv/test_kv_mse.py --mode model --calibrate
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
from model.sure_quant_kv import SureQuantKVCache
from model.sure_quantizer import SureQuantizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KV-cache rotation-quantization MSE test")
    parser.add_argument(
        "--mode",
        choices=("synthetic", "model"),
        default="synthetic",
        help="synthetic: random tensors, no model; model: real LLaVA KV cache",
    )
    parser.add_argument("--num-bits", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--rotation-strategy", choices=("rotation", "stiefel"), default="rotation")
    parser.add_argument("--scale-granularity", choices=("per_block", "per_vector_block"),
                        default="per_vector_block")
    parser.add_argument("--clip-ratio", type=float, default=1.0)
    parser.add_argument("--quantize-k", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quantize-v", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    # synthetic geometry
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    # model mode
    parser.add_argument("--image", type=str, default=None)
    # calibration
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calibration-steps", type=int, default=100)
    parser.add_argument("--calibration-lr", type=float, default=0.005)
    parser.add_argument("--calibration-sample-num", type=int, default=128,
                        help="Number of calibration dataset samples (truncates the dataset)")
    parser.add_argument("--validation-fraction", type=float, default=0.2,
                        help="Fraction of the calibration dataset used as validation")
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


def collect_kv_cache(
    model,
    processor,
    samples,
    prompt: str,
    device: str,
) -> list[list[torch.Tensor]]:
    """Collect per-layer native KV-cache vectors by running forward over samples.

    Each sample is run through a single prefill forward and its decoder K/V cache
    is flattened to ``[N, head_dim]`` per layer (``N = num_heads * seq_len``),
    concatenated across samples on CPU.

    Returns:
        ``kv_per_layer[i] == [k, v]`` where ``k``/``v`` are ``[N, head_dim]``.
    """
    from llava_quant.llava_wa_kv.sure_quant_kv_llava import _iter_cache_layers

    kv_per_layer: list[list[torch.Tensor]] | None = None
    for sample in samples:
        inputs = processor(images=sample["image"], text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, output_hidden_states=False)
        layers = [
            (k.reshape(-1, k.shape[-1]).detach().cpu(),
             v.reshape(-1, v.shape[-1]).detach().cpu())
            for k, v in _iter_cache_layers(outputs.past_key_values)
        ]
        if kv_per_layer is None:
            kv_per_layer = [[k, v] for k, v in layers]
        else:
            for i, (k, v) in enumerate(layers):
                kv_per_layer[i][0] = torch.cat((kv_per_layer[i][0], k), dim=0)
                kv_per_layer[i][1] = torch.cat((kv_per_layer[i][1], v), dim=0)
        del inputs, outputs
    return kv_per_layer


@torch.inference_mode()
def _mse_2d(x: torch.Tensor, quantizer: SureQuantizer | None) -> float:
    """MSE for a ``[N, head_dim]`` tensor through a :class:`SureQuantizer`."""
    if quantizer is None:
        return 0.0
    x_hat = quantizer(x)["x_hat"]
    return float((x.float() - x_hat.float()).square().mean())


def _print_mse_row(label: str, k_mse: float, v_mse: float) -> None:
    print(f"  {label:<24} K MSE: {k_mse:>14.8f}   V MSE: {v_mse:>14.8f}")


def _report_layer_mse(layer: SureQuantKVCache, k_flat: torch.Tensor, v_flat: torch.Tensor) -> tuple[float, float]:
    k_mse = _mse_2d(k_flat, layer.k_quantizer)
    v_mse = _mse_2d(v_flat, layer.v_quantizer)
    return k_mse, v_mse


# ---------------------------------------------------------------------------
# Synthetic mode on cpu
# ---------------------------------------------------------------------------
def run_synthetic(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)
    print("=" * 70)
    print("KV-cache rotation quantization MSE — synthetic mode")
    print("=" * 70)
    print(
        f"  head_dim={args.head_dim}, num_bits={args.num_bits}, "
        f"block_size={args.block_size}, scale={args.scale_granularity}, "
        f"clip_ratio={args.clip_ratio}"
    )
    print(f"  geometry: batch={args.batch}, heads={args.num_heads}, "
          f"seq={args.seq_len}, head_dim={args.head_dim}")

    layer = SureQuantKVCache(
        head_dim=args.head_dim,
        num_bits=args.num_bits,
        block_size=args.block_size,
        rotation_strategy=args.rotation_strategy,
        scale_granularity=args.scale_granularity,
        clip_ratio=args.clip_ratio,
        quantize_k=args.quantize_k,
        quantize_v=args.quantize_v,
    ).to(device)

    generator = torch.Generator().manual_seed(args.seed)
    num_vectors = args.batch * args.num_heads * args.seq_len
    k_flat = torch.randn(num_vectors, args.head_dim, generator=generator,
                         dtype=torch.float32, device=device)
    v_flat = torch.randn(num_vectors, args.head_dim, generator=generator,
                         dtype=torch.float32, device=device)

    k_mse, v_mse = _report_layer_mse(layer, k_flat, v_flat)
    print("\n  Reconstruction MSE (Hadamard-only baseline):")
    _print_mse_row("baseline", k_mse, v_mse)


# ---------------------------------------------------------------------------
# Model mode
# ---------------------------------------------------------------------------
def run_model(args: argparse.Namespace) -> None:
    from PIL import Image

    from llava_quant.llava_wa.config import CHECKPOINT, DEFAULT_PROMPT, CALIBRATION_DATA_PATHS
    from llava_quant.llava_wa.data import make_prompt
    from llava_quant.llava_wa_kv.sure_quant_kv_llava import (
        LLaVAKVSureQuantizer,
        calibrate_kv_layer,
    )
    from llava_quant.llava_wa.search import _release_cuda_memory
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    device = _resolve_device(args.device)
    if device == "cpu":
        print("[warn] model mode on CPU will be very slow; consider --device cuda")

    print("=" * 70)
    print("KV-cache rotation quantization MSE — model mode (LLaVA-1.5-7b)")
    print("=" * 70)

    print(f"[load] model from {CHECKPOINT}")
    model = LlavaForConditionalGeneration.from_pretrained(
        CHECKPOINT, device_map=device, torch_dtype=torch.float16
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
    # Weight loading and dtype conversion leave transient blocks in the CUDA
    # allocator cache; reclaim them before the KV forward passes.
    _release_cuda_memory()

    if args.calibrate:
        from datasets import load_dataset

        cfg = _build_config(args)

        # 1. Load the calibration dataset (parquet of images) and truncate it.
        data_path = CALIBRATION_DATA_PATHS[0]
        dataset = load_dataset("parquet", data_files=data_path, split="train")
        sample_count = min(args.calibration_sample_num, len(dataset))
        dataset = dataset.select(range(sample_count))
        print(f"Loaded {sample_count} calibration samples from {data_path}")

        # 2. Split the dataset into train / validation by validation_fraction.
        num_val = min(max(1, round(sample_count * args.validation_fraction)), sample_count - 1)
        order = torch.randperm(sample_count, generator=torch.Generator().manual_seed(args.seed))
        val_indices = order[:num_val].tolist()
        train_indices = order[num_val:].tolist()
        train_samples = [dataset[i] for i in train_indices]
        val_samples = [dataset[i] for i in val_indices]
        print(f"Split dataset into {len(train_samples)} train / {len(val_samples)} validation samples")

        # 3. Collect native KV-cache vectors for both splits.
        prompt = make_prompt(processor, DEFAULT_PROMPT)
        print("[collect] training KV cache ...")
        train_kv = collect_kv_cache(model, processor, train_samples, prompt, device)
        # The collection forwards leave cached activation blocks behind; free
        # them so training starts from a clean cache.
        _release_cuda_memory()

        # 4. Calibrate per-layer Givens rotations on the training KV vectors.
        print("\n[calibrate] training per-layer Givens rotations ...")
        for i in range(kv_quant.num_layers):
            calibrate_kv_layer(
                kv_quant.layers[i],
                train_kv[i][0].to(device),
                train_kv[i][1].to(device),
                cfg,
            )
            # Each call builds a fresh Adam optimizer and autograd graph; drop
            # them so the next layer does not inherit a fragmented cache.
            _release_cuda_memory()

        del train_kv
        _release_cuda_memory()

        print("[collect] validation KV cache ...")
        val_kv = collect_kv_cache(model, processor, val_samples, prompt, device)

        # 5. Report MSE on the held-out validation KV vectors.
        print("\n  Validation per-layer MSE after calibration:")
        total_k = total_v = 0.0
        for i in range(kv_quant.num_layers):
            k_mse, v_mse = _report_layer_mse(
                kv_quant.layers[i],
                val_kv[i][0].to(device),
                val_kv[i][1].to(device),
            )
            total_k += k_mse
            total_v += v_mse
            # Each layer moves its full [N, head_dim] validation tensors onto the
            # GPU and upcasts them to float32 for the MSE; reclaim that per-layer.
            _release_cuda_memory()
            if i % 4 == 0 or i == kv_quant.num_layers - 1:
                _print_mse_row(f"layer_{i}", k_mse, v_mse)
        _print_mse_row("MEAN", total_k / kv_quant.num_layers, total_v / kv_quant.num_layers)
        return

    # Baseline: image-backed prefill, quantize the full cache, and score MSE
    # against the stashed native cache.
    image_path = args.image
    if image_path is None:
        default_image = _REPO_ROOT / "sample_img" / "cat2.jpg"
        image_path = str(default_image) if default_image.exists() else None
    if image_path is None:
        raise ValueError("image path is None")

    prompt = make_prompt(processor, DEFAULT_PROMPT)
    raw_image = Image.open(image_path)
    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)
    
    print("[forward] collecting native KV cache ...")
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_hidden_states=False)
    past_kv = outputs.past_key_values

    # quantize_prefill may return a new cache object (legacy tuple format), so
    # use its return value when scoring.
    past_kv = kv_quant.quantize_prefill(past_kv)
    metrics = kv_quant.evaluate_metrics(past_kv)
    print("\n  Reconstruction MSE (Hadamard-only baseline):")
    for i, scores in metrics["layer_scores"].items():
        if int(i.split("_")[1]) % 4 == 0:
            _print_mse_row(i, scores["k_mse"], scores["v_mse"])
    _print_mse_row("MEAN", metrics["mean_k_mse"], metrics["mean_v_mse"])


def main() -> None:
    args = parse_args()
    if args.mode == "synthetic":
        run_synthetic(args)
    else:
        run_model(args)


if __name__ == "__main__":
    main()

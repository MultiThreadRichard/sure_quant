#!/usr/bin/env python3
"""Compare TurboQuant vs SureQuant KV-cache quantization on LLaVA (KV-only).

w16a16kv4

For each sample image under ``sample_img/`` with the fixed prompt
``Please describe this image.``, run KV-cache quantization inference through two
methods and compare
-------

* **TurboQuant** — ``mme.llava_kv_quant_turbo.LLaVAKVOptimizedQuantizer``
  (4-bit uniform KV cache, no calibration).  Its prefill/decode loop mirrors
  ``LLaVAInferEngine.generate`` (same ``DynamicCache`` + per-step quantization).
* **SureQuant** — ``llava_quant.llava_wa_kv.sure_quant_kv_llava.LLaVAKVSureQuantizer``
  (rotation + block-uniform, the native method from ``sure_kv_calib_infer.py``).
  Optionally Givens-calibrated via ``--calibrate``.

Weights and activations stay full-precision here (KV-only quantization).

Usage:
    CUDA_VISIBLE_DEVICES=0 python llava_quant/llava_wa_kv/sure_vs_turbo_kv_compare.py
    CUDA_VISIBLE_DEVICES=0 python llava_quant/llava_wa_kv/sure_vs_turbo_kv_compare.py --calibrate
    CUDA_VISIBLE_DEVICES=0 python llava_quant/llava_wa_kv/sure_vs_turbo_kv_compare.py --image sample_img/cat2.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tqdm import tqdm


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

# Imported at module level (not inside ``main``): ``generate_full_precision``
# below calls ``generate_step_logits_for_original``, and a function-local import
# would leave it unbound in the module namespace.  ``llava_quant.llava_wa`` is a
# plain package with no import-time side effects, so this is cheap.
from llava_quant.llava_wa.utils import (
    compute_cos_similarity,
    compute_pearson_correlation,
    compute_step_kl,
    generate_step_logits_for_original,
)

_SAMPLE_DIR = _REPO_ROOT / "sample_img"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TurboQuant vs SureQuant KV-cache quantization on LLaVA"
    )
    # SureQuant quantization knobs
    parser.add_argument("--num-bits", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--rotation-strategy", choices=("rotation", "stiefel"),
                        default="rotation")
    parser.add_argument("--scale-granularity", choices=("per_block", "per_vector_block"),
                        default="per_vector_block")
    parser.add_argument("--clip-ratio", type=float, default=1.0)
    parser.add_argument("--quantize-k", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quantize-v", action=argparse.BooleanOptionalAction, default=True)
    # SureQuant calibration (TurboQuant is fixed 4-bit, no calibration)
    parser.add_argument("--calibrate", action="store_true",
                        help="Givens-calibrate the SureQuant KV quantizer before inference")
    parser.add_argument("--calibration-steps", type=int, default=100)
    parser.add_argument("--calibration-lr", type=float, default=0.005)
    parser.add_argument("--calibration-sample-num", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    # Inference
    parser.add_argument("--image", type=str, default=None,
                        help="Restrict to a single image path (default: all of sample_img/)")
    parser.add_argument("--prompt", type=str, default="Please describe this image.")
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


def _sample_images(args: argparse.Namespace) -> list[str]:
    if args.image:
        return [args.image]
    return sorted(
        str(p) for p in _SAMPLE_DIR.glob("*") if p.suffix.lower() in _IMAGE_EXTS
    )


def _decode_with_native_update(quantizer, past_kv, seq_len_before: int):
    """Call the method-specific decode quantizer (keeps the native cache in sync)."""
    if hasattr(quantizer, "quantize_decode_with_native_update"):  # SureQuant
        return quantizer.quantize_decode_with_native_update(past_kv, seq_len_before)
    return quantizer.quantize_decode_with_native_kv_update(past_kv, seq_len_before)  # TurboQuant


@torch.no_grad()
def generate_quantized_kv(
    quantizer,
    model,
    processor,
    raw_image,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> tuple[torch.Tensor, object, list[torch.Tensor]]:
    """Greedy decode with the KV cache quantized after every step.

    Mirrors ``LLaVAInferEngine.generate`` (DynamicCache + prefill/decode) but is
    quantizer-agnostic: it works with both ``LLaVAKVOptimizedQuantizer`` and
    ``LLaVAKVSureQuantizer``.  Decode steps use the native-updating variant so
    each quantizer retains a full native cache for the MSE metric.

    This loop stays hand-written (rather than calling ``generate``) because the
    per-step quantizer call between the forward and the next decode step has no
    hook in the HuggingFace generation loop.

    Returns:
        ``(generated_ids, final_past_kv, step_logits)`` where ``step_logits``
        holds one float32 ``[1, vocab]`` CPU tensor per decode step, in the same
        format ``utils.decode_greedy`` produces so ``compute_step_kl`` can take
        it directly.
    """
    from transformers import DynamicCache

    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]

    past_kv = DynamicCache()
    generated = input_ids
    step_logits: list[torch.Tensor] = []
    eos_token_id = processor.tokenizer.eos_token_id

    for step in range(max_new_tokens):
        if step == 0:
            outputs = model(
                input_ids=generated,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = quantizer.quantize_prefill(outputs.past_key_values)
        else:
            seq_len_before = generated.shape[1] - 1
            outputs = model(
                input_ids=generated[:, -1:],
                attention_mask=attention_mask,
                pixel_values=None,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = _decode_with_native_update(quantizer, outputs.past_key_values, seq_len_before)

        next_logits = outputs.logits[:, -1, :]
        step_logits.append(next_logits.float().cpu())
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

        del outputs

        if next_token.item() == eos_token_id:
            break

    return generated, past_kv, step_logits


def compute_kv_mse(method: str, quantizer, past_kv) -> tuple[float, float]:
    """Mean K/V MSE between the stashed native cache and the quantized cache.

    Both quantizers stash a native copy during prefill/decode; only the storage
    format differs (TurboQuant: numpy dict ``[L,H,S,D]``, SureQuant: list of
    ``(k, v)`` tensors).
    """
    if method == "turbo":
        native_k = quantizer.native_past_kv["k_cache"]
        native_v = quantizer.native_past_kv["v_cache"]
        num_layers = native_k.shape[0]
        k_hat = np.stack(
            [past_kv.key_cache[i].squeeze(0).float().cpu().numpy() for i in range(num_layers)]
        )
        v_hat = np.stack(
            [past_kv.value_cache[i].squeeze(0).float().cpu().numpy() for i in range(num_layers)]
        )
        k_mse = float(np.mean((native_k.astype(np.float32) - k_hat) ** 2))
        v_mse = float(np.mean((native_v.astype(np.float32) - v_hat) ** 2))
    else:
        from llava_quant.llava_wa_kv.sure_quant_kv_llava import _get_cache_layer

        k_mses, v_mses = [], []
        for i, (k_native, v_native) in enumerate(quantizer.native_past_kv):
            k_hat, v_hat = _get_cache_layer(past_kv, i)
            k_mses.append(float((k_native.float() - k_hat.float()).square().mean()))
            v_mses.append(float((v_native.float() - v_hat.float()).square().mean()))
        k_mse = float(np.mean(k_mses))
        v_mse = float(np.mean(v_mses))
    return k_mse, v_mse


def run_single(
    quantizer,
    method: str,
    model,
    processor,
    raw_image,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> dict:
    generated, past_kv, step_logits = generate_quantized_kv(
        quantizer, model, processor, raw_image, prompt, max_new_tokens, device,
    )
    text = processor.decode(generated[0], skip_special_tokens=True)
    k_mse, v_mse = compute_kv_mse(method, quantizer, past_kv)
    return {
        "generated": generated,
        "text": text,
        "k_mse": k_mse,
        "v_mse": v_mse,
        "step_logits": step_logits,
    }


@torch.no_grad()
def generate_full_precision(
    model,
    processor,
    raw_image,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> dict:
    """Full-precision (no KV quantization) greedy baseline via ``model.generate``.

    Uses ``utils.generate_step_logits_for_original`` so the baseline exposes the
    same per-step raw logits the quantized runs collect, which is what lets
    ``compute_step_kl`` compare the two sides step by step.
    """
    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)
    return generate_step_logits_for_original(
        model, inputs, max_new_tokens,
        eos_token_id=processor.tokenizer.eos_token_id,
    )


def main() -> None:
    from PIL import Image
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    from llava_quant.llava_wa.config import CHECKPOINT
    from llava_quant.llava_wa.data import make_prompt
    from llava_quant.llava_wa.search import _release_cuda_memory
    from llava_quant.llava_wa_kv.sure_quant_kv_llava import LLaVAKVSureQuantizer
    from mme.llava_kv_quant_turbo import LLaVAKVOptimizedQuantizer

    args = parse_args()
    device = _resolve_device(args.device)
    if device == "cpu":
        print("[warn] running on CPU will be very slow; consider --device cuda")

    print("=" * 72)
    print("KV-cache quantization comparison: TurboQuant vs SureQuant")
    print("=" * 72)

    # --- Load full-precision model (weights/activations NOT quantized). ---
    print(f"[load] model from {CHECKPOINT}")
    model = LlavaForConditionalGeneration.from_pretrained(
        CHECKPOINT, device_map=device, torch_dtype=torch.float16,
    ).eval()
    processor = AutoProcessor.from_pretrained(CHECKPOINT)

    turbo_kv = LLaVAKVOptimizedQuantizer(model)
    sure_kv = LLaVAKVSureQuantizer(
        model,
        num_bits=args.num_bits,
        block_size=args.block_size,
        rotation_strategy=args.rotation_strategy,
        scale_granularity=args.scale_granularity,
        clip_ratio=args.clip_ratio,
        quantize_k=args.quantize_k,
        quantize_v=args.quantize_v,
    ).to(device)
    print(f"[build] turboquant 4-bit KV | surequant {args.num_bits}-bit KV "
          f"({len(sure_kv.layers)} layers)")
    _release_cuda_memory()

    # --- Optionally Givens-calibrate the SureQuant quantizer. ---
    if args.calibrate:
        from llava_quant.llava_wa_kv.sure_kv_calib_infer import (
            _build_config,
            calibrate_kv_quantizers,
        )
        print("[calibrate] calibrating SureQuant KV Givens rotations ...")
        calibrate_kv_quantizers(
            model, processor, sure_kv, _build_config(args), args, device,
        )
    else:
        print("[calibrate] skipped")

    prompt = make_prompt(processor, args.prompt)
    image_paths = _sample_images(args)
    print(f"[infer] {len(image_paths)} image(s), prompt={args.prompt!r}, "
          f"max_new_tokens={args.max_new_tokens}")

    results = []
    for img_path in image_paths:
        raw_image = Image.open(img_path).convert("RGB")
        turbo = run_single(
            turbo_kv, "turbo", model, processor, raw_image, prompt,
            args.max_new_tokens, device,
        )
        _release_cuda_memory()
        sure = run_single(
            sure_kv, "sure", model, processor, raw_image, prompt,
            args.max_new_tokens, device,
        )
        _release_cuda_memory()
        fp = generate_full_precision(
            model, processor, raw_image, prompt, args.max_new_tokens, device,
        )
        fp_text = processor.decode(fp["generated"][0], skip_special_tokens=True)
        _release_cuda_memory()

        results.append({
            "image": img_path,
            "turbo": turbo,
            "sure": sure,
            "fp": fp,
            "fp_text": fp_text,
        })

        print("\n" + "=" * 72)
        print(f"Image: {img_path}")
        print("-" * 72)
        # print(f"  {'method':<14} {'k_mse':>14} {'v_mse':>14}")
        # print(f"  {'turboquant':<14} {turbo['k_mse']:>14.8f} {turbo['v_mse']:>14.8f}")
        # print(f"  {'surequant':<14} {sure['k_mse']:>14.8f} {sure['v_mse']:>14.8f}")
        # print("-" * 72)
        print(f"  [fullprec ] {fp_text}")
        print(f"  [turboquant] {turbo['text']}")
        print(f"  [surequant ] {sure['text']}")

    # --- Summary ---
    print("\n" + "=" * 72)
    print("Summary (mean k_mse / v_mse across images)")
    print("=" * 72)
    print(f"  {'image':<34} {'turbo_k':>12} {'turbo_v':>12} {'sure_k':>12} {'sure_v':>12}")
    for r in results:
        t, s = r["turbo"], r["sure"]
        print(f"  {Path(r['image']).name:<34} {t['k_mse']:>12.6f} {t['v_mse']:>12.6f} "
              f"{s['k_mse']:>12.6f} {s['v_mse']:>12.6f}")
    mean_turbo_k = float(np.mean([r["turbo"]["k_mse"] for r in results]))
    mean_turbo_v = float(np.mean([r["turbo"]["v_mse"] for r in results]))
    mean_sure_k = float(np.mean([r["sure"]["k_mse"] for r in results]))
    mean_sure_v = float(np.mean([r["sure"]["v_mse"] for r in results]))
    print(f"  {'MEAN':<34} {mean_turbo_k:>12.6f} {mean_turbo_v:>12.6f} "
          f"{mean_sure_k:>12.6f} {mean_sure_v:>12.6f}")

    # --- Comparison vs full-precision baseline. ---
    # kl is the average over decode steps of KL(fp || q) between the two
    # next-token distributions.  Both sides stop early on EOS, so only the
    # overlapping steps count: min(len(fp step_logits), len(q step_logits)).
    # cos/pcc stay on the generated token ids, which is a different (coarser)
    # signal.
    print("\n" + "=" * 72)
    print("vs full-precision baseline")
    print("=" * 72)
    print(f"  {'image':<16} {'method':<10} {'kl':>12} "
          f"{'cos_sim':>10} {'pearson':>10}")
    turbo_metrics = {"kl": [], "cos": [], "pcc": []}
    sure_metrics = {"kl": [], "cos": [], "pcc": []}
    for r in results:
        fp_ids = r["fp"]["generated"][0]
        fp_step_logits = r["fp"]["step_logits"]
        for method, res, acc in (
            ("turbo", r["turbo"], turbo_metrics),
            ("sure", r["sure"], sure_metrics),
        ):
            q_ids = res["generated"][0]
            kl, _ = compute_step_kl(fp_step_logits, res["step_logits"])
            cos = compute_cos_similarity(fp_ids, q_ids)
            pcc = compute_pearson_correlation(fp_ids, q_ids)
            acc["kl"].append(kl)
            acc["cos"].append(cos)
            acc["pcc"].append(pcc)
            print(f"  {Path(r['image']).name:<16} {method:<10} "
                  f"{kl:>12.4e} {cos:>10.6f} {pcc:>10.6f}")
    print("-" * 72)
    for method, acc in (("turbo", turbo_metrics), ("sure", sure_metrics)):
        print(f"  {'MEAN':<16} {method:<10} "
              f"{np.mean(acc['kl']):>12.4e} "
              f"{np.mean(acc['cos']):>10.6f} "
              f"{np.mean(acc['pcc']):>10.6f}")


if __name__ == "__main__":
    main()

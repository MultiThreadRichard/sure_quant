#!/usr/bin/env python3
"""Compare TurboQuant vs SureQuant KV-cache quantization on MM-NIAH (retrieval-text).

Same protocol and same metrics as ``sure_vs_turbo_kv_compare.py``, but the
samples come from MM-NIAH instead of ``sample_img/``:

* data      — ``.../MM-NIAH/mm_niah_val/annotations/retrieval-text.jsonl``
              (one JSON dict per line; ``images_list`` entries are relative and
              get the MM-NIAH image root prepended)
* prompt    — ``context + question + answer instruction`` rendered through the
              LLaVA chat template; the ``<image>`` placeholders in ``context``
              line up one-to-one with ``images_list``.
* metrics   — K/V reconstruction MSE plus, against a full-precision baseline:
              the mean per-step ``KL(fp || q)`` between next-token logit
              distributions, and ``compute_cos_similarity`` /
              ``compute_pearson_correlation`` over the *generated* token slice.
              The NIAH answer hit rate (does the reference answer appear in the
              newly generated assistant text) is reported as well.
              The step-wise KL lives in ``llava_quant.llava_wa.utils``
              (``compute_step_kl``) so other KV/weight comparisons can reuse it,
              along with the shared greedy loop (``decode_greedy``) that produces
              the step logits for both sides.

The full-precision baseline is decoded by the very same prefill/decode loop as
the quantized runs (``decode_greedy`` with ``quantizer=None``), which is what
makes the per-step logits available.  Two consequences for the comparison with
``sure_vs_turbo_kv_compare.py``:

* ``compute_kl_for_quantization`` is NOT used here.  It histograms token ids —
  nominal values — after flattening the entire sequence, so a ~2000-token
  identical prompt swamps the handful of generated tokens and KL is pinned near
  zero no matter what the quantization does.  ``utils.compute_step_kl`` replaces
  it; ``compute_kl_for_quantization`` itself is left untouched.
* cos/pcc are evaluated on ``generated[prompt_len:]`` rather than the whole
  sequence, for the same dilution reason.

Methods (unchanged from the image-prompt script)
------------------------------------------------

* **TurboQuant** — ``mme.llava_kv_quant_turbo.LLaVAKVOptimizedQuantizer``
  (4-bit uniform KV cache, no calibration).
* **SureQuant** — ``llava_quant.llava_wa_kv.sure_quant_kv_llava.LLaVAKVSureQuantizer``
  (rotation + block-uniform).  Optionally Givens-calibrated via ``--calibrate``.

Weights and activations stay full-precision here (KV-only quantization).

Note on context length: MM-NIAH contexts run from ~600 to ~80k tokens while
LLaVA-1.5 only supports 4096 positions (576 tokens per image), so only samples
whose estimated prompt length falls within
``[--min-seq-len, --max-seq-len]`` are evaluated.  Use ``--max-samples`` to cap
how many fitting samples are evaluated.

Usage:
    CUDA_VISIBLE_DEVICES=0 python llava_quant/llava_wa_kv/sure_vs_turbo_mmniah_kv_compare.py
    CUDA_VISIBLE_DEVICES=0 python llava_quant/llava_wa_kv/sure_vs_turbo_mmniah_kv_compare.py --calibrate
    CUDA_VISIBLE_DEVICES=0 python llava_quant/llava_wa_kv/sure_vs_turbo_mmniah_kv_compare.py \
        --max-samples 8 --min-seq-len 2048 --max-seq-len 4096 --max-new-tokens 32
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from tqdm import tqdm
import time


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

# Imported at module level (not inside ``main``): ``run_single`` below uses
# ``decode_greedy`` too, and a function-local import would leave it unbound in
# the module namespace.  ``llava_quant.llava_wa`` is a plain package with no
# import-time side effects, so this is cheap.
from llava_quant.llava_wa.utils import (
    _release_native_cache,
    compute_cos_similarity,
    compute_pearson_correlation,
    compute_step_kl,
    decode_greedy,
)

_MMNIAH_ROOT = Path("/home/ecnu01/workspace/data/MM-NIAH/mm_niah_val")
_MMNIAH_ANNOTATIONS = _MMNIAH_ROOT / "annotations" / "retrieval-text.jsonl"
_MMNIAH_IMAGES_ROOT = _MMNIAH_ROOT / "mm_niah_dev" / "images"

# LLaVA-1.5 expands every ``<image>`` placeholder into 576 vision tokens.
_IMAGE_TOKENS = 576

_ANSWER_INSTRUCTION = "Answer the question using a single word or phrase."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TurboQuant vs SureQuant KV-cache quantization on MM-NIAH"
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
    # MM-NIAH data
    parser.add_argument("--annotations", type=str, default=str(_MMNIAH_ANNOTATIONS),
                        help="MM-NIAH retrieval-text.jsonl path")
    parser.add_argument("--images-root", type=str, default=str(_MMNIAH_IMAGES_ROOT),
                        help="Prefix prepended to every entry of images_list")
    parser.add_argument("--max-samples", type=int, default=32,
                        help="Evaluate at most this many fitting samples (-1 = all)")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Skip the first N samples of the jsonl")
    parser.add_argument("--min-seq-len", type=int, default=2048,
                        help="Skip samples whose tokenized prompt is shorter than this")
    parser.add_argument("--max-seq-len", type=int, default=4096,
                        help="Skip samples whose tokenized prompt exceeds this "
                             "(LLaVA-1.5 supports at most 4096)")
    # Inference
    parser.add_argument("--max-new-tokens", type=int, default=128)
    # Misc
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to dump per-sample results as JSON")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    if requested != "cpu" and not torch.cuda.is_available():
        print("[warn] CUDA not available; falling back to CPU")
        return "cpu"
    return requested


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a MM-NIAH jsonl file into a list of dicts (one per line)."""
    samples: list[dict] = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def resolve_image_paths(sample: dict, images_root: str | Path) -> list[str]:
    """Prepend the MM-NIAH image root to every relative ``images_list`` entry."""
    root = Path(images_root)
    return [str(root / rel) for rel in sample["images_list"]]


def build_prompt(processor, sample: dict) -> str:
    """Render ``context + question + answer instruction`` as a LLaVA chat prompt.

    ``context`` already carries one ``<image>`` placeholder per entry of
    ``images_list``, so no extra image token is added here.
    """
    text = f"{sample['context']}\n{sample['question']}\n{_ANSWER_INSTRUCTION}"
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def estimate_prompt_tokens(processor, sample: dict) -> int:
    """Prompt length without decoding the images, used to skip oversized samples.

    The tokenizer counts each ``<image>`` as a single token; the processor later
    expands it to ``_IMAGE_TOKENS``.  Exact for LLaVA-1.5, where every image has
    the same 336x336 size.
    """
    n_text = len(processor.tokenizer(build_prompt(processor, sample))["input_ids"])
    return n_text + len(sample["images_list"]) * (_IMAGE_TOKENS - 1)


def prepare_inputs(processor, sample: dict, images_root: str | Path, device: str):
    """Processor outputs for one MM-NIAH sample (all of ``images_list``)."""
    from PIL import Image

    paths = resolve_image_paths(sample, images_root)
    images = [Image.open(p).convert("RGB") for p in paths]
    prompt = build_prompt(processor, sample)
    return processor(images=images, text=prompt, return_tensors="pt").to(device)


def answer_hit(assistant_text: str, answer: str) -> bool:
    """Whether ``answer`` appears in the newly generated (assistant) text."""
    if not answer:
        return False
    return _normalize(answer) in _normalize(assistant_text)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def compute_kv_mse(method: str, quantizer, past_kv) -> tuple[float, float]:
    """Mean K/V MSE between the stashed native cache and the quantized cache.

    Same implementation as ``sure_vs_turbo_kv_compare.compute_kv_mse``.
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
    inputs,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> dict:
    """One quantized greedy run plus its KV reconstruction error."""
    out = decode_greedy(model, inputs, max_new_tokens, eos_token_id, quantizer=quantizer)
    k_mse, v_mse = compute_kv_mse(method, quantizer, out["past_kv"])
    return {
        "generated": out["generated"],
        "step_logits": out["step_logits"],
        "k_mse": k_mse,
        "v_mse": v_mse,
    }


def _select_samples(processor, args: argparse.Namespace) -> list[dict]:
    """Samples whose prompt length is within [--min-seq-len, --max-seq-len]."""
    samples = read_jsonl(args.annotations)
    print(f"[data] {len(samples)} samples in {args.annotations}")
    selected, n_empty, n_short, n_long, n_missing = [], 0, 0, 0, 0
    for idx, sample in enumerate(samples):
        if idx < args.start_index:
            continue
        if len(sample.get("images_list", [])) == 0:
            n_empty += 1
            continue
        est_len = estimate_prompt_tokens(processor, sample)
        if est_len < args.min_seq_len:
            n_short += 1
            continue
        if est_len > args.max_seq_len:
            n_long += 1
            continue
        paths = resolve_image_paths(sample, args.images_root)
        if any(not Path(p).exists() for p in paths):
            n_missing += 1
            continue
        selected.append(sample)
        if args.max_samples >= 0 and len(selected) >= args.max_samples:
            break
    print(f"[data] selected {len(selected)} samples "
          f"(skipped: {n_empty} w/o image, "
          f"{n_short} under {args.min_seq_len} tokens, "
          f"{n_long} over {args.max_seq_len} tokens, "
          f"{n_missing} with missing image files)")
    return selected


def main() -> None:
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    from llava_quant.llava_wa.config import CHECKPOINT
    from llava_quant.llava_wa.search import _release_cuda_memory, seed_everything
    from llava_quant.llava_wa_kv.sure_quant_kv_llava import LLaVAKVSureQuantizer
    from mme.llava_kv_quant_turbo import LLaVAKVOptimizedQuantizer

    args = parse_args()
    seed_everything(args.seed)
    device = _resolve_device(args.device)
    if device == "cpu":
        print("[warn] running on CPU will be very slow; consider --device cuda")

    print("=" * 72)
    print("MM-NIAH KV-cache quantization comparison: TurboQuant vs SureQuant")
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

    samples = _select_samples(processor, args)
    if not samples:
        raise SystemExit("no MM-NIAH sample fits --max-seq-len; raise it or lower --start-index")
    print(f"[infer] {len(samples)} sample(s), max_new_tokens={args.max_new_tokens}")

    eos_token_id = processor.tokenizer.eos_token_id
    results = []
    skipped_infer = 0
    for sample in tqdm(samples, desc="MM-NIAH"):
        inputs = prepare_inputs(processor, sample, args.images_root, device)
        prompt_len = int(inputs["input_ids"].shape[1])
        if prompt_len > args.max_seq_len:
            # Exact tokenization disagreed with the estimate: skip.
            skipped_infer += 1
            del inputs
            _release_cuda_memory()
            continue

        # Full-precision baseline first: same decode loop, no quantizer.
        fp = decode_greedy(model, inputs, args.max_new_tokens, eos_token_id)
        # The baseline cache is ~1 GiB per sample and is not used by any metric:
        # dropping it here is what keeps memory flat across samples.
        fp.pop("past_kv", None)
        _release_cuda_memory()
        turbo = run_single(turbo_kv, "turbo", model, inputs,
                           args.max_new_tokens, eos_token_id)
        _release_cuda_memory()
        sure = run_single(sure_kv, "sure", model, inputs,
                          args.max_new_tokens, eos_token_id)
        _release_native_cache(turbo_kv, sure_kv)
        _release_cuda_memory()

        for res in (turbo, sure, fp):
            res["text"] = _assistant_text(processor, res["generated"], prompt_len)

        # Metrics are taken here, while the per-step logits are still alive, so
        # that only scalars accumulate in ``results``.
        fp_ids = fp["generated"][0]
        metrics = {"fp_hit": answer_hit(fp["text"], sample["answer"])}
        for method, res in (("turbo", turbo), ("sure", sure)):
            q_ids = res["generated"][0]
            kl, n_steps = compute_step_kl(fp["step_logits"], res["step_logits"])
            metrics[method] = {
                "kl": kl,
                "n_steps": n_steps,
                "n_fp_steps": len(fp["step_logits"]),
                # Generated slice only: the shared prompt prefix would dominate.
                "cos": compute_cos_similarity(fp_ids[prompt_len:], q_ids[prompt_len:]),
                "pcc": compute_pearson_correlation(fp_ids[prompt_len:], q_ids[prompt_len:]),
                "hit": answer_hit(res["text"], sample["answer"]),
            }

        record = {
            "id": sample.get("id"),
            "image_paths": resolve_image_paths(sample, args.images_root),
            "num_images": len(sample["images_list"]),
            "seq_len": prompt_len,
            "question": sample["question"],
            "answer": sample["answer"],
            "k_mse": {"turbo": turbo["k_mse"], "sure": sure["k_mse"]},
            "v_mse": {"turbo": turbo["v_mse"], "sure": sure["v_mse"]},
            "n_gen_tokens": {
                "fullprec": len(fp["step_logits"]),
                "turbo": len(turbo["step_logits"]),
                "sure": len(sure["step_logits"]),
            },
            "text": {
                "fullprec": fp["text"],
                "turbo": turbo["text"],
                "sure": sure["text"],
            },
            "metrics": metrics,
        }
        results.append(record)

        print("\n" + "=" * 72)
        print(f"Sample id={record['id']} | seq_len={prompt_len} "
              f"| images={record['num_images']} | q={sample['question']!r}")
        print(f"  answer      = {sample['answer']!r}")
        print(f"  [fullprec ] {fp['text']!r}")
        print(f"  [turboquant] {turbo['text']!r}")
        print(f"  [surequant ] {sure['text']!r}")

        # Drop the heavy per-sample state before the next iteration.
        del fp, turbo, sure, inputs
        _release_cuda_memory()

    if skipped_infer:
        print(f"[warn] {skipped_infer} sample(s) skipped after exact tokenization")

    # --- Summary: KV reconstruction error. ---
    print("\n" + "=" * 72)
    print("Summary (mean k_mse / v_mse across samples)")
    print("=" * 72)
    print(f"  {'id':<8} {'seq_len':>8} {'turbo_k':>12} {'turbo_v':>12} {'sure_k':>12} {'sure_v':>12}")
    for r in results:
        print(f"  {str(r['id']):<8} {r['seq_len']:>8} "
              f"{r['k_mse']['turbo']:>12.6f} {r['v_mse']['turbo']:>12.6f} "
              f"{r['k_mse']['sure']:>12.6f} {r['v_mse']['sure']:>12.6f}")
    print(f"  {'MEAN':<8} {'':>8} "
          f"{np.mean([r['k_mse']['turbo'] for r in results]):>12.6f} "
          f"{np.mean([r['v_mse']['turbo'] for r in results]):>12.6f} "
          f"{np.mean([r['k_mse']['sure'] for r in results]):>12.6f} "
          f"{np.mean([r['v_mse']['sure'] for r in results]):>12.6f}")

    # --- Summary: comparison vs full-precision baseline. ---
    # KL is the mean per-step KL between next-token distributions; cos/pcc are
    # computed on the *generated* slice only (the identical prompt prefix would
    # otherwise dominate both).
    print("\n" + "=" * 72)
    print("vs full-precision baseline (step-wise logits KL; cos/pcc on generated tokens)")
    print("=" * 72)
    print(f"  {'id':<8} {'method':<10} {'kl':>10} {'cos_sim':>10} {'pearson':>10} "
          f"{'hit':>6} {'steps':>7}  (kl in nats/step)")
    # ``steps`` is deliberately per-sample only: averaging how many decode steps
    # got compared is a coverage diagnostic, not a quality number, and reads as
    # one in a MEAN row.
    acc = {"turbo": {"kl": [], "cos": [], "pcc": [], "hit": []},
           "sure": {"kl": [], "cos": [], "pcc": [], "hit": []}}
    fp_hits = []
    for r in results:
        fp_hits.append(float(r["metrics"]["fp_hit"]))
        for method in ("turbo", "sure"):
            m = r["metrics"][method]
            acc[method]["kl"].append(m["kl"])
            acc[method]["cos"].append(m["cos"])
            acc[method]["pcc"].append(m["pcc"])
            acc[method]["hit"].append(float(m["hit"]))
            print(f"  {str(r['id']):<8} {method:<10} "
                  f"{m['kl']:>10.4e} {m['cos']:>10.6f} {m['pcc']:>10.6f} "
                  f"{str(m['hit']):>6} {m['n_steps']:>3}/{m['n_fp_steps']:<3}")
    print("-" * 72)
    for method in ("turbo", "sure"):
        a = acc[method]
        print(f"  {'MEAN':<8} {method:<10} "
              f"{np.mean(a['kl']):>10.4e} {np.mean(a['cos']):>10.6f} "
              f"{np.mean(a['pcc']):>10.6f} {np.mean(a['hit']):>6.3f}")
    print(f"  {'MEAN':<8} {'fullprec':<10} {'':>10} {'':>10} {'':>10} "
          f"{np.mean(fp_hits):>6.3f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "annotations": args.annotations,
                "images_root": args.images_root,
                "num_bits": args.num_bits,
                "block_size": args.block_size,
                "rotation_strategy": args.rotation_strategy,
                "scale_granularity": args.scale_granularity,
                "clip_ratio": args.clip_ratio,
                "calibrate": args.calibrate,
                "min_seq_len": args.min_seq_len,
                "max_seq_len": args.max_seq_len,
                "max_new_tokens": args.max_new_tokens,
                "max_samples": args.max_samples,
            },
            "summary": {
                "n": len(results),
                "turbo": {k: float(np.mean(v)) for k, v in acc["turbo"].items()},
                "sure": {k: float(np.mean(v)) for k, v in acc["sure"].items()},
                "fullprec_hit": float(np.mean(fp_hits)) if fp_hits else 0.0,
                "turbo_k_mse": float(np.mean([r["k_mse"]["turbo"] for r in results])),
                "turbo_v_mse": float(np.mean([r["v_mse"]["turbo"] for r in results])),
                "sure_k_mse": float(np.mean([r["k_mse"]["sure"] for r in results])),
                "sure_v_mse": float(np.mean([r["v_mse"]["sure"] for r in results])),
            },
            "samples": results,
        }
        with open(out_path, "w", encoding="utf-8") as fout:
            json.dump(payload, fout, ensure_ascii=False, indent=2)
        print(f"\n[dump] wrote {out_path}")


def _assistant_text(processor, generated: torch.Tensor, prompt_len: int) -> str:
    """Decode only the tokens produced after the prompt (skip the context echo)."""
    return processor.decode(generated[0, prompt_len:], skip_special_tokens=True).strip()


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f">>>>>>>>>>>>> Done, elapsed time: {elapsed:.2f} seconds")
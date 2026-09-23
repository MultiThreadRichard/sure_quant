#!/usr/bin/env python3
"""Compare TurboQuant vs SureQuant KV-cache quantization on MMLongBench Summarization.

w16a16kv4
------------------------------------------------

* **TurboQuant** — ``mme.llava_kv_quant_turbo.LLaVAKVOptimizedQuantizer``
  (4-bit uniform KV cache, no calibration).
* **SureQuant** — ``llava_quant.llava_wa_kv.sure_quant_kv_llava.LLaVAKVSureQuantizer``
  (rotation + block-uniform).  Optionally Givens-calibrated via ``--calibrate``.

Weights and activations stay full-precision here (KV-only quantization).

Data
----
MMLongBench Summarization, taken from the upstream release layout::

    <mmlb_root>/mmlb_data_example/summ/gov_K128.jsonl     (241 samples, GAO reports)
    <mmlb_root>/mmlb_data_example/summ/lexsum_K128.jsonl  (146 samples, civil-rights suits)
    <mmlb_root>/mmlb_image/...                            (page scans)

Prompts and image paths follow the official loaders verbatim
(``load_gov_report`` / ``load_multi_lexsum`` in the MMLongBench repo):

* per-page line  ``Document {doc_id:.15} (page {page_id}): <image>``
* ``user_template`` is the official task instruction with ``{context}`` filled in
* ``system_template`` (``"Summary:"``) is prefilled as the assistant turn, so the
  model continues right after it — the same effect as the official
  ``continue_final_message=True``.

Length caveat: MMLongBench samples are ~128K tokens (~51 page images), while
LLaVA-1.5 supports 4096 positions (576 tokens per image).  Only the **first**
``--max-images`` pages are kept, and the matching ``Document ...`` lines are
dropped from the context so prompt text and images stay consistent.  With the
defaults (5 images, 384 new tokens) the prompt is ~3.1k tokens and leaves
comfortable headroom.

Scoring: the official metric for this task is ROUGE, so the ``hit`` flag of the
MM-NIAH script is replaced by ROUGE-1/2/L F-measures against the gold summary.
KL / cos / pcc / k_mse / v_mse are unchanged, and one column is added:

* ``div`` — first decode step whose greedy argmax differs from the baseline

Greedy decoding is a bootstrap, and a summary is long enough for the two
streams to actually split.  Once they do, every later step compares
distributions from *different* contexts, so the mean ``kl`` stops measuring
cache error and becomes an end-to-end divergence number that a single flipped
argmax can inflate by orders of magnitude.  ``div`` is printed next to ``kl``
so that a small ``div`` alongside a large ``kl`` is read as divergence rather
than as bad quantization.

Usage:
    CUDA_VISIBLE_DEVICES=0 python llava_quant/llava_wa_kv/sure_vs_turbo_mmlb_summ_kv_compare.py
    CUDA_VISIBLE_DEVICES=0 python llava_quant/llava_wa_kv/sure_vs_turbo_mmlb_summ_kv_compare.py --calibrate
    CUDA_VISIBLE_DEVICES=0 python llava_quant/llava_wa_kv/sure_vs_turbo_mmlb_summ_kv_compare.py \
        --task lexsum --max-samples 4 --max-images 4 --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import json
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

_MMLB_ROOT = Path("/home/ecnu01/workspace/data/MMLongBench")
_MMLB_IMAGES_ROOT = _MMLB_ROOT / "mmlb_image"

# LLaVA-1.5 expands every ``<image>`` placeholder into 576 vision tokens.
_IMAGE_TOKENS = 576

# --- Official MMLongBench Summarization prompt templates (see module docstring) ---
_ITEM_TEMPLATE = "Document {doc_id:.15} (page {page_id}): <image>"
_SYSTEM_TEMPLATE = "Summary:"

_TASKS = {
    "gov": {
        "annotations": _MMLB_ROOT / "mmlb_data_example" / "summ" / "gov_K128.jsonl",
        "user_template": (
            "You are given a government report from U.S. Government Accountability "
            "Office (GAO), and you are tasked to summarize the report. Write a concise "
            "summary (around 550 words) organized in multiple paragraphs. Where "
            "applicable, the summary should contain a short description of why GAO did "
            "this study, what GAO found, and what GAO recommends.\n\n"
            "Government Report:\n{context}\n\nNow please summarize the report."
        ),
    },
    "lexsum": {
        "annotations": _MMLB_ROOT / "mmlb_data_example" / "summ" / "lexsum_K128.jsonl",
        "user_template": (
            "You are given the legal documents in a civil rights lawsuit, and you are "
            "tasked to summarize the case. Write a concise summary of one paragraph "
            "(200 to 250 words). The summary should contain a short description of the "
            "background, the parties involved, and the outcomes of the case.\n\n"
            "Legal documents:\n{context}\n\nNow please summarize the case."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TurboQuant vs SureQuant KV-cache quantization "
                    "on MMLongBench Summarization"
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
    # MMLongBench data
    parser.add_argument("--task", choices=tuple(_TASKS), default="gov",
                        help="Which Summarization sub-task to run (gov = GAO reports, "
                             "lexsum = civil-rights lawsuits)")
    parser.add_argument("--annotations", type=str, default=None,
                        help="Override the jsonl path (default: the --task's file)")
    parser.add_argument("--images-root", type=str, default=str(_MMLB_IMAGES_ROOT),
                        help="Prefix prepended to every entry of image_list")
    parser.add_argument("--max-images", type=int, default=5,
                        help="Keep the first N pages of each sample (default 5).  "
                             "LLaVA-1.5 allows at most ~6 (4096 positions / 576 per image).")
    parser.add_argument("--max-samples", type=int, default=16,
                        help="Evaluate at most this many fitting samples (-1 = all)")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Skip the first N samples of the jsonl")
    parser.add_argument("--min-seq-len", type=int, default=2048,
                        help="Skip samples whose tokenized prompt is shorter than this")
    parser.add_argument("--max-seq-len", type=int, default=4096,
                        help="Skip samples whose tokenized prompt exceeds this "
                             "(LLaVA-1.5 supports at most 4096)")
    # Inference
    parser.add_argument("--max-new-tokens", type=int, default=384,
                        help="Official MMLongBench generation_max_length for summ is 384")
    # Misc
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    if requested != "cpu" and not torch.cuda.is_available():
        print("[warn] CUDA not available; falling back to CPU")
        return "cpu"
    return requested


def read_jsonl(path: str | Path) -> list[dict]:
    """Read an MMLongBench jsonl file into a list of dicts (one per line)."""
    samples: list[dict] = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def kept_image_list(sample: dict, max_images: int) -> list[str]:
    """The first ``max_images`` relative image paths of a sample.

    MMLongBench page images are ordered, so a prefix keeps the opening pages and
    the context lines stay aligned with the images one-for-one.
    """
    image_list = sample["image_list"]
    return image_list if max_images < 0 else image_list[:max_images]


def resolve_image_paths(sample: dict, images_root: str | Path, max_images: int) -> list[str]:
    """Prepend the MMLongBench image root to every kept ``image_list`` entry."""
    root = Path(images_root)
    return [str(root / rel) for rel in kept_image_list(sample, max_images)]


def build_context(sample: dict, max_images: int) -> str:
    """The official per-page ``Document {doc_id} (page {page_id}): <image>`` block.

    ``doc_id`` / ``page_id`` are recovered from the relative path exactly as
    ``load_gov_report`` does upstream, so the rendered text is identical for the
    pages that survive the ``--max-images`` cut.
    """
    lines = []
    for rel in kept_image_list(sample, max_images):
        doc_id = rel.split("/")[-2]
        page_id = rel.split("page")[1].split(".")[0]
        lines.append(_ITEM_TEMPLATE.format(doc_id=doc_id, page_id=page_id))
    return "\n\n".join(lines)


def build_prompt(processor, sample: dict, user_template: str, max_images: int) -> str:
    """Render one MMLongBench sample as a LLaVA chat prompt.

    ``{context}`` carries one ``<image>`` placeholder per kept page, so no extra
    image token is added here.  The official assistant prefill
    (``system_template`` = ``"Summary:"``) is appended after the template's
    ``ASSISTANT:`` marker, which is what makes generation continue mid-turn.
    """
    text = user_template.format(context=build_context(sample, max_images))
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    return f"{prompt} {_SYSTEM_TEMPLATE}"


def estimate_prompt_tokens(processor, sample: dict, user_template: str,
                           max_images: int) -> int:
    """Prompt length without decoding the images, used to skip oversized samples.

    The tokenizer counts each ``<image>`` as a single token; the processor later
    expands it to ``_IMAGE_TOKENS``.  Exact for LLaVA-1.5, where every page image
    has the same 336x336 size.
    """
    n_text = len(processor.tokenizer(
        build_prompt(processor, sample, user_template, max_images)
    )["input_ids"])
    return n_text + len(kept_image_list(sample, max_images)) * (_IMAGE_TOKENS - 1)


def prepare_inputs(processor, sample: dict, user_template: str, images_root: str | Path,
                   max_images: int, device: str):
    """Processor outputs for one MMLongBench sample (the kept page images)."""
    from PIL import Image

    paths = resolve_image_paths(sample, images_root, max_images)
    images = [Image.open(p).convert("RGB") for p in paths]
    prompt = build_prompt(processor, sample, user_template, max_images)
    return processor(images=images, text=prompt, return_tensors="pt").to(device)


def reference_summary(sample: dict) -> str:
    """Gold summary, flattened the way the official loader does.

    The gov reports store the summary as a list of ``{section_title, paragraphs}``
    sections; the lexsum files already carry a plain string.
    """
    summary = sample["answer"] if "answer" in sample else sample.get("summary", "")
    if isinstance(summary, str):
        return summary
    return "\n\n".join(
        aspect["section_title"] + ":\n" + "\n".join(aspect["paragraphs"])
        for aspect in summary
    )


def compute_rouge(hypothesis: str, reference: str) -> dict[str, float]:
    """ROUGE-1/2/L F-measures of a generated summary against the gold summary."""
    from rouge_score import rouge_scorer

    # Built per call: the scorer holds a stemmer and is not thread-safe, and
    # per-sample construction costs nothing next to a decode loop.
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {key: float(scores[key].fmeasure) for key in ("rouge1", "rouge2", "rougeL")}


def first_divergence(fp_ids: torch.Tensor, q_ids: torch.Tensor,
                     prompt_len: int) -> int | None:
    """Index of the first generated step whose argmax differs from the baseline.

    ``None`` when one stream is a prefix of the other.

    This matters more here than in the MM-NIAH test: summaries are long, and
    greedy decoding is a bootstrap — once the two streams split, every later
    step compares next-token distributions from *different* contexts, so the
    mean KL stops measuring quantization error and is dominated by the split.
    """
    n = min(len(fp_ids) - prompt_len, len(q_ids) - prompt_len)
    for i in range(n):
        if fp_ids[prompt_len + i].item() != q_ids[prompt_len + i].item():
            return i
    return None


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


def _select_samples(processor, args: argparse.Namespace,
                    user_template: str) -> list[dict]:
    """Samples whose prompt length is within [--min-seq-len, --max-seq-len]."""
    samples = read_jsonl(args.annotations)
    print(f"[data] {len(samples)} samples in {args.annotations}")
    selected, n_empty, n_short, n_long, n_missing = [], 0, 0, 0, 0
    for idx, sample in enumerate(samples):
        if idx < args.start_index:
            continue
        if len(sample.get("image_list", [])) == 0:
            n_empty += 1
            continue
        est_len = estimate_prompt_tokens(processor, sample, user_template, args.max_images)
        if est_len < args.min_seq_len:
            n_short += 1
            continue
        if est_len > args.max_seq_len:
            n_long += 1
            continue
        paths = resolve_image_paths(sample, args.images_root, args.max_images)
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


def _check_length_budget(args: argparse.Namespace) -> None:
    """Warn when the prompt budget cannot also fit ``--max-new-tokens``.

    Every kept page costs ``_IMAGE_TOKENS`` positions, so ``--max-images`` and
    ``--max-new-tokens`` trade off directly against LLaVA-1.5's 4096 window.
    """
    needed = args.max_images * _IMAGE_TOKENS + args.max_new_tokens
    if needed > args.max_seq_len:
        print(f"[warn] {args.max_images} image(s) x {_IMAGE_TOKENS} tokens + "
              f"{args.max_new_tokens} new tokens = {needed} > --max-seq-len "
              f"{args.max_seq_len}: lower --max-images or --max-new-tokens, "
              f"or the prompt will be truncated mid-summary")


def main() -> None:
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    from llava_quant.llava_wa.config import CHECKPOINT
    from llava_quant.llava_wa.search import _release_cuda_memory, seed_everything
    from llava_quant.llava_wa_kv.sure_quant_kv_llava import LLaVAKVSureQuantizer
    from mme.llava_kv_quant_turbo import LLaVAKVOptimizedQuantizer

    args = parse_args()
    task = _TASKS[args.task]
    if args.annotations is None:
        args.annotations = str(task["annotations"])
    user_template = task["user_template"]

    seed_everything(args.seed)
    device = _resolve_device(args.device)
    if device == "cpu":
        print("[warn] running on CPU will be very slow; consider --device cuda")

    print("=" * 72)
    print("MMLongBench Summarization KV-cache quantization comparison: "
          "TurboQuant vs SureQuant")
    print("=" * 72)
    _check_length_budget(args)

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

    samples = _select_samples(processor, args, user_template)
    if not samples:
        raise SystemExit("no MMLongBench sample fits --max-seq-len; reduce --max-images "
                         "or raise --max-seq-len")
    print(f"[infer] {len(samples)} sample(s), max_images={args.max_images}, "
          f"max_new_tokens={args.max_new_tokens}")

    eos_token_id = processor.tokenizer.eos_token_id
    results = []
    skipped_infer = 0
    for sample in tqdm(samples, desc="MMLB-summ"):
        inputs = prepare_inputs(
            processor, sample, user_template, args.images_root, args.max_images, device,
        )
        prompt_len = int(inputs["input_ids"].shape[1])
        if prompt_len > args.max_seq_len:
            # Exact tokenization disagreed with the estimate: skip.
            skipped_infer += 1
            del inputs
            _release_cuda_memory()
            continue

        reference = reference_summary(sample)

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
        metrics = {"fp_rouge": compute_rouge(fp["text"], reference)}
        for method, res in (("turbo", turbo), ("sure", sure)):
            q_ids = res["generated"][0]
            kl, n_steps = compute_step_kl(fp["step_logits"], res["step_logits"])
            # ``div`` locates the step where the two greedy streams split; from
            # there on ``kl`` compares different contexts rather than measuring
            # cache error, so it is reported alongside as a read-it-accordingly flag.
            div = first_divergence(fp_ids, q_ids, prompt_len)
            metrics[method] = {
                "kl": kl,
                "n_steps": n_steps,
                "first_div": div,
                "n_fp_steps": len(fp["step_logits"]),
                # Generated slice only: the shared prompt prefix would dominate.
                "cos": compute_cos_similarity(fp_ids[prompt_len:], q_ids[prompt_len:]),
                "pcc": compute_pearson_correlation(fp_ids[prompt_len:], q_ids[prompt_len:]),
                "rouge": compute_rouge(res["text"], reference),
            }

        record = {
            "id": sample.get("id"),
            "image_paths": resolve_image_paths(sample, args.images_root, args.max_images),
            "num_images_total": len(sample["image_list"]),
            "num_images_used": len(kept_image_list(sample, args.max_images)),
            "seq_len": prompt_len,
            "answer": reference,
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
              f"| images={record['num_images_used']}/{record['num_images_total']}")
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
    print(f"  {'id':<32} {'seq_len':>8} {'turbo_k':>12} {'turbo_v':>12} "
          f"{'sure_k':>12} {'sure_v':>12}")
    for r in results:
        print(f"  {str(r['id']):<32} {r['seq_len']:>8} "
              f"{r['k_mse']['turbo']:>12.6f} {r['v_mse']['turbo']:>12.6f} "
              f"{r['k_mse']['sure']:>12.6f} {r['v_mse']['sure']:>12.6f}")
    print(f"  {'MEAN':<32} {'':>8} "
          f"{np.mean([r['k_mse']['turbo'] for r in results]):>12.6f} "
          f"{np.mean([r['v_mse']['turbo'] for r in results]):>12.6f} "
          f"{np.mean([r['k_mse']['sure'] for r in results]):>12.6f} "
          f"{np.mean([r['v_mse']['sure'] for r in results]):>12.6f}")

    # --- Summary: comparison vs full-precision baseline. ---
    # KL is the mean per-step KL between next-token distributions; cos/pcc are
    # computed on the *generated* slice only (the identical prompt prefix would
    # otherwise dominate both).  ROUGE is the task metric, scored against the
    # gold summary rather than against the baseline's own text.
    #
    # ``div`` is reported next to ``kl`` because a summary is long enough for
    # greedy decoding to actually split: from step ``div`` on, ``kl`` compares
    # distributions from *different* contexts, so a single flipped argmax can
    # inflate the mean by orders of magnitude.  A small ``div`` next to a large
    # ``kl`` means the number reflects divergence, not cache error.
    print("\n" + "=" * 72)
    print("vs full-precision baseline")
    print("=" * 72)
    print(f"  {'id':<32} {'method':<8} {'kl':>10} {'cos_sim':>9} {'pearson':>9} "
          f"{'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'div':>5} "
          f"{'steps':>7}  (kl in nats/step; div = first differing step, - = none)")
    # ``steps`` is deliberately per-sample only: averaging how many decode steps
    # got compared is a coverage diagnostic, not a quality number, and reads as
    # one in a MEAN row.
    acc = {"turbo": {"kl": [], "cos": [], "pcc": [], "r1": [], "r2": [], "rL": []},
           "sure": {"kl": [], "cos": [], "pcc": [], "r1": [], "r2": [], "rL": []}}
    fp_rouge = {"r1": [], "r2": [], "rL": []}
    for r in results:
        for key, short in (("rouge1", "r1"), ("rouge2", "r2"), ("rougeL", "rL")):
            fp_rouge[short].append(r["metrics"]["fp_rouge"][key])
        for method in ("turbo", "sure"):
            m = r["metrics"][method]
            acc[method]["kl"].append(m["kl"])
            acc[method]["cos"].append(m["cos"])
            acc[method]["pcc"].append(m["pcc"])
            for key, short in (("rouge1", "r1"), ("rouge2", "r2"), ("rougeL", "rL")):
                acc[method][short].append(m["rouge"][key])
            div = "-" if m["first_div"] is None else str(m["first_div"])
            print(f"  {str(r['id']):<32} {method:<8} "
                  f"{m['kl']:>10.4e} {m['cos']:>9.6f} {m['pcc']:>9.6f} "
                  f"{m['rouge']['rouge1']:>8.4f} {m['rouge']['rouge2']:>8.4f} "
                  f"{m['rouge']['rougeL']:>8.4f} {div:>5} "
                  f"{m['n_steps']:>3}/{m['n_fp_steps']:<3}")
    print("-" * 72)
    for method in ("turbo", "sure"):
        a = acc[method]
        print(f"  {'MEAN':<32} {method:<8} "
              f"{np.mean(a['kl']):>10.4e} {np.mean(a['cos']):>9.6f} "
              f"{np.mean(a['pcc']):>9.6f} "
              f"{np.mean(a['r1']):>8.4f} {np.mean(a['r2']):>8.4f} "
              f"{np.mean(a['rL']):>8.4f} {'':>5}")
    print(f"  {'MEAN':<32} {'fullprec':<8} {'':>10} {'':>9} {'':>9} "
          f"{np.mean(fp_rouge['r1']):>8.4f} {np.mean(fp_rouge['r2']):>8.4f} "
          f"{np.mean(fp_rouge['rL']):>8.4f} {'':>5}")


def _assistant_text(processor, generated: torch.Tensor, prompt_len: int) -> str:
    """Decode only the tokens produced after the prompt (skip the context echo)."""
    return processor.decode(generated[0, prompt_len:], skip_special_tokens=True).strip()


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f">>>>>>>>>>>>> Done, elapsed time: {elapsed:.2f} seconds")

#!/usr/bin/env python3
"""Evaluate a SureQuant-quantized LLaVA on MMLongBench Summarization: fp16 vs w4aN.

Unlike ``llava_quant/llava_wa_kv/sure_vs_turbo_mmlb_summ_kv_compare.py`` -- which
keeps full-precision weights and compares two *KV-cache* quantizers -- this script
grades the actually-saved SureQuant checkpoint selected by ``--model-path``:
weights are INT4, and the **KV cache stays fp16** (no KV quantization is
involved).  Both the weight-only save
(``llava_sure_all_w4a16_sharegpt4v_00``, activations fp16) and the weight+activation
save (``llava_sure_all_w4a4_sharegpt4v_00``, activations INT4) load through the
same path -- the label shown in tables is derived from the save's config.

* **fp16** -- ``--baseline-checkpoint`` (default the LLaVA-1.5-7b base the save was
  quantized from), loaded with plain ``from_pretrained``.
* **w4a16 / w4a4** -- the SureQuant save, loaded through
  ``llava_quant.llava_wa.persistence.load_quantized_model`` (same path the MMBench
  eval uses).

Both sides run through the *same* ``decode_greedy`` loop with ``quantizer=None``,
so the only difference between the two streams is the weight quantization; that is
what makes the per-step KL / cos / pcc meaningful.

Data
----
MMLongBench Summarization, from the upstream release layout::

    <mmlb_root>/mmlb_data_example/summ/gov_K128.jsonl     (241 samples, GAO reports)
    <mmlb_root>/mmlb_data_example/summ/lexsum_K128.jsonl  (146 samples, civil-rights suits)
    <mmlb_root>/mmlb_image/...                            (page scans)

Prompts and image paths follow the official loaders verbatim (``load_gov_report``
/ ``load_multi_lexsum``): per-page line ``Document {doc_id:.15} (page {page_id}): <image>``,
the official ``user_template`` with ``{context}`` filled in, and ``system_template``
(``"Summary:"``) prefilled as the assistant turn so the model continues mid-turn --
the same effect as the official ``continue_final_message=True``.

Length caveat: MMLongBench samples are ~128K tokens (~51 page images) while
LLaVA-1.5 supports 4096 positions (576 tokens per image).  Only the **first**
``--max-images`` pages are kept, and the matching ``Document ...`` lines are
dropped from the context so prompt text and images stay consistent.

Scoring: ROUGE-1/2/L F-measures against the gold summary (the official task metric
for summarization), plus KL / cos / pcc / ``div`` measured against the fp16 stream.

Usage:
    CUDA_VISIBLE_DEVICES=0 python bench_eval/mmlb_summ_eval_surequant.py
    CUDA_VISIBLE_DEVICES=0 python bench_eval/mmlb_summ_eval_surequant.py --skip-baseline
    CUDA_VISIBLE_DEVICES=0 python bench_eval/mmlb_summ_eval_surequant.py \\
        --task lexsum --max-samples 4 --max-images 4 --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from tqdm import tqdm


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from llava_quant.llava_wa.config import parse_dtype
from llava_quant.llava_wa.data import _model_input_device
from llava_quant.llava_wa.utils import (
    compute_step_cos_pcc,
    compute_step_kl,
    decode_greedy,
)

# Same data layout as the reference KV script, but rooted at this machine's copy.
_MMLB_ROOT = Path("/data1/stu_Jiangtp/data/MMLongBench")
_MMLB_IMAGES_ROOT = _MMLB_ROOT / "mmlb_image"

# Full-precision base the SureQuant save was quantized from.  Kept as a literal
# (like ``mmbench_eval_surequant``) instead of importing ``llava_wa.config`` so
# this eval does not follow that file's calibration-time edits.
DEFAULT_FP16_CHECKPOINT = "/data1/stu_Jiangtp/models/llava-1.5-7b-hf"
DEFAULT_MODEL_PATH = (
    "/data1/stu_Jiangtp/sure_quant/model_saved/"
    "llava_sure_all_w4a16_sharegpt4v_00/best_quantized_model"
)

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

# Label the quantized stream gets in every table/JSON field.  Defaulted here and
# re-derived from the checkpoint's ``surequant_config.json`` in ``main`` (or
# overridden with ``--quant-label``), so a w4a4 save is not reported as w4a16.
_QUANT_METHOD = "w4a16"


def derive_quant_label(model_path: str | Path) -> str:
    """Bit-width label (``w4a4`` / ``w4a16``) read from a SureQuant save's config.

    Weights are always packed INT4 in these saves; the activation side is 4-bit
    when the checkpoint quantizes activations and left fp16 otherwise.
    """
    config_path = Path(model_path) / "surequant_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print(f"[warn] could not read {config_path}; labelling the run 'w4a16'")
        return "w4a16"
    num_bits = config.get("surequant", {}).get("num_bits", 4)
    quantize_activation = config.get("model_quantization", {}).get(
        "quantize_activation", False
    )
    return f"w{num_bits}a{num_bits if quantize_activation else 16}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a SureQuant-quantized LLaVA (w4a16) on MMLongBench "
                    "Summarization against the fp16 baseline; KV cache is full precision."
    )
    # Models
    parser.add_argument("--baseline-checkpoint", type=str, default=DEFAULT_FP16_CHECKPOINT,
                        help="fp16 LLaVA checkpoint used as the reference stream")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                        help="SureQuant save to grade (needs surequant_config.json)")
    parser.add_argument("--quant-label", type=str, default=None,
                        help="Label for the quantized stream in tables/JSON "
                             "(default: derived from surequant_config.json, e.g. w4a16/w4a4)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Run only the quantized model (drops KL/cos/pcc/div, "
                             "roughly halves runtime and memory)")
    parser.add_argument("--device-map", type=str, default="cuda")
    parser.add_argument("--torch-dtype", type=str, default="float16")
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
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to dump per-sample results as JSON")
    parser.add_argument("--seed", type=int, default=42)
    return parser


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
                   max_images: int, device):
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

    ``None`` when one stream is a prefix of the other.  Summaries are long enough
    for the two greedy streams to actually split; from step ``div`` on, ``kl``
    compares distributions from *different* contexts, so it is reported next to
    ``div`` to be read as divergence rather than as quantization error.
    """
    n = min(len(fp_ids) - prompt_len, len(q_ids) - prompt_len)
    for i in range(n):
        if fp_ids[prompt_len + i].item() != q_ids[prompt_len + i].item():
            return i
    return None


def run_model(model, inputs, max_new_tokens: int, eos_token_id: int | None) -> dict:
    """One greedy run through the shared decode loop (no KV quantizer)."""
    out = decode_greedy(model, inputs, max_new_tokens, eos_token_id)
    # The cache is per-sample state no metric needs; dropping it here keeps
    # memory flat across samples (~1 GiB at 2k tokens).
    out.pop("past_kv", None)
    return out


def _assistant_text(processor, generated: torch.Tensor, prompt_len: int) -> str:
    """Decode only the tokens produced after the prompt (skip the context echo)."""
    return processor.decode(generated[0, prompt_len:], skip_special_tokens=True).strip()


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


def load_models(args: argparse.Namespace):
    """Load the SureQuant save plus (unless skipped) the fp16 baseline."""
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    from llava_quant.llava_wa.persistence import load_quantized_model

    dtype = parse_dtype(args.torch_dtype)
    model_path = Path(args.model_path).expanduser()
    if not (model_path / "surequant_config.json").is_file():
        raise FileNotFoundError(
            f"Not a SureQuant checkpoint (no surequant_config.json): {model_path}"
        )

    print(f"[load] quantized model from {model_path}")
    quant_model = load_quantized_model(
        model_path, device_map=args.device_map, torch_dtype=dtype
    )

    fp16_model = None
    if not args.skip_baseline:
        baseline_path = Path(args.baseline_checkpoint).expanduser()
        print(f"[load] fp16 baseline from {baseline_path}")
        fp16_model = LlavaForConditionalGeneration.from_pretrained(
            baseline_path, device_map=args.device_map, torch_dtype=dtype,
        ).eval()

    # The two checkpoints share an identical chat template, so a single processor
    # renders byte-identical prompts for both streams.
    processor = AutoProcessor.from_pretrained(model_path)
    return fp16_model, quant_model, processor


def main() -> None:
    from llava_quant.llava_wa.search import _release_cuda_memory, seed_everything

    global _QUANT_METHOD

    args = build_parser().parse_args()
    task = _TASKS[args.task]
    if args.annotations is None:
        args.annotations = str(task["annotations"])
    user_template = task["user_template"]

    seed_everything(args.seed)
    _QUANT_METHOD = args.quant_label or derive_quant_label(args.model_path)

    print("=" * 72)
    print(f"MMLongBench Summarization SureQuant {_QUANT_METHOD} vs fp16 "
          f"-- KV cache full precision")
    print("=" * 72)
    _check_length_budget(args)

    fp16_model, quant_model, processor = load_models(args)
    _release_cuda_memory()

    samples = _select_samples(processor, args, user_template)
    if not samples:
        raise SystemExit("no MMLongBench sample fits --max-seq-len; reduce --max-images "
                         "or raise --max-seq-len")
    print(f"[infer] {len(samples)} sample(s), max_images={args.max_images}, "
          f"max_new_tokens={args.max_new_tokens}, "
          f"baseline={'off' if args.skip_baseline else 'on'}")

    eos_token_id = processor.tokenizer.eos_token_id
    input_device = _model_input_device(quant_model)
    results = []
    skipped_infer = 0
    for sample in tqdm(samples, desc="MMLB-summ"):
        inputs = prepare_inputs(
            processor, sample, user_template, args.images_root, args.max_images, input_device,
        )
        prompt_len = int(inputs["input_ids"].shape[1])
        if prompt_len > args.max_seq_len:
            # Exact tokenization disagreed with the estimate: skip.
            skipped_infer += 1
            del inputs
            _release_cuda_memory()
            continue

        reference = reference_summary(sample)

        # Baseline first; both streams share this one ``inputs`` dict, which
        # ``decode_greedy`` never mutates (it grows its own copy of the ids).
        fp = None
        if fp16_model is not None:
            fp = run_model(fp16_model, inputs, args.max_new_tokens, eos_token_id)
            _release_cuda_memory()
        quant = run_model(quant_model, inputs, args.max_new_tokens, eos_token_id)
        _release_cuda_memory()

        for res in (r for r in (fp, quant) if r is not None):
            res["text"] = _assistant_text(processor, res["generated"], prompt_len)

        # Metrics are taken here, while the per-step logits are still alive, so
        # that only scalars accumulate in ``results``.
        metrics: dict = {}
        quant_metrics: dict = {"rouge": compute_rouge(quant["text"], reference)}
        if fp is not None:
            fp_ids = fp["generated"][0]
            q_ids = quant["generated"][0]
            kl, n_steps = compute_step_kl(fp["step_logits"], quant["step_logits"])
            # cos/pcc ride the same per-step logits -- and the same n -- as ``kl``:
            # they measure the next-token distributions, not the sampled token ids.
            # Ids are arbitrary vocabulary indices, so correlating them said nothing
            # about similarity and, once the streams diverge, degenerated (a 2-token
            # sample forces PCC to exactly +/-1).
            cos, pcc, _ = compute_step_cos_pcc(fp["step_logits"], quant["step_logits"])
            quant_metrics.update({
                "kl": kl,
                "n_steps": n_steps,
                "n_fp_steps": len(fp["step_logits"]),
                "first_div": first_divergence(fp_ids, q_ids, prompt_len),
                "cos": cos,
                "pcc": pcc,
            })
            metrics["fp_rouge"] = compute_rouge(fp["text"], reference)
        metrics[_QUANT_METHOD] = quant_metrics

        record = {
            "id": sample.get("id"),
            "image_paths": resolve_image_paths(sample, args.images_root, args.max_images),
            "num_images_total": len(sample["image_list"]),
            "num_images_used": len(kept_image_list(sample, args.max_images)),
            "seq_len": prompt_len,
            "answer": reference,
            "n_gen_tokens": {
                "fullprec": None if fp is None else len(fp["step_logits"]),
                _QUANT_METHOD: len(quant["step_logits"]),
            },
            "text": {
                "fullprec": None if fp is None else fp["text"],
                _QUANT_METHOD: quant["text"],
            },
            "metrics": metrics,
        }
        results.append(record)

        print("\n" + "=" * 72)
        print(f"Sample id={record['id']} | seq_len={prompt_len} "
              f"| images={record['num_images_used']}/{record['num_images_total']}")
        if fp is not None:
            print(f"  [fullprec ] {fp['text']!r}")
        print(f"  [{_QUANT_METHOD:<9}] {quant['text']!r}")

        # Drop the heavy per-sample state before the next iteration.
        del fp, quant, inputs
        _release_cuda_memory()

    if skipped_infer:
        print(f"[warn] {skipped_infer} sample(s) skipped after exact tokenization")

    _print_summary(results, args)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "task": args.task,
                "annotations": args.annotations,
                "images_root": args.images_root,
                "model_path": args.model_path,
                "baseline_checkpoint": None if args.skip_baseline else args.baseline_checkpoint,
                "max_images": args.max_images,
                "min_seq_len": args.min_seq_len,
                "max_seq_len": args.max_seq_len,
                "max_new_tokens": args.max_new_tokens,
                "max_samples": args.max_samples,
            },
            "summary": _summary_dict(results, args),
            "samples": results,
        }
        with open(out_path, "w", encoding="utf-8") as fout:
            json.dump(payload, fout, ensure_ascii=False, indent=2)
        print(f"\n[dump] wrote {out_path}")


def _summary_dict(results: list[dict], args: argparse.Namespace) -> dict:
    """Aggregate scalars for the optional JSON dump."""
    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    summary: dict = {
        "n": len(results),
        _QUANT_METHOD: {
            key: mean([r["metrics"][_QUANT_METHOD]["rouge"][key] for r in results])
            for key in ("rouge1", "rouge2", "rougeL")
        },
    }
    if not args.skip_baseline:
        summary["fullprec_rouge"] = {
            key: mean([r["metrics"]["fp_rouge"][key] for r in results])
            for key in ("rouge1", "rouge2", "rougeL")
        }
        for key in ("kl", "cos", "pcc"):
            summary[f"{_QUANT_METHOD}_{key}"] = mean(
                [r["metrics"][_QUANT_METHOD][key] for r in results]
            )
    return summary


def _print_summary(results: list[dict], args: argparse.Namespace) -> None:
    """Per-sample ROUGE table plus the KL/cos/pcc comparison against fp16."""
    print("\n" + "=" * 72)
    if args.skip_baseline:
        print("ROUGE (quantized model only; no fp16 baseline was run)")
        print("=" * 72)
        print(f"  {'id':<32} {'seq_len':>8} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8}")
        for r in results:
            rouge = r["metrics"][_QUANT_METHOD]["rouge"]
            print(f"  {str(r['id']):<32} {r['seq_len']:>8} "
                  f"{rouge['rouge1']:>8.4f} {rouge['rouge2']:>8.4f} {rouge['rougeL']:>8.4f}")
        print("-" * 72)
        print(f"  {'MEAN':<32} {'':>8} "
              f"{np.mean([r['metrics'][_QUANT_METHOD]['rouge']['rouge1'] for r in results]):>8.4f} "
              f"{np.mean([r['metrics'][_QUANT_METHOD]['rouge']['rouge2'] for r in results]):>8.4f} "
              f"{np.mean([r['metrics'][_QUANT_METHOD]['rouge']['rougeL'] for r in results]):>8.4f}")
        return

    # All three distributional columns share one convention: they compare the
    # per-step next-token *distributions* over the min(n_fp, n_quant) steps both
    # greedy streams produced, then average over those steps.  ROUGE is the task
    # metric, scored against the gold summary rather than against the baseline's
    # own text.  ``div`` is reported next to ``kl`` because a summary is long
    # enough for greedy decoding to actually split, after which ``kl`` reflects
    # divergence.
    print("vs full-precision baseline")
    print("=" * 72)
    print(f"  {'id':<32} {'method':<8} {'kl':>10} {'cos_sim':>9} {'pearson':>9} "
          f"{'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'div':>5} "
          f"{'steps':>7}  (kl in nats/step; cos/pcc are per-step means over the "
          f"next-token logits; div = first differing step, - = none)")
    acc = {"kl": [], "cos": [], "pcc": [], "r1": [], "r2": [], "rL": []}
    fp_rouge = {"r1": [], "r2": [], "rL": []}
    for r in results:
        for key, short in (("rouge1", "r1"), ("rouge2", "r2"), ("rougeL", "rL")):
            fp_rouge[short].append(r["metrics"]["fp_rouge"][key])
        m = r["metrics"][_QUANT_METHOD]
        acc["kl"].append(m["kl"])
        acc["cos"].append(m["cos"])
        acc["pcc"].append(m["pcc"])
        for key, short in (("rouge1", "r1"), ("rouge2", "r2"), ("rougeL", "rL")):
            acc[short].append(m["rouge"][key])
        div = "-" if m["first_div"] is None else str(m["first_div"])
        print(f"  {str(r['id']):<32} {_QUANT_METHOD:<8} "
              f"{m['kl']:>10.4e} {m['cos']:>9.6f} {m['pcc']:>9.6f} "
              f"{m['rouge']['rouge1']:>8.4f} {m['rouge']['rouge2']:>8.4f} "
              f"{m['rouge']['rougeL']:>8.4f} {div:>5} "
              f"{m['n_steps']:>3}/{m['n_fp_steps']:<3}")
    print("-" * 72)
    print(f"  {'MEAN':<32} {_QUANT_METHOD:<8} "
          f"{np.mean(acc['kl']):>10.4e} {np.mean(acc['cos']):>9.6f} "
          f"{np.mean(acc['pcc']):>9.6f} "
          f"{np.mean(acc['r1']):>8.4f} {np.mean(acc['r2']):>8.4f} "
          f"{np.mean(acc['rL']):>8.4f} {'':>5}")
    print(f"  {'MEAN':<32} {'fullprec':<8} {'':>10} {'':>9} {'':>9} "
          f"{np.mean(fp_rouge['r1']):>8.4f} {np.mean(fp_rouge['r2']):>8.4f} "
          f"{np.mean(fp_rouge['rL']):>8.4f} {'':>5}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f">>>>>>>>>>>>> Done, elapsed time: {elapsed:.2f} seconds")

#!/usr/bin/env python3
"""Evaluate a SureQuant-quantized LLaVA on MM-NIAH (retrieval-text): fp16 vs w4aN.

Unlike ``llava_quant/llava_wa_kv/sure_vs_turbo_mmniah_kv_compare.py`` -- which
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
what makes the per-step KL / cos / pcc meaningful.  The task metric is the same
substring ``hit`` flag as the reference script, scored against ``sample["answer"]``.

Length caveat (inherited from the reference script): MM-NIAH contexts run from
~600 to ~80k tokens while LLaVA-1.5 supports 4096 positions (576 tokens per image),
so only samples whose estimated prompt length falls within
``[--min-seq-len, --max-seq-len]`` are evaluated.

Usage:
    CUDA_VISIBLE_DEVICES=0 python bench_eval/mmniah_eval_surequant.py
    CUDA_VISIBLE_DEVICES=0 python bench_eval/mmniah_eval_surequant.py --skip-baseline
    CUDA_VISIBLE_DEVICES=0 python bench_eval/mmniah_eval_surequant.py \\
        --max-samples 8 --min-seq-len 2048 --max-seq-len 4096 --max-new-tokens 32
"""

from __future__ import annotations

import argparse
import json
import re
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
_MMNIAH_ROOT = Path("/data1/stu_Jiangtp/data/MM-NIAH/mm_niah_val")
_MMNIAH_ANNOTATIONS = _MMNIAH_ROOT / "annotations" / "retrieval-text.jsonl"
_MMNIAH_IMAGES_ROOT = _MMNIAH_ROOT / "mm_niah_dev" / "images"

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

_ANSWER_INSTRUCTION = "Answer the question using a single word or phrase."

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
        description="Evaluate a SureQuant-quantized LLaVA (w4a16) on MM-NIAH "
                    "against the fp16 baseline; KV cache is full precision."
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
    parser.add_argument("--seed", type=int, default=42)
    return parser


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


def prepare_inputs(processor, sample: dict, images_root: str | Path, device):
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


def first_divergence(fp_ids: torch.Tensor, q_ids: torch.Tensor,
                     prompt_len: int) -> int | None:
    """Index of the first generated step whose argmax differs from the baseline.

    ``None`` when one stream is a prefix of the other.  Reported next to ``kl``
    because once the two greedy streams split, every later step compares
    distributions from *different* contexts, so the mean KL reflects divergence
    rather than weight-quantization error.
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
    seed_everything(args.seed)
    _QUANT_METHOD = args.quant_label or derive_quant_label(args.model_path)

    print("=" * 72)
    print(f"MM-NIAH (retrieval-text) SureQuant {_QUANT_METHOD} vs fp16 "
          f"-- KV cache full precision")
    print("=" * 72)

    fp16_model, quant_model, processor = load_models(args)
    _release_cuda_memory()

    samples = _select_samples(processor, args)
    if not samples:
        raise SystemExit("no MM-NIAH sample fits --max-seq-len; raise it or lower --start-index")
    print(f"[infer] {len(samples)} sample(s), max_new_tokens={args.max_new_tokens}, "
          f"baseline={'off' if args.skip_baseline else 'on'}")

    eos_token_id = processor.tokenizer.eos_token_id
    input_device = _model_input_device(quant_model)
    results = []
    skipped_infer = 0
    for sample in tqdm(samples, desc="MM-NIAH"):
        inputs = prepare_inputs(processor, sample, args.images_root, input_device)
        prompt_len = int(inputs["input_ids"].shape[1])
        if prompt_len > args.max_seq_len:
            # Exact tokenization disagreed with the estimate: skip.
            skipped_infer += 1
            del inputs
            _release_cuda_memory()
            continue

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
        if fp is not None:
            metrics["fp_hit"] = answer_hit(fp["text"], sample["answer"])
            fp_ids = fp["generated"][0]
            q_ids = quant["generated"][0]
            kl, n_steps = compute_step_kl(fp["step_logits"], quant["step_logits"])
            # cos/pcc ride the same per-step logits -- and the same n -- as ``kl``:
            # they measure the next-token distributions, not the sampled token ids.
            # Ids are arbitrary vocabulary indices, so correlating them said nothing
            # about similarity and, once the streams diverge, degenerated (a 2-token
            # sample forces PCC to exactly +/-1).
            cos, pcc, _ = compute_step_cos_pcc(fp["step_logits"], quant["step_logits"])
            metrics[_QUANT_METHOD] = {
                "kl": kl,
                "n_steps": n_steps,
                "n_fp_steps": len(fp["step_logits"]),
                "first_div": first_divergence(fp_ids, q_ids, prompt_len),
                "cos": cos,
                "pcc": pcc,
                "hit": answer_hit(quant["text"], sample["answer"]),
            }
        else:
            metrics[_QUANT_METHOD] = {
                "hit": answer_hit(quant["text"], sample["answer"]),
            }

        record = {
            "id": sample.get("id"),
            "image_paths": resolve_image_paths(sample, args.images_root),
            "num_images": len(sample["images_list"]),
            "seq_len": prompt_len,
            "question": sample["question"],
            "answer": sample["answer"],
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
              f"| images={record['num_images']} | q={sample['question']!r}")
        print(f"  answer      = {sample['answer']!r}")
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
                "annotations": args.annotations,
                "images_root": args.images_root,
                "model_path": args.model_path,
                "baseline_checkpoint": None if args.skip_baseline else args.baseline_checkpoint,
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
    hits = [float(r["metrics"][_QUANT_METHOD]["hit"]) for r in results]
    summary = {
        "n": len(results),
        _QUANT_METHOD: {"hit": float(np.mean(hits)) if hits else 0.0},
    }
    if not args.skip_baseline:
        fp_hits = [float(r["metrics"]["fp_hit"]) for r in results]
        summary["fullprec_hit"] = float(np.mean(fp_hits)) if fp_hits else 0.0
        for key in ("kl", "cos", "pcc"):
            values = [r["metrics"][_QUANT_METHOD][key] for r in results]
            summary[f"{_QUANT_METHOD}_{key}"] = float(np.mean(values))
    return summary


def _print_summary(results: list[dict], args: argparse.Namespace) -> None:
    """Per-sample table plus the w4a16-vs-fp16 means."""
    print("\n" + "=" * 72)
    if args.skip_baseline:
        print("Hit rate (quantized model only; no fp16 baseline was run)")
        print("=" * 72)
        print(f"  {'id':<8} {'seq_len':>8} {'images':>7} {'hit':>6}")
        for r in results:
            print(f"  {str(r['id']):<8} {r['seq_len']:>8} {r['num_images']:>7} "
                  f"{str(r['metrics'][_QUANT_METHOD]['hit']):>6}")
        print("-" * 72)
        print(f"  {'MEAN':<8} {'':>8} {'':>7} "
              f"{np.mean([float(r['metrics'][_QUANT_METHOD]['hit']) for r in results]):>6.3f}")
        return

    # All three distributional columns share one convention: they compare the
    # per-step next-token *distributions* over the min(n_fp, n_quant) steps both
    # greedy streams produced, then average over those steps.  ``div`` locates the
    # step where the two streams split, so a small ``div`` next to a large ``kl``
    # is read as divergence rather than as weight-quantization error.
    print("vs full-precision baseline")
    print("=" * 72)
    print(f"  {'id':<8} {'method':<8} {'kl':>10} {'cos_sim':>10} {'pearson':>10} "
          f"{'hit':>6} {'div':>5} {'steps':>7}  "
          f"(kl in nats/step; cos/pcc are per-step means over the next-token logits; "
          f"div = first differing step, - = none)")
    acc = {"kl": [], "cos": [], "pcc": [], "hit": []}
    fp_hits = []
    for r in results:
        fp_hits.append(float(r["metrics"]["fp_hit"]))
        m = r["metrics"][_QUANT_METHOD]
        acc["kl"].append(m["kl"])
        acc["cos"].append(m["cos"])
        acc["pcc"].append(m["pcc"])
        acc["hit"].append(float(m["hit"]))
        div = "-" if m["first_div"] is None else str(m["first_div"])
        print(f"  {str(r['id']):<8} {_QUANT_METHOD:<8} "
              f"{m['kl']:>10.4e} {m['cos']:>10.6f} {m['pcc']:>10.6f} "
              f"{str(m['hit']):>6} {div:>5} "
              f"{m['n_steps']:>3}/{m['n_fp_steps']:<3}")
    print("-" * 72)
    print(f"  {'MEAN':<8} {_QUANT_METHOD:<8} "
          f"{np.mean(acc['kl']):>10.4e} {np.mean(acc['cos']):>10.6f} "
          f"{np.mean(acc['pcc']):>10.6f} {np.mean(acc['hit']):>6.3f}")
    print(f"  {'MEAN':<8} {'fullprec':<8} {'':>10} {'':>10} {'':>10} "
          f"{np.mean(fp_hits):>6.3f}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f">>>>>>>>>>>>> Done, elapsed time: {elapsed:.2f} seconds")

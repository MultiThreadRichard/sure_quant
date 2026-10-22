"""MMBench evaluation for a SureQuant-quantized LLaVA model and its fp16 base.

``--model-type`` selects what is graded: ``quantized`` (the default) loads a
SureQuant save through ``--model-path``, ``fp16`` loads the full-precision
checkpoint instead. Both modes write:

* ``answers_<tag>.jsonl`` -- one record per (question, round), schema-compatible
  with ShareGPT4V's ``scripts/convert_mmbench_for_submission.py``.
* ``<tag>.xlsx`` -- the MMBench submission file (same column surgery as that
  converter: drop the hidden columns, insert ``prediction`` at position 6).
* ``metrics_<tag>.json`` -- accuracy overall and per category, when the
  annotation file carries answers (the dev split does; the test split does not).

The prompt follows ShareGPT4V's ``eval/model_vqa_mmbench.py`` (hint + question +
lettered options + "Answer with the option's letter..."), but the chat template
comes from this repo's ``llava_wa.data.make_prompt`` so the text lands in the
exact format the model was calibrated with. The processor is loaded from the
graded checkpoint itself, and the base checkpoint and its SureQuant save share an
identical chat template, so the two modes render byte-identical prompts and their
scores are directly comparable. Option rotation for the standard CircularEval
protocol is available via ``--all-rounds``.

Note on scoring: the dev answers are graded locally by extracting the chosen
letter from the generated text, which is a heuristic (MMBench's own server uses
a more involved parser). ``--max-new-tokens``, ``--temperature`` and
``--all-rounds`` all change the number, so quote the flags alongside any score.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import re
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
print(f"REPO_ROOT: {REPO_ROOT}")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from llava_quant.llava_wa.config import parse_dtype  # noqa: E402
from llava_quant.llava_wa.data import _model_input_device, make_prompt  # noqa: E402
from llava_quant.llava_wa.persistence import load_quantized_model  # noqa: E402

ALL_OPTIONS = ("A", "B", "C", "D")
DEFAULT_MODEL_PATH = (
    "/data1/stu_Jiangtp/sure_quant/model_saved/"
    "llava_sure_all_w4a16_sharegpt4v_00/best_quantized_model"
)
# Full-precision baseline, i.e. the checkpoint the SureQuant save was quantized
# from. Kept as a literal rather than imported from ``llava_wa.config.CHECKPOINT``
# so this eval script does not follow that file's calibration-time edits.
DEFAULT_FP16_CHECKPOINT = "/data1/stu_Jiangtp/models/llava-1.5-7b-hf"
DEFAULT_QUESTION_FILE = (
    "/data1/stu_Jiangtp/data/ShareGPT4V/playground/data/eval/mmbench/"
    "mmbench_dev_20230712.tsv"
)
SINGLE_PRED_SUFFIX = "Answer with the option's letter from the given choices directly."
# Columns the official submission converter removes before inserting `prediction`.
HIDDEN_COLUMNS = ("hint", "category", "source", "image", "comment", "l2-category")

# Ordered most-specific first. Applied to the uppercased output.
_CHOICE_PATTERNS = (
    re.compile(r"^\s*\(?([A-D])\)?\s*[.:,\-]?\s*$"),  # a bare "B" / "B." / "(B)"
    re.compile(r"ANSWER\s*(?:IS|:|\s)\s*\(?([A-D])\b"),  # "the answer is B"
    re.compile(r"^\s*\(?([A-D])\)?\s*[.:)]"),  # "B. because ..." / "(B) ..."
    re.compile(r"\bOPTION\s*\(?([A-D])\b"),  # "option B is correct"
)


def is_none(value: Any) -> bool:
    """Mirror ShareGPT4V's blank/NaN option test."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.lower() in {"nan", "none"}:
        return True
    return False


def get_options(row: pd.Series) -> list[str]:
    """Collect consecutive non-blank A/B/C/D values (some rows have only 2-3)."""
    options = []
    for option in ALL_OPTIONS:
        value = row[option]
        if is_none(value):
            break
        options.append(value)
    return options


def load_image_from_base64(encoded: str) -> Image.Image:
    """Decode the TSV's inline base64 JPEG into an RGB PIL image."""
    payload = encoded.strip()
    if payload.startswith("data:"):  # tolerate a data-URI prefix
        payload = payload.split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(payload))) as image:
        return image.convert("RGB")


def build_question(row: pd.Series, options: list[str], single_pred_prompt: bool) -> str:
    """Assemble the question text exactly as the reference eval does.

    The image placeholder is *not* included here: ``make_prompt`` injects it via
    the chat template, so adding it would emit two image tokens for one image.
    """
    question = row["question"]
    hint = row["hint"]
    if not is_none(hint):
        question = f"{hint}\n{question}"
    for letter, option in zip(ALL_OPTIONS, options):
        question = f"{question}\n{letter}. {option}"
    if single_pred_prompt:
        question = f"{question}\n{SINGLE_PRED_SUFFIX}"
    return question


def extract_choice(text: str, valid_letters: str) -> str | None:
    """Best-effort extraction of the chosen letter from a generated answer."""
    stripped = text.strip()
    if not stripped:
        return None
    upper = stripped.upper()
    if len(upper) == 1 and upper in valid_letters:
        return upper
    for pattern in _CHOICE_PATTERNS:
        match = pattern.search(upper)
        if match and match.group(1) in valid_letters:
            return match.group(1)
    # Final fallback, restricted to short outputs: there a standalone letter is
    # almost certainly the answer, while in a long rambling answer it is usually
    # an article and guessing would silently corrupt the score.
    if len(stripped) <= 32:
        match = re.search(r"\b([A-D])\b", upper)
        if match and match.group(1) in valid_letters:
            return match.group(1)
    return None


def get_chunk(items: list[Any], num_chunks: int, chunk_idx: int) -> list[Any]:
    """Split ``items`` into ``num_chunks`` contiguous shards and take one."""
    if num_chunks < 1:
        raise ValueError("--num-chunks must be at least 1")
    if not 0 <= chunk_idx < num_chunks:
        raise ValueError(f"--chunk-idx must be in [0, {num_chunks - 1}]")
    if num_chunks == 1:
        return items
    chunk_size = math.ceil(len(items) / num_chunks)
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    return chunks[chunk_idx]


def resolve_model_source(args: argparse.Namespace) -> None:
    """Fill in the mode-dependent defaults for ``--model-path`` and ``--tag``.

    Mutates ``args`` in place, so every later consumer (the loader, ``model_id``,
    the artifact paths, the submission xlsx) agrees on what was graded. The
    defaults are keyed off ``--model-type``: an fp16 run without ``--tag`` gets its
    own artifact names, so it cannot silently overwrite the quantized answers in
    the shared default ``--output-dir``.
    """
    if args.model_path is None:
        args.model_path = (
            DEFAULT_FP16_CHECKPOINT if args.model_type == "fp16" else DEFAULT_MODEL_PATH
        )
    args.model_path = str(Path(args.model_path).expanduser())
    if not args.tag:
        args.tag = "fp16" if args.model_type == "fp16" else "surequant_w4a16"
    print(
        f"Model source: {args.model_path} "
        f"(model-type={args.model_type}, tag={args.tag})"
    )


def load_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any]:
    """Load the checkpoint selected by ``--model-type`` plus its processor.

    The processor is always loaded from the same directory as the model, so the
    chat template follows whichever checkpoint is being graded. Call
    :func:`resolve_model_source` first: this reads the already-resolved
    ``args.model_path``.
    """
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model_path = Path(args.model_path)
    dtype = parse_dtype(args.torch_dtype)
    if args.model_type == "fp16":
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(
                f"Not a HF model directory (no config.json): {model_path}"
            )
        print(f"Loading full-precision model from {model_path} ({args.torch_dtype})")
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, device_map=args.device_map, torch_dtype=dtype
        )
    else:
        if not (model_path / "surequant_config.json").is_file():
            raise FileNotFoundError(
                f"Not a SureQuant checkpoint (no surequant_config.json): {model_path}"
            )
        print(f"Loading quantized model from {model_path}")
        model = load_quantized_model(
            model_path, device_map=args.device_map, torch_dtype=dtype
        )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


@torch.inference_mode()
def generate_answer(
    model: Any,
    processor: Any,
    image: Image.Image,
    question: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float | None,
    num_beams: int,
) -> str:
    """Single-image, single-turn generation; returns only the assistant text."""
    inputs = processor(
        images=image, text=make_prompt(processor, question), return_tensors="pt"
    ).to(_model_input_device(model))

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "use_cache": True,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
    if num_beams > 1:
        # beam search is incompatible with sampling
        gen_kwargs.update(num_beams=num_beams, do_sample=False)

    generated_ids = model.generate(**inputs, **gen_kwargs)
    prompt_length = inputs["input_ids"].shape[1]
    answer = processor.batch_decode(
        generated_ids[:, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()
    del inputs, generated_ids
    return answer


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Walk the annotation file, generate answers, and persist every artifact."""
    resolve_model_source(args)
    questions = pd.read_table(Path(args.question_file).expanduser())
    if args.limit > 0:
        questions = questions.iloc[: args.limit]
    rows = get_chunk(list(questions.iterrows()), args.num_chunks, args.chunk_idx)
    print(f"Evaluating {len(rows)} questions (chunk {args.chunk_idx}/{args.num_chunks})")

    model, processor = load_model_and_processor(args)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    answers_path = output_dir / f"answers_{args.tag}.jsonl"
    metrics_path = output_dir / f"metrics_{args.tag}.json"

    # Per (question_id, round): correctness is only known when the split has answers.
    graded: dict[int, list[bool]] = defaultdict(list)
    per_category: dict[str, list[bool]] = defaultdict(list)
    per_l2_category: dict[str, list[bool]] = defaultdict(list)
    unparsed = 0
    start = time.time()

    with answers_path.open("w", encoding="utf-8") as ans_file:
        for position, (_row_index, row) in enumerate(rows, start=1):
            question_id = int(row["index"])  # numpy int64 is not JSON-serializable
            options = get_options(row)
            if not options:
                print(f"[Warning] question {question_id} has no options; skipping")
                continue
            valid_letters = "".join(ALL_OPTIONS[: len(options)])
            image = load_image_from_base64(row["image"])
            answer_key = row["answer"]
            has_answer = not is_none(answer_key)
            if has_answer:
                answer_index = ALL_OPTIONS.index(str(answer_key).strip().upper())

            num_rounds = len(options) if args.all_rounds else 1
            for round_idx in range(num_rounds):
                # In round r the options are rotated left r times, so the option
                # originally at index i now sits at display position (i - r) % n
                # and therefore carries the *displayed* letter
                # ALL_OPTIONS[(i - r) % n] -- that is what the model must answer.
                rotated = list(options[round_idx:]) + list(options[:round_idx])
                # ShareGPT4V's ``option_char`` convention differs: position k holds
                # the *original* label of the option displayed there, not the letter
                # shown at k. It maps a model answer back to the original option;
                # round 0 is the identity, and only round 0 feeds the submission.
                rotated_letters = (
                    ALL_OPTIONS[round_idx: len(options)] + ALL_OPTIONS[:round_idx]
                )
                question = build_question(row, rotated, args.single_pred_prompt)
                text = generate_answer(
                    model,
                    processor,
                    image,
                    question,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                )
                choice = extract_choice(text, valid_letters)
                expected = None
                if has_answer:
                    # The letter the model is shown for the correct option is its
                    # display position, NOT its original label -- indexing
                    # rotated_letters here cancels the rotation and silently grades
                    # every round but the first against the wrong option.
                    expected = ALL_OPTIONS[(answer_index - round_idx) % len(options)]
                    assert (
                        rotated_letters[ALL_OPTIONS.index(expected)]
                        == ALL_OPTIONS[answer_index]
                    ), f"rotation bookkeeping broken at round {round_idx} (id {question_id})"
                    correct = choice == expected
                    graded[question_id].append(correct)

                if choice is None:
                    unparsed += 1
                prediction = choice if choice is not None else text
                ans_file.write(
                    json.dumps(
                        {
                            "question_id": question_id,
                            "round_id": round_idx,
                            "prompt": question,
                            "text": text,
                            "options": rotated,
                            "option_char": rotated_letters,
                            "prediction": prediction,
                            "expected": expected,
                            "answer_id": uuid.uuid4().hex,
                            "model_id": Path(args.model_path).name,
                            "model_type": args.model_type,
                            "metadata": {
                                "category": None if is_none(row["category"]) else row["category"],
                                "l2_category": None
                                if is_none(row["l2-category"])
                                else row["l2-category"],
                                "correct": None if expected is None else (choice == expected),
                            },
                        }
                    )
                    + "\n"
                )
                ans_file.flush()
                if position % 20 == 0 or position == len(rows):
                    elapsed = time.time() - start
                    print(
                        f"[{position}/{len(rows)}] round {round_idx + 1}/{num_rounds} "
                        f"elapsed {elapsed:.0f}s ({position / max(elapsed, 1e-9):.2f} q/s)"
                    )

    # A question counts as correct under CircularEval only if every round is.
    totals = {qid: all(flags) for qid, flags in graded.items()}
    for qid, flags in graded.items():
        row = questions.loc[questions["index"] == qid].iloc[0]
        per_category[str(row["category"])].append(all(flags))
        per_l2_category[str(row["l2-category"])].append(all(flags))
    graded_count = len(totals)
    correct_count = sum(totals.values())
    metrics = {
        "tag": args.tag,
        "model_type": args.model_type,
        "model_path": str(args.model_path),
        "question_file": str(args.question_file),
        "num_questions_evaluated": len(rows),
        "num_questions_graded": graded_count,
        "num_unparsed_outputs": unparsed,
        "accuracy": (correct_count / graded_count) if graded_count else None,
        "circular_eval": bool(args.all_rounds),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_beams": args.num_beams,
            "single_pred_prompt": args.single_pred_prompt,
        },
        "per_category": {
            name: sum(flags) / len(flags) for name, flags in sorted(per_category.items())
        },
        "per_l2_category": {
            name: sum(flags) / len(flags)
            for name, flags in sorted(per_l2_category.items())
        },
        "elapsed_seconds": round(time.time() - start, 1),
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {answers_path}")
    print(f"Wrote {metrics_path}")
    if metrics["accuracy"] is not None:
        print(
            f"Accuracy: {metrics['accuracy']:.4f} "
            f"({correct_count}/{graded_count} questions"
            f"{', CircularEval' if args.all_rounds else ''})"
        )
    if unparsed:
        print(f"[Warning] {unparsed} outputs had no extractable choice letter")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def build_submission(
    args: argparse.Namespace, answers_path: Path, xlsx_path: Path
) -> None:
    """Write the MMBench submission xlsx from the answer records.

    Mirrors ``convert_mmbench_for_submission.py``: drop the hidden columns, add
    ``prediction`` at position 6, and fill it by matching index to question_id.
    Under ``--all-rounds`` the first round's prediction is used, since the
    submission format expects exactly one row per question.
    """
    frame = pd.read_table(Path(args.question_file).expanduser())
    submission = frame.copy().drop(columns=list(HIDDEN_COLUMNS))
    submission.insert(6, "prediction", None)

    first_round: dict[int, str] = {}
    for line in answers_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["round_id"] == 0:
            first_round[record["question_id"]] = (
                record["prediction"] if args.prediction_format == "letter" else record["text"]
            )
    for question_id, prediction in first_round.items():
        submission.loc[frame["index"] == question_id, "prediction"] = prediction

    missing = int(submission["prediction"].isna().sum())
    if missing:
        print(f"[Warning] {missing} questions have no prediction (not evaluated in this chunk)")
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_excel(xlsx_path, index=False, engine="openpyxl")
    print(f"Wrote {xlsx_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a SureQuant-quantized LLaVA on MMBench dev."
    )
    parser.add_argument(
        "--model-type",
        choices=("quantized", "fp16"),
        default="quantized",
        help=(
            "quantized grades a SureQuant save via load_quantized_model; fp16 "
            "grades the full-precision HF checkpoint with plain from_pretrained. "
            "Selects the default for --model-path and --tag."
        ),
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Checkpoint to grade. Defaults to the SureQuant save for "
            "--model-type quantized, or to the fp16 base checkpoint for fp16."
        ),
    )
    parser.add_argument("--question-file", default=DEFAULT_QUESTION_FILE)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "logs" / "mmbench_eval"))
    parser.add_argument(
        "--tag",
        default=None,
        help="Artifact name suffix. Defaults to the --model-type (surequant_w4a16 or fp16).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 uses greedy decoding (reproducible); ShareGPT4V's reference used 0.2.",
    )
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument(
        "--all-rounds",
        action="store_true",
        help="Rotate options and ask each question len(options) times (CircularEval).",
    )
    parser.add_argument(
        "--single-pred-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append the 'answer with the option's letter' instruction.",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates every question.")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument(
        "--prediction-format",
        choices=("letter", "text"),
        default="letter",
        help="What the submission xlsx stores; raw text always goes to the jsonl.",
    )
    parser.add_argument(
        "--no-submission", action="store_true", help="Skip writing the xlsx."
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = run_eval(args)
    if not args.no_submission:
        build_submission(
            args,
            Path(args.output_dir).expanduser() / f"answers_{args.tag}.jsonl",
            Path(args.output_dir).expanduser() / f"{args.tag}.xlsx",
        )
    if metrics["accuracy"] is not None:
        print(f"Done. Accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f">>>>>>>>>>>>> Done, elapsed time: {elapsed:.2f} seconds")

"""ShareGPT4V-calibrated variant of the SureQuant LLaVA grid-search CLI.

Compared with ``llava_quant/grid_search/llava_quant_calib_wa_grid_search.py``
the only change is the calibration data source. :func:`run_grid_search` below
redirects ``llava_wa.search.load_calib_data`` to
:func:`load_sharegpt4v_calib_data`, which samples COCO train2017 image/caption
pairs from the ShareGPT4V 100k annotations instead of reading the flickr30k
parquet shard. Quantization, calibration, scoring, and persistence all remain
the shared implementation, so the two entry points cannot drift apart.
"""

from __future__ import annotations

import gc
import json
import os
import random
import re
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
print(f"REPO_ROOT: {REPO_ROOT}")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch import nn  # noqa: E402

from llava_quant.llava_wa import search as search_module  # noqa: E402
from llava_quant.llava_wa.config import build_parser, CHECKPOINT  # noqa: E402
from llava_quant.llava_wa.data import _model_input_device, make_prompt  # noqa: E402
from llava_quant.llava_wa.modeling import selected_linear_names  # noqa: E402



SHAREGPT4V_ANNOTATION = Path(
    "/data1/stu_Jiangtp/data/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap100k.json"
)
SHAREGPT4V_IMAGE_ROOT = Path("/data1/stu_Jiangtp/data/ShareGPT4V/coco/train2017")
# COCO train2017 filenames are zero-padded 12-digit ids, e.g. 000000000009.jpg.
COCO_TRAIN2017_FILENAME = re.compile(r"^\d{12}\.jpg$")
# ShareGPT4V appends this literal token to every human turn; ``make_prompt``
# inserts the real image placeholder, so the literal copy must be stripped.
IMAGE_TOKEN = "<image>"


def _human_instruction(record: dict[str, Any], fallback: str) -> str:
    """Return the first human turn's text, minus its literal image token.

    ``make_prompt`` adds the real image placeholder to the chat template, so the
    literal ``<image>`` that ShareGPT4V appends to each human turn is removed
    here; keeping it would emit two placeholders for a single image. Records
    without a usable human turn fall back to ``fallback``.
    """
    for turn in record.get("conversations") or ():
        if turn.get("from") == "human" and isinstance(turn.get("value"), str):
            text = turn["value"].replace(IMAGE_TOKEN, " ").strip()
            if text:
                return text
            break
    return fallback


def select_sharegpt4v_samples(
    *,
    num_samples: int,
    seed: int,
    annotation_path: str | Path = SHAREGPT4V_ANNOTATION,
    image_root: str | Path = SHAREGPT4V_IMAGE_ROOT,
    image_column: str = "image",
    prompt_fallback: str = "",
    show_sample: bool = True,
) -> list[dict[str, Any]]:
    """Return ``num_samples`` COCO train2017 image/instruction pairs.

    Samples are kept only when the annotation's image basename follows the COCO
    ``000000xxxxxx.jpg`` convention *and* the file exists under ``image_root``;
    duplicate image paths collapse to their first occurrence. Each returned dict
    carries the record ``id``, the resolved ``image`` path, and the ``prompt``
    taken from the record's first human turn. The draw uses a private
    ``random.Random(seed)`` instance, so it is reproducible regardless of the
    global RNG state, and the result is sorted by annotation id for a stable,
    comparable calibration set.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    annotation_path = Path(annotation_path).expanduser()
    image_root = Path(image_root).expanduser()
    with annotation_path.open(encoding="utf-8") as annotation_file:
        records = json.load(annotation_file)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list of records in {annotation_path}")

    candidates: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    skipped_missing = 0
    for record in records:
        relative_path = record.get(image_column)
        if not isinstance(relative_path, str):
            continue
        filename = os.path.basename(relative_path)
        if not COCO_TRAIN2017_FILENAME.match(filename):
            continue
        image_path = image_root / filename
        if not image_path.is_file():
            skipped_missing += 1
            continue
        image_key = str(image_path)
        if image_key in seen_images:
            continue
        seen_images.add(image_key)
        candidates.append(
            {
                "id": record.get("id"),
                "image": image_key,
                "prompt": _human_instruction(record, prompt_fallback),
            }
        )

    print(
        f"ShareGPT4V filter: {len(candidates)} unique COCO train2017 pairs "
        f"from {len(records)} annotations ({skipped_missing} missing images skipped)"
    )
    if num_samples > len(candidates):
        raise ValueError(
            f"Requested {num_samples} calibration samples but only "
            f"{len(candidates)} are available under {image_root}"
        )
    selected = random.Random(seed).sample(candidates, num_samples)
    selected.sort(key=lambda item: str(item["id"]))
    print(f"Sampled {len(selected)} calibration pairs with seed={seed}")
    if show_sample and selected:
        example = selected[0]
        print(f"Example calibration pair: id={example['id']}")
        print(f"  image : {example['image']}")
        print(f"  prompt: {example['prompt']!r}")
    return selected


@torch.inference_mode()
def collect_calibration_data(
    model: nn.Module,
    processor: Any,
    samples: Iterable[dict[str, Any]],
    *,
    max_samples_per_layer: int,
    seed: int,
    layer_names: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Bounded CPU-resident activation collection with per-sample prompts.

    Mirrors ``llava_wa.data.collect_calibration_data`` (same hook placement and
    same bounded reservoir per layer), except that every sample supplies its own
    instruction through ``sample["prompt"]``. The shared version applies one
    fixed prompt to all samples, which cannot express ShareGPT4V's per-image
    instructions. Samples must carry a non-empty ``prompt``;
    :func:`select_sharegpt4v_samples` guarantees this by substituting its
    ``prompt_fallback`` for records without a usable human turn.
    """
    model.eval()
    activations: dict[str, torch.Tensor] = {}
    generator = torch.Generator().manual_seed(seed)

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
            if not inputs or inputs[0] is None:
                return
            value = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu()
            if name in activations:
                value = torch.cat((activations[name], value), dim=0)
            if len(value) > max_samples_per_layer:
                indices = torch.randperm(len(value), generator=generator)
                value = value[indices[:max_samples_per_layer]]
            activations[name] = value
        return hook

    handles = [
        module.register_forward_hook(make_hook(name))
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and (layer_names is None or name in layer_names)
    ]
    input_device = _model_input_device(model)
    try:
        for sample in samples:
            prompt = make_prompt(processor, sample["prompt"])
            with Image.open(sample["image"]) as image:
                rgb_image = image.convert("RGB")
            inputs = processor(
                images=rgb_image, text=prompt, return_tensors="pt"
            ).to(input_device)
            model(**inputs)
            del inputs, rgb_image
    finally:
        for handle in handles:
            handle.remove()

    for name, values in activations.items():
        activations[name] = values.contiguous()
        print(f"Kept {len(values)} calibration vectors for {name}")
    return activations


def load_sharegpt4v_calib_data(
    *,
    calibration_sample_num: int,
    max_samples_per_layer: int,
    prompt_text: str,
    image_column: str,
    block_size: int,
    device_map: str,
    torch_dtype: torch.dtype,
    seed: int,
    quantize_vision: bool,
    quantize_mm_proj: bool,
    quantize_language: bool,
    annotation_path: str | Path = SHAREGPT4V_ANNOTATION,
    image_root: str | Path = SHAREGPT4V_IMAGE_ROOT,
) -> tuple[Any, dict[str, torch.Tensor]]:
    """Signature-compatible replacement for ``llava_wa.data.load_calib_data``.

    Mirrors the original contract (returns the processor plus bounded CPU
    activation tensors) but differs in two ways: samples come from ShareGPT4V
    COCO train2017 annotations rather than the ``datasets`` parquet shard, and
    each sample is calibrated with its own ShareGPT4V instruction instead of the
    single ``prompt_text``. ``prompt_text`` is still accepted because the caller
    passes it, but it now only serves as the fallback instruction for records
    without a usable human turn. Activation collection uses the configured
    ``CHECKPOINT``.
    """
    try:
        from transformers import AutoProcessor, LlavaForConditionalGeneration
    except ImportError as error:
        raise RuntimeError("Calibration loading requires the transformers package") from error

    processor = AutoProcessor.from_pretrained(CHECKPOINT)
    model = LlavaForConditionalGeneration.from_pretrained(
        CHECKPOINT, device_map=device_map, torch_dtype=torch_dtype
    )

    samples = select_sharegpt4v_samples(
        num_samples=calibration_sample_num,
        seed=seed,
        annotation_path=annotation_path,
        image_root=image_root,
        image_column=image_column,
        prompt_fallback=prompt_text,
    )
    layer_names = selected_linear_names(
        model,
        block_size=block_size,
        quantize_vision=quantize_vision,
        quantize_mm_proj=quantize_mm_proj,
        quantize_language=quantize_language,
    )
    calibration_data = collect_calibration_data(
        model,
        processor,
        samples,
        max_samples_per_layer=max_samples_per_layer,
        seed=seed,
        layer_names=layer_names,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Released full-precision calibration model memory")
    return processor, calibration_data


def _validate_paths(args: Any) -> None:
    """Fail fast on unusable paths before any 7B model is loaded.

    Every problem is collected and reported at once, so a run is not restarted
    once per typo. Only the paths this CLI actually opens are checked: the
    checkpoint, the ``--test-images`` the best-trial preview reopens on every
    new best, the ShareGPT4V annotation file and image root, and the output
    directory (created here so a permissions problem surfaces up front rather
    than after the first trial). ``--best-trial-config`` is deliberately not
    checked because this grid-search path never reads it.
    """
    problems: list[str] = []

    checkpoint = Path(CHECKPOINT)
    if not checkpoint.is_dir():
        problems.append(f"checkpoint directory is missing: {checkpoint}")
    elif not (checkpoint / "config.json").is_file():
        problems.append(f"checkpoint has no config.json (not a HF model dir): {checkpoint}")

    for image_path in args.test_images:
        if not Path(image_path).expanduser().is_file():
            problems.append(f"--test-images file is missing: {image_path}")

    annotation = Path(SHAREGPT4V_ANNOTATION)
    if not annotation.is_file():
        problems.append(f"ShareGPT4V annotations are missing: {annotation}")

    image_root = Path(SHAREGPT4V_IMAGE_ROOT)
    if not image_root.is_dir():
        problems.append(f"ShareGPT4V image root is missing: {image_root}")

    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        problems.append(f"--output-dir cannot be created ({error}): {output_dir}")
    else:
        if not os.access(output_dir, os.W_OK):
            problems.append(f"--output-dir is not writable: {output_dir}")

    if problems:
        raise RuntimeError(
            "Pre-flight path check failed:\n  - " + "\n  - ".join(problems)
        )
    print(f"Pre-flight path check passed; output dir: {output_dir}")


def run_grid_search(args: Any) -> dict[str, Any]:
    """Run the shared grid search with ShareGPT4V COCO calibration data."""
    args.mode = "grid"
    print(f"args: {args}")
    _validate_paths(args)
    print(
        f"Calibration source: {SHAREGPT4V_ANNOTATION} "
        f"(images under {SHAREGPT4V_IMAGE_ROOT})"
    )
    original_loader = search_module.load_calib_data
    search_module.load_calib_data = load_sharegpt4v_calib_data
    try:
        return search_module.run_grid_search(args)
    finally:
        search_module.load_calib_data = original_loader


def grid_search() -> None:
    args = build_parser().parse_args()
    start = time.time()
    summary = run_grid_search(args)
    print(
        f"Best trial: {summary['best_trial']}; "
        f"validation MSE: {summary['best_score']:.8g}; "
        f"elapsed: {time.time() - start:.2f}s"
    )


def main() -> None:
    grid_search()


if __name__ == "__main__":
    main()
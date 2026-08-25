"""Smoke test that loads the saved best quantized LLaVA model and runs
descriptive generation with the user's prompt for every image inside
the project's ``val`` directory (and its subdirectories).

This test requires an NVIDIA GPU because the checkpoint was produced with
CUDA tensors; it is intentionally skipped on CPU-only environments.

CUDA_VISIBLE_DEVICES=0 nohup pytest test_best_model_val.py -s -v > speed_test_log/test_SureQuant-llm-w4a16.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup pytest test_best_model_val.py -s -v > speed_test_log/test_SureQuant-w4a4.log 2>&1 &

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import pytest
import torch

# ---------------------------------------------------------------------------
# Project paths / constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# TODO
# BEST_MODEL_DIR = "/home/ecnu01/sure_quant_models/w4a16_language_only/20260823/best_quantized_model"  # SureQuant-llm-w4a16
BEST_MODEL_DIR = "/home/ecnu01/sure_quant_models/20260808/best_quantized_model"  # SureQuant-w4a4



#VAL_DIR = _PROJECT_ROOT / "tests" / "datasets" / "val_small"
VAL_DIR = Path("/home/ecnu01/workspace/remotesensing_sample")
INFERENCE_PROMPT = "Please describe this image\n"
INFERENCE_MAX_NEW_TOKENS = 128
SAMPLE_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".webp", ".png"}


def _apply_chat_template(processor: Any, text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": text}, {"type": "image"}],
        }
    ]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def _run_one_generation(
    model: Any, processor: Any, image_path: Path
) -> tuple[str, int]:
    """Generate an assistant caption for a single image and return
    ``(assistant_text, generated_tokens)``."""
    from PIL import Image

    prompt = _apply_chat_template(processor, INFERENCE_PROMPT)
    device = next(model.get_input_embeddings().parameters()).device

    with Image.open(image_path) as image:
        inputs = processor(
            images=image.convert("RGB"),
            text=prompt,
            return_tensors="pt",
        ).to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=INFERENCE_MAX_NEW_TOKENS,
            do_sample=False,
        )

    prompt_length = inputs["input_ids"].shape[1]
    assistant_ids = generated_ids[:, prompt_length:]
    assistant_text = processor.batch_decode(
        assistant_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()
    return assistant_text, assistant_ids.shape[1]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def loaded_best_model():
    """Load the saved best quantized model once per test module."""
    # if not BEST_MODEL_DIR.is_dir():
    #     pytest.skip(f"Best-model directory not found: {BEST_MODEL_DIR}")
    # if not torch.cuda.is_available():
    #     pytest.skip("CUDA device is required to load this saved checkpoint")

    from scripts.llava_quant_calib_wa import load_quantized_model

    device_map = "cuda"
    model = load_quantized_model(
        BEST_MODEL_DIR, device_map=device_map, torch_dtype=torch.float16
    )

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(BEST_MODEL_DIR)
    model.eval()
    yield model, processor

    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.inference
def test_best_model_val_images(loaded_best_model):
    """Generate captions for every image under ``val/`` (recursively) and print them."""
    model, processor = loaded_best_model

    # if not VAL_DIR.is_dir():
    #     pytest.skip(f"val directory not found: {VAL_DIR}")

    image_paths: list[Path] = sorted(
        p for p in VAL_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in SAMPLE_IMAGE_SUFFIXES
    )

    if not image_paths:
        pytest.skip(f"No image files found under {VAL_DIR}")

    # Materialize so we can print the count.
    print(f"\n>>> Discovered {len(image_paths)} test image(s) under {VAL_DIR}")
    for p in image_paths:
        print(f"    - {p}")

    results: list[tuple[Path, str, int]] = []
    for image_path in image_paths:
        print(f"\n>>> Running inference on: {image_path}")
        text, num_tokens = _run_one_generation(model, processor, image_path)
        results.append((image_path, text, num_tokens))

    # ---------------- block print ----------------
    print("\n" + "=" * 72)
    print(f"Model dir        : {BEST_MODEL_DIR}")
    print(f"Prompt           : {INFERENCE_PROMPT!r}")
    print(f"# test images    : {len(results)}")
    print("-" * 72)
    for image_path, text, num_tokens in results:
        print(f"Image            : {image_path}")
        print(f"Generated tokens : {num_tokens}")
        print("Assistant output :")
        print(text)
        print("-" * 72)
    print("=" * 72)

    # ---------------- very light assertions ----------------
    assert len(results) > 0, "No test images were discovered"
    for image_path, text, _num_tokens in results:
        assert isinstance(text, str), f"caption is not a string for {image_path}"
        assert len(text) > 0, f"empty caption for {image_path}"

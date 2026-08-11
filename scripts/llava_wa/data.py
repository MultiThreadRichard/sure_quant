from __future__ import annotations

import gc
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from scripts.llava_wa.config import CALIBRATION_DATA_PATHS, CHECKPOINT
from scripts.llava_wa.modeling import selected_linear_names


def make_prompt(processor: Any, text: str) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": text}, {"type": "image"}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def _model_input_device(model: nn.Module) -> torch.device:
    return next(model.get_input_embeddings().parameters()).device


@torch.inference_mode()
def generate_assistant_outputs(
    model: nn.Module,
    processor: Any,
    image_paths: Sequence[str | Path],
    *,
    prompt_text: str,
    max_new_tokens: int,
) -> list[dict[str, str]]:
    """Generate Assistant-only text for the requested images."""
    from PIL import Image

    model.eval()
    prompt = make_prompt(processor, prompt_text)
    device = _model_input_device(model)
    results = []
    for image_path_value in image_paths:
        image_path = Path(image_path_value).expanduser()
        with Image.open(image_path) as image:
            inputs = processor(
                images=image.convert("RGB"), text=prompt, return_tensors="pt"
            ).to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_length = inputs["input_ids"].shape[1]
        assistant_ids = generated_ids[:, prompt_length:]
        assistant_text = processor.batch_decode(
            assistant_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()
        results.append({"image": str(image_path), "assistant": assistant_text})
        print(f"\n===== Best-model inference: {image_path} =====")
        print(f"Assistant: {assistant_text}")
        del inputs, generated_ids, assistant_ids
    return results


def collect_calibration_data(
    model: nn.Module,
    processor: Any,
    samples: Iterable[dict[str, Any]],
    *,
    prompt: str,
    max_samples_per_layer: int,
    seed: int,
    layer_names: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Collect bounded CPU-resident linear inputs."""
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
        with torch.inference_mode():
            for sample in samples:
                inputs = processor(
                    images=sample["image"], text=prompt, return_tensors="pt"
                ).to(input_device)
                model(**inputs)
                del inputs
    finally:
        for handle in handles:
            handle.remove()

    for name, values in activations.items():
        activations[name] = values.contiguous()
        print(f"Kept {len(values)} calibration vectors for {name}")
    return activations


def split_calibration_data(
    data: dict[str, torch.Tensor], validation_fraction: float, seed: int
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    generator = torch.Generator().manual_seed(seed)
    train, validation = {}, {}
    for name, values in data.items():
        if len(values) < 2:
            train[name] = validation[name] = values
            continue
        order = torch.randperm(len(values), generator=generator)
        validation_size = min(max(1, round(len(values) * validation_fraction)), len(values) - 1)
        validation[name] = values[order[:validation_size]]
        train[name] = values[order[validation_size:]]
    return train, validation


def load_calib_data(
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
) -> tuple[Any, dict[str, torch.Tensor]]:
    try:
        from datasets import load_dataset
        from transformers import AutoProcessor, LlavaForConditionalGeneration
    except ImportError as error:
        raise RuntimeError("Calibration loading requires datasets and transformers") from error

    processor = AutoProcessor.from_pretrained(CHECKPOINT)
    model = LlavaForConditionalGeneration.from_pretrained(
        CHECKPOINT, device_map=device_map, torch_dtype=torch_dtype
    )
    data_path = CALIBRATION_DATA_PATHS[0]
    dataset = load_dataset("parquet", data_files=data_path, split="train")
    sample_count = min(calibration_sample_num, len(dataset))
    dataset = dataset.select(range(sample_count))
    print(f"Loaded {sample_count} calibration samples from {data_path}")

    layer_names = selected_linear_names(
        model,
        block_size=block_size,
        quantize_vision=quantize_vision,
        quantize_mm_proj=quantize_mm_proj,
        quantize_language=quantize_language,
    )
    samples = ({"image": item[image_column]} for item in dataset)
    calibration_data = collect_calibration_data(
        model,
        processor,
        samples,
        prompt=make_prompt(processor, prompt_text),
        max_samples_per_layer=max_samples_per_layer,
        seed=seed,
        layer_names=layer_names,
    )
    del model, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Released full-precision calibration model memory")
    return processor, calibration_data


def load_data_for_eval_logits(
    *,
    calibration_sample_num: int,
    eval_sample_num: int,
    prompt_text: str,
    image_column: str,
    device_map: str,
    torch_dtype: torch.dtype,
    max_new_tokens: int = 128,
   ) -> list[tuple[dict[str, torch.Tensor], torch.Tensor]]:
    """Load the last ``eval_sample_num`` samples and run inference to get logits.

    Returns:
       a list of ``(inputs, logits)`` tuples
    """
    try:
        from datasets import load_dataset
        from transformers import AutoProcessor, LlavaForConditionalGeneration
    except ImportError as error:
        raise RuntimeError(
            "load_data_for_eval_logits requires datasets and transformers"
        ) from error

    processor = AutoProcessor.from_pretrained(CHECKPOINT)
    model = LlavaForConditionalGeneration.from_pretrained(
        CHECKPOINT, device_map=device_map, torch_dtype=torch_dtype
    )
    model.eval()
    data_path = CALIBRATION_DATA_PATHS[0]
    dataset = load_dataset("parquet", data_files=data_path, split="train")
    if len(dataset) < calibration_sample_num + eval_sample_num:
        raise ValueError("load_data_for_eval_logits len(dataset) not enough")

    # Select the last ``eval_sample_num`` samples as the held-out eval set.
    total = len(dataset)
    dataset = dataset.select(range(total - eval_sample_num, total))
    print(f"len(dataset) for eval logits: {len(dataset)}")

    prompt = make_prompt(processor, prompt_text)
    input_device = _model_input_device(model)
    eval_pairs: list[tuple[dict[str, torch.Tensor], torch.Tensor]] = []
    with torch.inference_mode():
        for index, sample in enumerate(dataset):
            inputs = processor(
                images=sample[image_column],
                text=prompt,
                return_tensors="pt",
            ).to(input_device)

            output = model.generate(**inputs, max_new_tokens=max_new_tokens)
            # Move inputs back to CPU so the returned pairs are device-agnostic
            cpu_inputs = {k: v.detach().cpu() for k, v in inputs.items()}
            eval_pairs.append((cpu_inputs, output[0].detach().cpu()))

    del model, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f">>>>>>>>>> load_data_for_eval_logits: {len(eval_pairs)} done")
    return eval_pairs


"""Calibrate, tune, and save a SureQuant-quantized LLaVA model.

The grid search is performed over the auxiliary loss weights.  Every trial
starts from the same pretrained checkpoint and random seed, and is ranked by
held-out activation reconstruction MSE.  Comparing the weighted training loss
would be invalid because its scale changes when the lambda values change.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import random
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import torch
import torch.nn as nn

# Allow ``python scripts/llava_quant_calib_wa.py`` without a machine-specific
# sys.path entry.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.default_config import SureQuantConfig
from loss.reconstruction import reconstruction_loss
from model.sure_quant_linear import SureQuantLinear
from model.sure_quantizer import SureQuantizer

if TYPE_CHECKING:
    from transformers import LlavaForConditionalGeneration


DEFAULT_PROMPT = "Please describe this image."
DEFAULT_INFERENCE_PROMPT = "Please describe the animal in this image\n"
DEFAULT_INFERENCE_IMAGES = (
    "/home/ecnu03/workspace/awq_learn/sample_img/sample1.jpg",
    "/home/ecnu03/workspace/awq_learn/sample_img/sample2.jpg",
)
CHECKPOINT = "/home/ecnu03/workspace/models/llava-1.5-7b-hf"
CALIBRATION_DATA_PATHS = (
    "/home/ecnu03/workspace/data/flickr30k/data/test-00000-of-00009.parquet",
)
LOSS_GRID_KEYS = ("lambda_rec", "lambda_dk", "lambda_bal", "lambda_range")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def quantize_linear_layer(
    linear: nn.Linear,
    *,
    num_bits: int,
    block_size: int,
    rotation_strategy: str,
    quantize_weight: bool,
) -> SureQuantLinear:
    """Wrap a compatible linear layer with activation/weight quantizers."""
    activation_quantizer = SureQuantizer(
        dim=linear.in_features,
        block_size=block_size,
        num_bits=num_bits,
        rotation_strategy=rotation_strategy,
    )
    weight_quantizer = None
    if quantize_weight and linear.out_features % block_size == 0:
        weight_quantizer = SureQuantizer(
            dim=linear.out_features,
            block_size=block_size,
            num_bits=num_bits,
            rotation_strategy=rotation_strategy,
        )
    return SureQuantLinear(linear, activation_quantizer, weight_quantizer)


def _replace_linears(
    root: nn.Module,
    *,
    num_bits: int,
    block_size: int,
    rotation_strategy: str,
    quantize_weight: bool,
) -> int:
    # Materialize the list before mutation so named_modules() never walks into
    # a wrapper inserted during this pass.
    linears = [
        (name, module)
        for name, module in root.named_modules()
        if name and isinstance(module, nn.Linear) and "lm_head" not in name
    ]
    replaced = 0
    for name, linear in linears:
        if linear.in_features % block_size != 0:
            print(
                f"Skipping {name}: in_features={linear.in_features} is not "
                f"divisible by block_size={block_size}"
            )
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        setattr(
            parent,
            child_name,
            quantize_linear_layer(
                linear,
                num_bits=num_bits,
                block_size=block_size,
                rotation_strategy=rotation_strategy,
                quantize_weight=quantize_weight,
            ),
        )
        replaced += 1
    return replaced


def quantize_llava_model(
    model: LlavaForConditionalGeneration,
    *,
    num_bits: int = 4,
    block_size: int = 128,
    rotation_strategy: str = "rotation",
    quantize_vision: bool = True,
    quantize_mm_proj: bool = True,
    quantize_language: bool = True,
    quantize_weight: bool = True,
) -> LlavaForConditionalGeneration:
    """Replace selected LLaVA linear layers with SureQuantLinear wrappers."""
    targets: list[tuple[str, nn.Module]] = []
    if quantize_vision:
        targets.append(("vision", model.vision_tower.vision_model.encoder.layers))
    if quantize_mm_proj:
        targets.append(("multimodal projector", model.multi_modal_projector))
    if quantize_language:
        targets.append(("language", model.language_model.model.layers))

    for label, root in targets:
        count = _replace_linears(
            root,
            num_bits=num_bits,
            block_size=block_size,
            rotation_strategy=rotation_strategy,
            quantize_weight=quantize_weight,
        )
        print(f"Wrapped {count} {label} linear layers")
    return model


def make_prompt(processor: Any, text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": text}, {"type": "image"}],
        }
    ]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def selected_linear_names(
    model: LlavaForConditionalGeneration,
    *,
    block_size: int,
    quantize_vision: bool,
    quantize_mm_proj: bool,
    quantize_language: bool,
) -> set[str]:
    """Return full model names for linears that will actually be wrapped."""
    roots: list[nn.Module] = []
    if quantize_vision:
        roots.append(model.vision_tower.vision_model.encoder.layers)
    if quantize_mm_proj:
        roots.append(model.multi_modal_projector)
    if quantize_language:
        roots.append(model.language_model.model.layers)
    selected_ids = {
        id(module)
        for root in roots
        for name, module in root.named_modules()
        if name
        and isinstance(module, nn.Linear)
        and "lm_head" not in name
        and module.in_features % block_size == 0
    }
    return {
        name
        for name, module in model.named_modules()
        if id(module) in selected_ids
    }


def _model_input_device(model: nn.Module) -> torch.device:
    """Return the language embedding device for a possibly dispatched model."""
    embeddings = model.get_input_embeddings()
    return next(embeddings.parameters()).device


@torch.inference_mode()
def generate_assistant_outputs(
    model: LlavaForConditionalGeneration,
    processor: Any,
    image_paths: Sequence[str | Path],
    *,
    prompt_text: str,
    max_new_tokens: int,
) -> list[dict[str, str]]:
    """Generate and return Assistant-only text for the inference images."""
    from PIL import Image

    model.eval()
    prompt = make_prompt(processor, prompt_text)
    device = _model_input_device(model)
    results: list[dict[str, str]] = []
    for image_path_value in image_paths:
        image_path = Path(image_path_value).expanduser()
        with Image.open(image_path) as image:
            inputs = processor(
                images=image.convert("RGB"), text=prompt, return_tensors="pt"
            ).to(device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        prompt_length = inputs["input_ids"].shape[1]
        assistant_ids = generated_ids[:, prompt_length:]
        assistant_text = processor.batch_decode(
            assistant_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()
        result = {"image": str(image_path), "assistant": assistant_text}
        results.append(result)
        print(f"\n===== Best-model inference: {image_path} =====")
        print(f"Assistant: {assistant_text}")
        del inputs, generated_ids, assistant_ids
    return results


def collect_calibration_data(
    model: LlavaForConditionalGeneration,
    processor: Any,
    samples: Iterable[dict[str, Any]],
    *,
    prompt: str,
    max_samples_per_layer: int,
    seed: int,
    layer_names: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Run samples one at a time and collect bounded, CPU-resident activations."""
    model.eval()
    activations: dict[str, torch.Tensor] = {}
    sampling_generator = torch.Generator().manual_seed(seed)

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
            if not inputs or inputs[0] is None:
                return
            value = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu()
            if name in activations:
                value = torch.cat((activations[name], value), dim=0)
            if len(value) > max_samples_per_layer:
                indices = torch.randperm(
                    len(value), generator=sampling_generator
                )[:max_samples_per_layer]
                value = value[indices]
            activations[name] = value

        return hook

    handles = [
        module.register_forward_hook(make_hook(name))
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and (layer_names is None or name in layer_names)
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

    result: dict[str, torch.Tensor] = {}
    for name, values in activations.items():
        result[name] = values.contiguous()
        print(f"Kept {len(values)} calibration vectors for {name}")
    return result


def split_calibration_data(
    data: dict[str, torch.Tensor], validation_fraction: float, seed: int
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Create deterministic per-layer train/validation splits."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    generator = torch.Generator().manual_seed(seed)
    train, validation = {}, {}
    for name, values in data.items():
        if len(values) < 2:
            train[name] = validation[name] = values
            continue
        order = torch.randperm(len(values), generator=generator)
        validation_size = max(1, round(len(values) * validation_fraction))
        validation_size = min(validation_size, len(values) - 1)
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
    """Load Flickr30k samples and collect calibration activations.

    Model and dataset locations intentionally follow the original
    ``load_calib_data`` implementation and are configured by module constants.
    """
    try:
        from datasets import load_dataset
        from transformers import AutoProcessor, LlavaForConditionalGeneration
    except ImportError as error:
        raise RuntimeError(
            "Calibration loading requires the 'datasets' and 'transformers' packages"
        ) from error

    processor = AutoProcessor.from_pretrained(CHECKPOINT)
    model = LlavaForConditionalGeneration.from_pretrained(
        CHECKPOINT,
        device_map=device_map,
        torch_dtype=torch_dtype,
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
    prompt = make_prompt(processor, prompt_text)
    samples = ({"image": item[image_column]} for item in dataset)
    calibration_data = collect_calibration_data(
        model,
        processor,
        samples,
        prompt=prompt,
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


def calibrate_weight_rotation(
    quantizer: SureQuantizer,
    weight: torch.Tensor,
    cfg: SureQuantConfig,
) -> list[dict[str, float | int]]:
    """Calibrate a weight rotation with bounded row mini-batches."""
    device = weight.device
    quantizer.to(device).train()
    optimizer = torch.optim.AdamW(
        quantizer.rotation.parameters(), lr=cfg.calibration_lr
    )
    values = weight.detach().T.contiguous()
    logs: list[dict[str, float | int]] = []
    for step in range(cfg.calibration_steps):
        if len(values) > cfg.calibration_batch_size:
            indices = torch.randperm(len(values), device=device)[
                : cfg.calibration_batch_size
            ]
            batch = values[indices]
        else:
            batch = values
        output = quantizer(batch)
        loss = reconstruction_loss(output["x_blk"], output["x_hat_blk"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logs.append({"step": step, "loss_rec": float(loss.detach())})
        del output, loss, batch
    quantizer.eval()
    return logs


def calibrate_all_quantizers(
    model: LlavaForConditionalGeneration,
    calibration_data: dict[str, torch.Tensor],
    cfg: SureQuantConfig,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Calibrate activation and weight quantizers, then bake quantized weights."""
    # Lazy import keeps CLI inspection and helper tests independent from the
    # optional multimodal stack imported by train/__init__.py.
    from train.calibrate_rotations import calibrate_rotation

    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, SureQuantLinear)
    ]
    logs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for index, (name, module) in enumerate(modules, start=1):
        print(f"Calibrating {name} ({index}/{len(modules)})")
        device = module.linear.weight.device
        layer_logs: dict[str, list[dict[str, Any]]] = {}
        if name in calibration_data:
            values = calibration_data[name]
            if values.shape[-1] != module.activation_quantizer.dim:
                raise ValueError(
                    f"{name}: activation dimension {values.shape[-1]} does not "
                    f"match {module.activation_quantizer.dim}"
                )
            layer_logs["activation"] = calibrate_rotation(
                module.activation_quantizer.to(device), values.to(device), cfg
            )
        else:
            print(f"No activation data for {name}; skipping activation calibration")

        if module.weight_quantizer is not None:
            layer_logs["weight"] = calibrate_rotation(
                module.weight_quantizer.to(device), module.linear.weight.detach().T.contiguous(), cfg
            )
            module.quantize_weight()
        logs[name] = layer_logs
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return logs


@torch.inference_mode()
def reconstruction_score(
    model: LlavaForConditionalGeneration,
    validation_data: dict[str, torch.Tensor],
    *,
    batch_size: int,
) -> tuple[float, dict[str, float]]:
    """Return mean held-out per-layer MSE and the individual layer scores."""
    layer_scores: dict[str, float] = {}
    for name, module in model.named_modules():
        if not isinstance(module, SureQuantLinear) or name not in validation_data:
            continue
        values = validation_data[name]
        squared_error = 0.0
        element_count = 0
        device = module.linear.weight.device
        for start in range(0, len(values), batch_size):
            batch = values[start : start + batch_size].to(device)
            output = module.activation_quantizer(batch)
            difference = output["x_blk"] - output["x_hat_blk"]
            squared_error += float(difference.float().square().sum())
            element_count += difference.numel()
        if element_count:
            layer_scores[name] = squared_error / element_count
    if not layer_scores:
        raise RuntimeError("No quantized layer matched the validation activations")
    return sum(layer_scores.values()) / len(layer_scores), layer_scores


def loss_grid(
    base_cfg: SureQuantConfig,
    grid_values: dict[str, Sequence[float]],
) -> list[SureQuantConfig]:
    """Expand and validate the loss-weight Cartesian product."""
    unknown = set(grid_values) - set(LOSS_GRID_KEYS)
    if unknown:
        raise ValueError(f"Unsupported loss grid keys: {sorted(unknown)}")
    for name, values in grid_values.items():
        if not values:
            raise ValueError(f"Grid for {name} cannot be empty")
        if any(value < 0 for value in values):
            raise ValueError(f"Grid for {name} contains a negative value")
    keys = [key for key in LOSS_GRID_KEYS if key in grid_values]
    return [
        replace(base_cfg, **dict(zip(keys, values)))
        for values in itertools.product(*(grid_values[key] for key in keys))
    ]


def _jsonable_config(cfg: SureQuantConfig) -> dict[str, Any]:
    value = asdict(cfg)
    value["target_types"] = list(value["target_types"])
    return value


def save_quantized_model(
    model: LlavaForConditionalGeneration,
    processor: Any,
    output_dir: Path,
    metadata: dict[str, Any],
    *,
    max_shard_size: str,
) -> None:
    """Save custom-module weights plus enough metadata to rebuild wrappers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # PyTorch shards preserve this custom module state without requiring a
    # safetensors shared-storage workaround.
    model.save_pretrained(
        output_dir, safe_serialization=False, max_shard_size=max_shard_size
    )
    processor.save_pretrained(output_dir)
    (output_dir / "surequant_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_quantized_model(
    output_dir: str | Path,
    *,
    device_map: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
) -> LlavaForConditionalGeneration:
    """Rebuild SureQuant wrappers and load a checkpoint saved by this script.

    ``LlavaForConditionalGeneration.from_pretrained(output_dir)`` alone cannot
    reconstruct custom ``SureQuantLinear`` modules, so loading is deliberately
    kept alongside the saving implementation.
    """
    from transformers import LlavaForConditionalGeneration
    from transformers.modeling_utils import load_sharded_checkpoint

    output_dir = Path(output_dir)
    metadata = json.loads(
        (output_dir / "surequant_config.json").read_text(encoding="utf-8")
    )
    quant_cfg = metadata["surequant"]
    model_cfg = metadata["model_quantization"]
    model = LlavaForConditionalGeneration.from_pretrained(
        metadata["base_checkpoint"],
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    quantize_llava_model(
        model,
        num_bits=quant_cfg["num_bits"],
        block_size=quant_cfg["block_size"],
        rotation_strategy=model_cfg["rotation_strategy"],
        quantize_vision=model_cfg["quantize_vision"],
        quantize_mm_proj=model_cfg["quantize_mm_proj"],
        quantize_language=model_cfg["quantize_language"],
        quantize_weight=model_cfg["quantize_weight"],
    )
    index_path = output_dir / "pytorch_model.bin.index.json"
    if index_path.exists():
        load_sharded_checkpoint(model, output_dir, strict=True)
    else:
        state_dict = torch.load(
            output_dir / "pytorch_model.bin", map_location="cpu", weights_only=True
        )
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def run_grid_search(args: argparse.Namespace) -> dict[str, Any]:
    """Collect activations, run all trials, and persist each new best model."""
    positive_values = {
        "calibration_samples": args.calibration_samples,
        "max_vectors_per_layer": args.max_vectors_per_layer,
        "calibration_steps": args.calibration_steps,
        "calibration_batch_size": args.calibration_batch_size,
        "evaluation_batch_size": args.evaluation_batch_size,
        "dk_sample_size": args.dk_sample_size,
        "inference_max_new_tokens": args.inference_max_new_tokens,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(f"These arguments must be positive: {', '.join(invalid)}")
    if not (
        args.quantize_vision or args.quantize_mm_proj or args.quantize_language
    ):
        raise ValueError("At least one model component must be selected for quantization")

    try:
        from transformers import LlavaForConditionalGeneration
    except ImportError as error:
        raise RuntimeError("Grid search requires the 'transformers' package") from error

    seed_everything(args.seed)
    processor, all_data = load_calib_data(
        calibration_sample_num=args.calibration_samples,
        max_samples_per_layer=args.max_vectors_per_layer,
        prompt_text=args.prompt,
        image_column=args.image_column,
        block_size=args.block_size,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        seed=args.seed,
        quantize_vision=args.quantize_vision,
        quantize_mm_proj=args.quantize_mm_proj,
        quantize_language=args.quantize_language,
    )
    train_data, validation_data = split_calibration_data(
        all_data, args.validation_fraction, args.seed
    )
    del all_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    base_cfg = SureQuantConfig(
        num_bits=args.num_bits,
        block_size=args.block_size,
        calibration_steps=args.calibration_steps,
        calibration_lr=args.calibration_lr,
        calibration_batch_size=args.calibration_batch_size,
        dk_sample_size=args.dk_sample_size,
        device=args.device_map,
    )
    trials = loss_grid(
        base_cfg,
        {
            "lambda_rec": args.lambda_rec_grid,
            "lambda_dk": args.lambda_dk_grid,
            "lambda_bal": args.lambda_bal_grid,
            "lambda_range": args.lambda_range_grid,
        },
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_dir / "best_quantized_model"
    inference_path = output_dir / "best_model_inference.json"
    results: list[dict[str, Any]] = []
    best_score = float("inf")
    best_trial = -1
    best_assistant_outputs: list[dict[str, str]] = []

    for trial_index, cfg in enumerate(trials, start=1):
        print(
            f"\n===== Trial {trial_index}/{len(trials)}: "
            + ", ".join(f"{key}={getattr(cfg, key)}" for key in LOSS_GRID_KEYS)
            + " ====="
        )
        seed_everything(args.seed)
        model = LlavaForConditionalGeneration.from_pretrained(
            CHECKPOINT,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
        )
        quantize_llava_model(
            model,
            num_bits=cfg.num_bits,
            block_size=cfg.block_size,
            rotation_strategy=args.rotation_strategy,
            quantize_vision=args.quantize_vision,
            quantize_mm_proj=args.quantize_mm_proj,
            quantize_language=args.quantize_language,
            quantize_weight=args.quantize_weight,
        )
        calibration_logs = calibrate_all_quantizers(model, train_data, cfg)
        score, layer_scores = reconstruction_score(
            model,
            validation_data,
            batch_size=args.evaluation_batch_size,
        )
        result = {
            "trial": trial_index,
            "loss_weights": {key: getattr(cfg, key) for key in LOSS_GRID_KEYS},
            "validation_reconstruction_mse": score,
            "layer_validation_mse": layer_scores,
            "final_training_losses": {
                name: {
                    kind: history[-1] if history else None
                    for kind, history in layer_logs.items()
                }
                for name, layer_logs in calibration_logs.items()
            },
        }
        results.append(result)
        print(f"Trial {trial_index} validation reconstruction MSE: {score:.8g}")

        if score < best_score:
            best_score = score
            best_trial = trial_index
            metadata = {
                "format_version": 1,
                "base_checkpoint": CHECKPOINT,
                "selection_metric": "mean_layer_validation_reconstruction_mse",
                "best_trial": best_trial,
                "best_score": best_score,
                "assistant_outputs_file": str(Path("..") / inference_path.name),
                "surequant": _jsonable_config(cfg),
                "model_quantization": {
                    "rotation_strategy": args.rotation_strategy,
                    "quantize_vision": args.quantize_vision,
                    "quantize_mm_proj": args.quantize_mm_proj,
                    "quantize_language": args.quantize_language,
                    "quantize_weight": args.quantize_weight,
                },
            }
            print(f"New best trial; saving quantized model to {best_model_dir}")
            save_quantized_model(
                model,
                processor,
                best_model_dir,
                metadata,
                max_shard_size=args.max_shard_size,
            )
            best_assistant_outputs = generate_assistant_outputs(
                model,
                processor,
                args.test_images,
                prompt_text=args.inference_prompt,
                max_new_tokens=args.inference_max_new_tokens,
            )
            inference_path.write_text(
                json.dumps(
                    {
                        "best_trial": best_trial,
                        "best_score": best_score,
                        "prompt": args.inference_prompt,
                        "max_new_tokens": args.inference_max_new_tokens,
                        "outputs": best_assistant_outputs,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        # Delete the binding in this scope before the next from_pretrained;
        # passing it to a helper would leave this reference (and GPU memory)
        # alive until the following assignment finishes.
        del model, calibration_logs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "best_trial": best_trial,
        "best_score": best_score,
        "selection_metric": "mean_layer_validation_reconstruction_mse",
        "best_quantized_model_dir": str(best_model_dir),
        "best_model_inference_file": str(inference_path),
        "best_assistant_outputs": best_assistant_outputs,
        "trials": results,
    }
    (output_dir / "grid_search_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def parse_float_grid(value: str) -> list[float]:
    try:
        result = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid float grid: {value}") from error
    if not result:
        raise argparse.ArgumentTypeError("A grid must contain at least one value")
    return result


def parse_dtype(value: str) -> torch.dtype:
    choices = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if value not in choices:
        raise argparse.ArgumentTypeError(f"dtype must be one of {sorted(choices)}")
    return choices[value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=False, default=f"{REPO_ROOT}/runs")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--inference-prompt", default=DEFAULT_INFERENCE_PROMPT)
    parser.add_argument(
        "--test-images",
        nargs=2,
        default=list(DEFAULT_INFERENCE_IMAGES),
        metavar=("IMAGE_1", "IMAGE_2"),
        help="Two images used to retain the original post-calibration inference test.",
    )
    parser.add_argument("--inference-max-new-tokens", type=int, default=128)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--max-vectors-per-layer", type=int, default=512)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--num-bits", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--rotation-strategy", choices=("rotation",), default="rotation")
    parser.add_argument("--calibration-steps", type=int, default=100)
    parser.add_argument("--calibration-lr", type=float, default=0.005)
    parser.add_argument("--calibration-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--dk-sample-size", type=int, default=128)
    parser.add_argument("--lambda-rec-grid", type=parse_float_grid, default=[1.0])
    parser.add_argument(
        "--lambda-dk-grid", type=parse_float_grid, default=[0.0, 0.01, 0.05]
    )
    parser.add_argument(
        "--lambda-bal-grid", type=parse_float_grid, default=[0.0, 0.01]
    )
    parser.add_argument(
        "--lambda-range-grid", type=parse_float_grid, default=[0.0, 0.01]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--torch-dtype", type=parse_dtype, default=torch.float16)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument(
        "--quantize-vision", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--quantize-mm-proj", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--quantize-language", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--quantize-weight", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    start = time.time()
    summary = run_grid_search(args)
    print(
        f"Best trial: {summary['best_trial']}; "
        f"validation MSE: {summary['best_score']:.8g}; "
        f"elapsed: {time.time() - start:.2f}s"
    )


if __name__ == "__main__":
    main()

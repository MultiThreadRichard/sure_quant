from __future__ import annotations

import argparse
import copy
import gc
import json
import random
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from config.default_config import SureQuantConfig
from scripts.llava_wa.calibration import calibrate_all_quantizers, reconstruction_score
from scripts.llava_wa.config import CHECKPOINT, LOSS_GRID_KEYS, SEARCH_GRID_KEYS, loss_grid
from scripts.llava_wa.data import (
    generate_assistant_outputs,
    load_calib_data,
    split_calibration_data,
)
from scripts.llava_wa.modeling import quantize_llava_model
from scripts.llava_wa.persistence import _jsonable_config, save_quantized_model


# Built-in copy of the current best grid-search parameters. This keeps best-mode
# usable when a metadata path is not supplied, while --best-trial-config remains
# the source of truth for future searches.
BEST_TRIAL_FALLBACK: dict[str, dict[str, Any]] = {
    "surequant": {
        "block_size": 128,
        "num_bits": 4,
        "clip_ratio": 0.9,
        "activation_scale_granularity": "per_vector_block",
        "weight_scale_granularity": "per_vector_block",
        "givens_pairs_strategy": "butterfly",
        "num_givens_layers": 2,
        "num_pairs_per_layer": 8,
        "order": "hadamard_givens",
        "lambda_rec": 1.0,
        "lambda_dk": 0.05,
        "lambda_bal": 0.0,
        "lambda_range": 0.01,
        "lambda_orth": 0.0,
        "calibration_steps": 100,
        "calibration_lr": 0.005,
        "calibration_batch_size": 128,
        "dk_sample_size": 1024,
        "scale_mode": "clipped_absmax",
        "target_types": ["weight", "activation"],
        "device": "cuda",
        "dtype": "float32",
    },
    "model_quantization": {
        "rotation_strategy": "rotation",
        "quantize_vision": True,
        "quantize_mm_proj": True,
        "quantize_language": True,
        "quantize_weight": True,
    },
}


def _validate_best_trial_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Best-trial config must be a JSON object")
    for section in ("surequant", "model_quantization"):
        if not isinstance(payload.get(section), dict):
            raise ValueError(f"Best-trial config requires an object section: {section}")
    return payload


def load_best_trial_config(path: str | Path | None) -> dict[str, Any]:
    """Load best-trial metadata, or return an isolated fallback copy."""
    if path is None:
        return copy.deepcopy(BEST_TRIAL_FALLBACK)
    config_path = Path(path).expanduser()
    with config_path.open(encoding="utf-8") as config_file:
        payload = json.load(config_file)
    return _validate_best_trial_payload(payload)


def build_cfg_and_scope_from_best_trial(
    payload: dict[str, Any],
) -> tuple[SureQuantConfig, dict[str, Any], str]:
    """Map persisted best-trial metadata to calibration and model settings."""
    payload = _validate_best_trial_payload(payload)
    config_fields = {item.name for item in fields(SureQuantConfig)}
    config_values = {
        key: value
        for key, value in payload["surequant"].items()
        if key in config_fields
    }
    if "target_types" in config_values:
        config_values["target_types"] = tuple(config_values["target_types"])
    cfg = SureQuantConfig(**config_values)

    model_config = payload["model_quantization"]
    scope = {
        key: bool(model_config.get(key, True))
        for key in (
            "quantize_vision",
            "quantize_mm_proj",
            "quantize_language",
            "quantize_weight",
        )
    }
    scope.update(
        {
            "clip_ratio": cfg.clip_ratio,
            "activation_scale_granularity": cfg.activation_scale_granularity,
            "weight_scale_granularity": cfg.weight_scale_granularity,
        }
    )
    rotation_strategy = str(model_config.get("rotation_strategy", "rotation"))
    return cfg, scope, rotation_strategy


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_args(
    args: argparse.Namespace, quantization_scope: dict[str, Any] | None = None
) -> None:
    positive_values = {
        "calibration_samples": args.calibration_samples,
        "max_vectors_per_layer": args.max_vectors_per_layer,
        "calibration_batch_size": args.calibration_batch_size,
        "evaluation_batch_size": args.evaluation_batch_size,
        "dk_sample_size": args.dk_sample_size,
        "inference_max_new_tokens": args.inference_max_new_tokens,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(f"These arguments must be positive: {', '.join(invalid)}")
    scope = quantization_scope or vars(args)
    if not any(
        scope[key]
        for key in ("quantize_vision", "quantize_mm_proj", "quantize_language")
    ):
        raise ValueError("At least one model component must be selected for quantization")


def _trial_configs(args: argparse.Namespace) -> list[SureQuantConfig]:
    base_cfg = SureQuantConfig(
        num_bits=args.num_bits,
        block_size=args.block_size,
        clip_ratio=args.clip_ratio_grid[0],
        calibration_steps=args.calibration_steps[0],
        calibration_lr=args.calibration_lr[0],
        calibration_batch_size=args.calibration_batch_size,
        dk_sample_size=args.dk_sample_size,
        device=args.device_map,
    )
    return loss_grid(
        base_cfg,
        {
            "calibration_steps": args.calibration_steps,
            "calibration_lr": args.calibration_lr,
            "clip_ratio": args.clip_ratio_grid,
            "lambda_rec": args.lambda_rec_grid,
            "lambda_dk": args.lambda_dk_grid,
            "lambda_bal": args.lambda_bal_grid,
            "lambda_range": args.lambda_range_grid,
        },
    )


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_grid_search(args: argparse.Namespace) -> dict[str, Any]:
    """Collect activations, evaluate all trials, and persist the best model."""
    _validate_args(args)
    trials = _trial_configs(args)
    try:
        from transformers import LlavaForConditionalGeneration
    except ImportError as error:
        raise RuntimeError("Grid search requires the transformers package") from error

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
    _release_cuda_memory()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_dir / "best_quantized_model"
    inference_path = output_dir / "best_model_inference.json"
    results: list[dict[str, Any]] = []
    best_score = float("inf")
    best_trial = -1
    best_assistant_outputs: list[dict[str, str]] = []

    for trial_index, cfg in enumerate(trials, start=1):
        parameters = ", ".join(f"{key}={getattr(cfg, key)}" for key in SEARCH_GRID_KEYS)
        print(f"\n===== Trial {trial_index}/{len(trials)}: {parameters} =====")
        seed_everything(args.seed)
        model = LlavaForConditionalGeneration.from_pretrained(
            CHECKPOINT, device_map=args.device_map, torch_dtype=args.torch_dtype
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
            clip_ratio=cfg.clip_ratio,
            activation_scale_granularity=cfg.activation_scale_granularity,
            weight_scale_granularity=cfg.weight_scale_granularity,
        )
        calibration_logs = calibrate_all_quantizers(model, train_data, cfg)
        score, layer_scores = reconstruction_score(
            model, validation_data, batch_size=args.evaluation_batch_size
        )
        result = {
            "trial": trial_index,
            "search_parameters": {key: getattr(cfg, key) for key in SEARCH_GRID_KEYS},
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
            best_score, best_trial = score, trial_index
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
                model, processor, best_model_dir, metadata, max_shard_size=args.max_shard_size
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
        del model, calibration_logs
        _release_cuda_memory()

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


def run_best_trial_calibration(args: argparse.Namespace) -> dict[str, Any]:
    """Recalibrate once with persisted best-trial parameters and save the model."""
    print(f"Loading best trial config from {args.best_trial_config}")
    payload = load_best_trial_config(args.best_trial_config)
    print(f"Loaded best trial config: {payload}")
    cfg, quantization_scope, rotation_strategy = build_cfg_and_scope_from_best_trial(
        payload
    )
    _validate_args(args, quantization_scope)
    try:
        from transformers import LlavaForConditionalGeneration
    except ImportError as error:
        raise RuntimeError(
            "Best-trial calibration requires the transformers package"
        ) from error

    source = (
        str(Path(args.best_trial_config).expanduser())
        if args.best_trial_config is not None
        else "BEST_TRIAL_FALLBACK"
    )
    checkpoint = str(payload.get("base_checkpoint", CHECKPOINT))
    seed_everything(args.seed)
    processor, all_data = load_calib_data(
        calibration_sample_num=args.calibration_samples,
        max_samples_per_layer=args.max_vectors_per_layer,
        prompt_text=args.prompt,
        image_column=args.image_column,
        block_size=cfg.block_size,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        seed=args.seed,
        quantize_vision=quantization_scope["quantize_vision"],
        quantize_mm_proj=quantization_scope["quantize_mm_proj"],
        quantize_language=quantization_scope["quantize_language"],
    )
    train_data, validation_data = split_calibration_data(
        all_data, args.validation_fraction, args.seed
    )
    del all_data
    _release_cuda_memory()

    model = LlavaForConditionalGeneration.from_pretrained(
        checkpoint, device_map=args.device_map, torch_dtype=args.torch_dtype
    )
    quantize_llava_model(
        model,
        num_bits=cfg.num_bits,
        block_size=cfg.block_size,
        rotation_strategy=rotation_strategy,
        **quantization_scope,
    )
    calibration_logs = calibrate_all_quantizers(model, train_data, cfg)
    score, layer_scores = reconstruction_score(
        model, validation_data, batch_size=args.evaluation_batch_size
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_dir / "best_quantized_model"
    inference_path = output_dir / "best_model_inference.json"
    source_trial = payload.get("best_trial")
    metadata = {
        "format_version": 1,
        "base_checkpoint": checkpoint,
        "selection_metric": "mean_layer_validation_reconstruction_mse",
        "best_trial": source_trial if source_trial is not None else 1,
        "best_score": score,
        "assistant_outputs_file": str(Path("..") / inference_path.name),
        "surequant": _jsonable_config(cfg),
        "model_quantization": {
            "rotation_strategy": rotation_strategy,
            **{
                key: quantization_scope[key]
                for key in (
                    "quantize_vision",
                    "quantize_mm_proj",
                    "quantize_language",
                    "quantize_weight",
                )
            },
        },
        "calibration": {
            "mode": "best_trial",
            "source": source,
            "source_best_trial": payload.get("best_trial"),
            "source_best_score": payload.get("best_score"),
        },
    }
    save_quantized_model(
        model, processor, best_model_dir, metadata, max_shard_size=args.max_shard_size
    )
    assistant_outputs = generate_assistant_outputs(
        model,
        processor,
        args.test_images,
        prompt_text=args.inference_prompt,
        max_new_tokens=args.inference_max_new_tokens,
    )
    inference_path.write_text(
        json.dumps(
            {
                "best_trial": 1,
                "best_score": score,
                "prompt": args.inference_prompt,
                "max_new_tokens": args.inference_max_new_tokens,
                "outputs": assistant_outputs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    summary = {
        "best_trial": 1,
        "best_score": score,
        "selection_metric": "mean_layer_validation_reconstruction_mse",
        "best_quantized_model_dir": str(best_model_dir),
        "best_model_inference_file": str(inference_path),
        "best_assistant_outputs": assistant_outputs,
        "layer_validation_mse": layer_scores,
        "final_training_losses": {
            name: {
                kind: history[-1] if history else None
                for kind, history in layer_logs.items()
            }
            for name, layer_logs in calibration_logs.items()
        },
        "calibration_source": source,
    }
    (output_dir / "best_trial_calibration_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    del model, calibration_logs
    _release_cuda_memory()
    return summary

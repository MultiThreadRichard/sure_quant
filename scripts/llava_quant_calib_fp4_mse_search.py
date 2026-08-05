"""LLaVA FP4 Quantization Grid Search.

EVAL SCORE: mse
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import argparse
import gc
import json
import random
from typing import Any

import torch

from config.default_config import SureQuantConfig
from scripts.llava_wa.calibration import calibrate_all_quantizers, reconstruction_score
from scripts.llava_wa.config import (
    CHECKPOINT,
    LOSS_GRID_KEYS,
    SEARCH_GRID_KEYS,
    build_parser,
    loss_grid,
)
from scripts.llava_wa.data import (
    generate_assistant_outputs,
    load_calib_data,
    split_calibration_data,
)
from scripts.llava_wa.persistence import _jsonable_config
from scripts.llava_wa.modeling_fp4 import (
    quantize_llava_model_fp4,
    save_quantized_model_fp4,
)




def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_args(args: argparse.Namespace) -> None:
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
    if not (args.quantize_vision or args.quantize_mm_proj or args.quantize_language):
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


def run_grid_search_fp4(args: argparse.Namespace) -> dict[str, Any]:
    """Collect activations, evaluate all FP4 trials, and persist the best model."""
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
        parameters = ", ".join(
            f"{key}={getattr(cfg, key)}" for key in SEARCH_GRID_KEYS
        )
        print(f"\n===== FP4 Trial {trial_index}/{len(trials)}: {parameters} =====")
        seed_everything(args.seed)
        model = LlavaForConditionalGeneration.from_pretrained(
            CHECKPOINT, device_map=args.device_map, torch_dtype=args.torch_dtype
        )
        quantize_llava_model_fp4(
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
            model,
            validation_data,
            batch_size=args.evaluation_batch_size,
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
                    "activation_scale_granularity": cfg.activation_scale_granularity,
                    "weight_scale_granularity": cfg.weight_scale_granularity,
                    "quant_type": "fp4_e2m1",
                },
            }
            print(f"New best trial; saving FP4 quantized model to {best_model_dir}")
            save_quantized_model_fp4(
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

        # TODO
        trial_outputs = generate_assistant_outputs(
            model,
            processor,
            args.test_images,
            prompt_text=args.inference_prompt,
            max_new_tokens=args.inference_max_new_tokens,
        )

        trial_path = output_dir / f"trial_{trial_index:02d}.json"
        trial_path.write_text(
            json.dumps(
                {
                    "trial": trial_index,
                    "score": score,
                    "prompt": args.inference_prompt,
                    "max_new_tokens": args.inference_max_new_tokens,
                    "outputs": trial_outputs,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        if trial_index > 20:
            break

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
    # (output_dir / "grid_search_results.json").write_text(
    #     json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    # )
    return summary


def main() -> None:
    args = build_parser().parse_args()
    start = time.time()
    summary = run_grid_search_fp4(args)
    print(
        f"Best FP4 trial: {summary['best_trial']}; "
        f"validation MSE: {summary['best_score']:.8g}; "
        f"elapsed: {time.time() - start:.2f}s"
    )


if __name__ == "__main__":
    main()
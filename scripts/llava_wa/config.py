from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import torch

from config.default_config import SureQuantConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
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
SEARCH_GRID_KEYS = (
    "calibration_steps",
    "calibration_lr",
    "clip_ratio",
    *LOSS_GRID_KEYS,
)
POSITIVE_GRID_KEYS = {"calibration_steps", "calibration_lr", "clip_ratio"}


def loss_grid(
    base_cfg: SureQuantConfig,
    grid_values: dict[str, Sequence[int | float]],
) -> list[SureQuantConfig]:
    """Expand calibration hyperparameters into independent configurations."""
    unknown = set(grid_values) - set(SEARCH_GRID_KEYS)
    if unknown:
        raise ValueError(f"Unsupported loss grid keys: {sorted(unknown)}")

    for key, values in grid_values.items():
        if not values:
            raise ValueError(f"Grid for {key} cannot be empty")
        if key in POSITIVE_GRID_KEYS and any(value <= 0 for value in values):
            raise ValueError(f"Grid for {key} must contain only positive values")
        if key == "clip_ratio" and any(value > 1 for value in values):
            raise ValueError("Grid for clip_ratio must contain values no greater than 1")
        if key in LOSS_GRID_KEYS and any(value < 0 for value in values):
            raise ValueError(f"Grid for {key} must contain only non-negative values")

    keys = [key for key in SEARCH_GRID_KEYS if key in grid_values]
    return [
        replace(base_cfg, **dict(zip(keys, values)))
        for values in itertools.product(*(grid_values[key] for key in keys))
    ]


def parse_float_grid(value: str) -> list[float]:
    try:
        values = [float(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated numbers") from exc
    if not values:
        raise argparse.ArgumentTypeError("Grid cannot be empty")
    return values


def parse_int_grid(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("Grid cannot be empty")
    return values


def parse_dtype(value: str) -> torch.dtype:
    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    try:
        return dtypes[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"Unsupported dtype: {value}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate, tune, and save a SureQuant-quantized LLaVA model."
    )
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--inference-prompt", default=DEFAULT_INFERENCE_PROMPT)
    parser.add_argument(
        "--test-images",
        nargs=2,
        default=list(DEFAULT_INFERENCE_IMAGES),
        metavar=("IMAGE_1", "IMAGE_2"),
    )
    parser.add_argument("--inference-max-new-tokens", type=int, default=128)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--max-vectors-per-layer", type=int, default=512)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--num-bits", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--clip-ratio-grid",
        type=parse_float_grid,
        default=[0.9, 1.0],
        help="Absmax clipping ratios searched for both activation and weight INT4.",
    )
    parser.add_argument("--rotation-strategy", choices=("rotation",), default="rotation")
    parser.add_argument("--calibration-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-dtype", type=parse_dtype, default=torch.float16)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--dk-sample-size", type=int, default=1024)
    parser.add_argument("--calibration-steps", type=parse_int_grid, default=[20, 50, 100])
    parser.add_argument(
        "--calibration-lr",
        type=parse_float_grid,
        default=[0.0001, 0.0005, 0.001, 0.005],
    )
    parser.add_argument("--lambda-rec-grid", type=parse_float_grid, default=[1.0])
    parser.add_argument("--lambda-dk-grid", type=parse_float_grid, default=[0.0, 0.01, 0.05])
    parser.add_argument("--lambda-bal-grid", type=parse_float_grid, default=[0.0, 0.01])
    parser.add_argument("--lambda-range-grid", type=parse_float_grid, default=[0.0, 0.01])
    parser.add_argument("--quantize-vision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quantize-mm-proj", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quantize-language", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quantize-weight", action=argparse.BooleanOptionalAction, default=True)
    return parser

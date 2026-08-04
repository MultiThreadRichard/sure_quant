"""Compatibility facade and CLI for modular LLaVA SureQuant calibration."""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.llava_wa.calibration import (  # noqa: E402
    calibrate_all_quantizers,
)
from scripts.llava_wa.config import build_parser, loss_grid  # noqa: E402
from scripts.llava_wa.data import generate_assistant_outputs  # noqa: E402
from scripts.llava_wa.persistence import load_quantized_model  # noqa: E402
from scripts.llava_wa.search import (  # noqa: E402
    BEST_TRIAL_FALLBACK,
    build_cfg_and_scope_from_best_trial,
    run_best_trial_calibration,
    load_best_trial_config,
    run_grid_search,
)


DEFAULT_BEST_TRIAL_CONFIG = REPO_ROOT / "runs" / "best_quantized_model" / "surequant_config.json"

def grid_search() -> None:
    args = build_parser().parse_args()
    start = time.time()
    summary = run_grid_search(args)
    print(
        f"Best trial: {summary['best_trial']}; "
        f"validation MSE: {summary['best_score']:.8g}; "
        f"elapsed: {time.time() - start:.2f}s"
    )


def calibrate_with_the_best_trial() -> None:
    args = build_parser().parse_args()
    if args.mode == "grid":
        start = time.time()
        summary = run_grid_search(args)
        print(
            f"Best trial: {summary['best_trial']}; "
            f"validation MSE: {summary['best_score']:.8g}; "
            f"elapsed: {time.time() - start:.2f}s"
        )
        return

    if args.best_trial_config is None and DEFAULT_BEST_TRIAL_CONFIG.is_file():
        args.best_trial_config = str(DEFAULT_BEST_TRIAL_CONFIG)
    start = time.time()
    summary = run_best_trial_calibration(args)
    print(
        f"Best-trial validation MSE: {summary['best_score']:.8g}; "
        f"model: {summary['best_quantized_model_dir']}; "
        f"elapsed: {time.time() - start:.2f}s"
    )


def main() -> None:
    calibrate_with_the_best_trial()

if __name__ == "__main__":
    main()

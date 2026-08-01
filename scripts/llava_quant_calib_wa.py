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

from scripts.llava_wa.search import run_grid_search # noqa: E402
from scripts.llava_wa.config import build_parser  # noqa: E402
from scripts.llava_wa.data import generate_assistant_outputs # noqa: E402
from scripts.llava_wa.config import loss_grid # noqa: E402
from scripts.llava_wa.persistence import load_quantized_model # noqa: E402

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

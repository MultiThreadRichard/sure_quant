from typing import TYPE_CHECKING, Any

from .calibrate_rotations import calibrate_rotation
from .calibrate_stiefel import calibrate_stiefel
from .stiefel_optimizer import StiefelOptimizer

if TYPE_CHECKING:
    from .high_level_api import SureQuantCalibrator

__all__ = [
    "calibrate_rotation",
    "calibrate_stiefel",
    "SureQuantCalibrator",
    "StiefelOptimizer",
]


def __getattr__(name: str) -> Any:
    if name == "SureQuantCalibrator":
        from .high_level_api import SureQuantCalibrator

        return SureQuantCalibrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

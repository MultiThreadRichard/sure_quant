from typing import TYPE_CHECKING, Any

from .export_rotation_params import export_sure_quantizer
from .checkpoint_io import load_sure_quantizer

if TYPE_CHECKING:
    from .utils import load_quantized_model, save_quantized_model

__all__ = ["export_sure_quantizer", "load_sure_quantizer", "save_quantized_model", "load_quantized_model"]


def __getattr__(name: str) -> Any:
    if name in {"save_quantized_model", "load_quantized_model"}:
        from . import utils

        return getattr(utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

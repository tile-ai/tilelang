"""Hardware models and tuning configuration data without TVM Schedule or TE.

Schedule-based template and policy generation is retired. Use explicit candidate
configurations with :mod:`tilelang.autotuner` to tune native TileLang kernels.
"""

from .arch import CDNA, CUDA, RDNA
from .roller import Hint, NoRasterization, Rasterization2DColumn, Rasterization2DRow

__all__ = ["CDNA", "CUDA", "RDNA", "Hint", "NoRasterization", "Rasterization2DColumn", "Rasterization2DRow"]


def __getattr__(name):
    if name.endswith("Template") or name in {"DefaultPolicy", "TensorCorePolicy", "PrimFuncNode", "OutputNode", "Edge"}:
        raise AttributeError(
            f"{name} belongs to the retired Schedule/TE-based Carver API. "
            "Use tilelang.autotuner with native TileLang kernels and explicit configurations. "
            "See MIGRATION_TIRX.md."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

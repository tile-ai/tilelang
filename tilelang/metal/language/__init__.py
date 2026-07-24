"""Metal language dialect: common TileLang plus Metal extensions."""

from __future__ import annotations

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL
from tilelang.language.builtin import (  # noqa: F401
    cooperative_tensor_fill,
    cooperative_tensor_load,
    cooperative_tensor_multiply_accumulate,
    cooperative_tensor_store,
)

from .tir import *  # noqa: F401,F403
from .tir import __all__ as _TIR_ALL

__tilelang_dialect__ = "metal"
__all__ = tuple(
    dict.fromkeys(
        (
            *_COMMON_ALL,
            *_TIR_ALL,
            "cooperative_tensor_fill",
            "cooperative_tensor_load",
            "cooperative_tensor_multiply_accumulate",
            "cooperative_tensor_store",
        )
    )
)

del _COMMON_ALL, _TIR_ALL

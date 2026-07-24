"""Metal language dialect: common TileLang plus Metal extensions."""

from __future__ import annotations

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL

from .tir import *  # noqa: F401,F403
from .tir import __all__ as _TIR_ALL

__tilelang_dialect__ = "metal"
__all__ = tuple(dict.fromkeys((*_COMMON_ALL, *_TIR_ALL)))

del _COMMON_ALL, _TIR_ALL

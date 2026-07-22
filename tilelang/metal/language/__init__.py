"""Metal language dialect: common TileLang plus Metal extensions."""

from __future__ import annotations

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL
from tilelang.metal.intrinsics import MPSIntrinEmitter as MPSIntrinEmitter

__tilelang_dialect__ = "metal"
__all__ = tuple((*_COMMON_ALL, "MPSIntrinEmitter"))

del _COMMON_ALL

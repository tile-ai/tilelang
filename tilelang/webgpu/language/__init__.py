"""WebGPU language dialect: common TileLang plus WebGPU extensions."""

from __future__ import annotations

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL

__tilelang_dialect__ = "webgpu"
__all__ = tuple(_COMMON_ALL)

del _COMMON_ALL

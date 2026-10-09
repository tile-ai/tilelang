"""CPU language dialect: common TileLang plus CPU extensions."""

from __future__ import annotations

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL

from .kernel import *  # noqa: F401,F403
from .kernel import __all__ as _KERNEL_ALL

__tilelang_dialect__ = "cpu"
__all__ = tuple(dict.fromkeys((*_COMMON_ALL, *_KERNEL_ALL)))

del _COMMON_ALL, _KERNEL_ALL

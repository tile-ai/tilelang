"""ROCm/HIP language dialect: common TileLang plus ROCm extensions."""

from __future__ import annotations

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL

from .intrinsics import *  # noqa: F401,F403
from .intrinsics import __all__ as _ROCM_ALL

__tilelang_dialect__ = "rocm"
__all__ = tuple(dict.fromkeys((*_COMMON_ALL, *_ROCM_ALL)))

del _COMMON_ALL, _ROCM_ALL

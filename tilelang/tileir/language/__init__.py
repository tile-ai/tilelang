"""TileIR language dialect: common TileLang plus CUDA Tile IR hints."""

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL
from .copy_op import copy  # noqa: F401
from .kernel import Kernel  # noqa: F401

__tilelang_dialect__ = "tileir"
__all__ = list(dict.fromkeys([*_COMMON_ALL, "Kernel", "copy"]))

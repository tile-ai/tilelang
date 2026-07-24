"""Metal language dialect: common TileLang plus Metal extensions."""

from __future__ import annotations

import importlib

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL

from .tir import *  # noqa: F401,F403
from .tir import __all__ as _TIR_ALL

__tilelang_dialect__ = "metal"
__all__ = tuple((*_COMMON_ALL, "MPSIntrinEmitter", *_TIR_ALL))

del _COMMON_ALL, _TIR_ALL


def __getattr__(name: str):
    # Deferred to break the metal.language <-> metal.intrinsics import cycle:
    # metal_macro_generator imports this facade, so eagerly importing the
    # emitter here would recurse during bootstrap.
    if name == "MPSIntrinEmitter":
        return importlib.import_module("tilelang.metal.intrinsics").MPSIntrinEmitter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

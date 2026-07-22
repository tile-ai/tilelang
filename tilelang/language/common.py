"""Backend-neutral TileLang language surface."""

from __future__ import annotations

import sys

_language = sys.modules["tilelang.language"]
__all__ = tuple(_language.__tilelang_common_all__)

__tilelang_dialect__ = "common"


def __getattr__(name: str):
    if name in __all__:
        return getattr(_language, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

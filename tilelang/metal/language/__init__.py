"""Metal language dialect overlay."""

from __future__ import annotations

from tilelang.language.dialect import export_core_language_symbols
from tilelang.metal.intrinsics import MPSIntrinEmitter as MPSIntrinEmitter

export_core_language_symbols(globals())
del export_core_language_symbols

__tilelang_dialect__ = "metal"
__all__ = sorted(name for name in globals() if not (name.startswith("__") and name.endswith("__")))

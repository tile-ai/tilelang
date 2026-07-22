"""ROCm/HIP language dialect overlay."""

from __future__ import annotations

from tilelang.language.dialect import export_core_language_symbols

export_core_language_symbols(globals())
del export_core_language_symbols

from .intrinsics import *  # noqa: E402,F401,F403

__tilelang_dialect__ = "rocm"
__all__ = sorted(name for name in globals() if not (name.startswith("__") and name.endswith("__")))

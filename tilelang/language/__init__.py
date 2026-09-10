"""Default TileLang language facade for the Ascend fork.

Upstream re-exports the CUDA dialect here. This fork re-exports the Ascend
dialect instead, so ``import tilelang.language as T`` yields the common surface
plus the Ascend extensions. Other backends are reached explicitly via
``tilelang.<backend>.language`` (which build on ``tilelang.language.common``).
"""

from __future__ import annotations

from tilelang.ascend.language import *  # noqa: F401,F403
from tilelang.ascend.language import __all__ as __all__  # noqa: F401

# Imported by name so static type checkers resolve the Ascend-typed launch
# signature through this facade (they cannot evaluate the dynamic __all__).
from tilelang.ascend.language import Kernel  # noqa: F401

__tilelang_dialect__ = "ascend"

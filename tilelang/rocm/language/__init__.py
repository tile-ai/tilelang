"""ROCm/HIP language dialect: common TileLang plus ROCm extensions."""

from __future__ import annotations

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL

from .intrinsics import *  # noqa: F401,F403
from .intrinsics import __all__ as _ROCM_ALL
from .kernel import *  # noqa: F401,F403
from .kernel import __all__ as _KERNEL_ALL

# The ROCm dialect's T.gemm shadows the common one: same semantics, plus the
# k_pack knob of the MFMA/WMMA lowering.
from .gemm_op import *  # noqa: F401,F403
from .gemm_op import __all__ as _GEMM_ALL

__tilelang_dialect__ = "rocm"
__all__ = tuple(dict.fromkeys((*_COMMON_ALL, *_ROCM_ALL, *_KERNEL_ALL, *_GEMM_ALL)))

del _COMMON_ALL, _ROCM_ALL, _KERNEL_ALL, _GEMM_ALL

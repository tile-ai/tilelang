"""Wrapping Layouts."""
# pylint: disable=invalid-name, unsupported-binary-operation

from importlib import import_module as _import_module

from .layout import Layout  # noqa: F401
from .fragment import Fragment  # noqa: F401
from .partial_fragment import (
    PartialFragment,  # noqa: F401
    make_fully_replicated_partial_fragment,  # noqa: F401
)
from .swizzle import (
    make_swizzled_layout,  # noqa: F401
    make_volta_swizzled_layout,  # noqa: F401
    make_wgmma_swizzled_layout,  # noqa: F401
    make_tcgen05mma_swizzled_layout,  # noqa: F401
    make_full_bank_swizzled_layout,  # noqa: F401
    make_half_bank_swizzled_layout,  # noqa: F401
    make_quarter_bank_swizzled_layout,  # noqa: F401
    make_linear_layout,  # noqa: F401
    make_gemm_fragment_8x8,  # noqa: F401
    make_gemm_fragment_8x8_transposed,  # noqa: F401
    make_fully_replicated_layout_fragment,  # noqa: F401
)

from .gemm_sp import make_cutlass_metadata_layout  # noqa: F401
from .swizzle_mode import SwizzleMode  # noqa: F401
from . import cute  # noqa: F401  (registers tl.cute.* FFI objects on import)

_ASCEND_EXPORTS = (
    "make_ascend_nz_layout",
    "make_ascend_compact_nz_layout",
    "make_ascend_major_k_layout",
    "make_ascend_major_mn_layout",
    "make_ascend_l0c_layout",
    "make_ascend_sf_layout",
    "make_strided_slice",
    "try_extract_fractal_layout",
    "FractalLayoutInfo",
)

__all__ = [name for name in globals() if not name.startswith("_")] + list(_ASCEND_EXPORTS)


def __getattr__(name):
    if name in _ASCEND_EXPORTS:
        value = getattr(_import_module(".ascend", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

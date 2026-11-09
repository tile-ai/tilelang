"""Ascend scalar math expressed with common TIR operations."""

from tvm import tirx

from tilelang.language.common import rsqrt as _common_rsqrt, sqrt as _common_sqrt
from tilelang.language.kernel import KernelLaunchFrame

from .frame import inside_simdvf
from .kernel import _get_current_simtvf

__all__ = ["rsqrt"]


def rsqrt(x, *, dtype=None) -> tirx.PrimExpr:
    """Compose scalar float32 reciprocal sqrt, preserving existing VF intrinsics.

    As with the common math wrappers, the input determines the result dtype.
    The optional ``dtype`` keyword is accepted for the same frontend contract.
    """
    if KernelLaunchFrame.Current() is None or _get_current_simtvf() is not None or inside_simdvf():
        return _common_rsqrt(x, dtype=dtype)

    root = _common_sqrt(x, dtype=dtype)
    if root.dtype != "float32":
        return _common_rsqrt(x, dtype=dtype)
    return tirx.const(1, root.dtype) / root

"""CUDA-specific floating-point arithmetic intrinsics."""

from __future__ import annotations

from tilelang.language import _construction
from tilelang.language import _intrinsics
import tvm_ffi as _ffi
from tvm_tirx import ir as _ir


from tvm_ffi import dtype as DataType
from tvm_ffi._dtype import DataTypeCode

from tvm_tirx.ir import Expr as PrimExpr


def _prepare_float_args(intrinsic: str, *args: PrimExpr) -> tuple[PrimExpr, ...]:
    converted = tuple(_ffi.convert(arg) for arg in args)
    dtype = _construction.dtype_of(converted[0])
    if DataType(dtype).type_code not in (DataTypeCode.FLOAT, DataTypeCode.BFLOAT):
        raise TypeError(f"T.{intrinsic} only supports floating-point inputs, but got {dtype}")
    if any(_construction.dtype_of(arg) != dtype for arg in converted[1:]):
        raise ValueError(
            f"T.{intrinsic} expects all inputs to have the same dtype, but got {[_construction.dtype_of(arg) for arg in converted]}"
        )
    return converted


def fmul(x: PrimExpr, y: PrimExpr) -> PrimExpr:
    """Multiply with CUDA round-to-nearest semantics.

    Unlike the ``*`` operator, this intrinsic preserves an explicit multiply
    boundary during CUDA lowering. Vectorized loops may lower it to packed
    multiply instructions when the target supports them.
    """
    x, y = _prepare_float_args("fmul", x, y)
    return _intrinsics.call_intrin(_construction.dtype_of(x), _ir.Op.get("tl.fmul"), x, y)


def fma(x: PrimExpr, y: PrimExpr, z: PrimExpr) -> PrimExpr:
    """Compute ``x * y + z`` as one CUDA fused multiply-add operation.

    The result uses round-to-nearest semantics. Vectorized loops may lower the
    operation to packed FMA instructions when the target supports them.
    """
    x, y, z = _prepare_float_args("fma", x, y, z)
    return _intrinsics.call_intrin(_construction.dtype_of(x), _ir.Op.get("tl.fma"), x, y, z)


__all__ = ["fma", "fmul"]

"""CUDA-specific floating-point arithmetic intrinsics."""

from __future__ import annotations

from tvm import DataType, DataTypeCode, tirx
from tvm.tirx import PrimExpr


def _prepare_float_args(intrinsic: str, *args: PrimExpr) -> tuple[PrimExpr, ...]:
    converted = tuple(tirx.convert(arg) for arg in args)
    dtype = converted[0].dtype
    if DataType(dtype).type_code not in (DataTypeCode.FLOAT, DataTypeCode.BFLOAT):
        raise TypeError(f"T.{intrinsic} only supports floating-point inputs, but got {dtype}")
    if any(arg.dtype != dtype for arg in converted[1:]):
        raise ValueError(f"T.{intrinsic} expects all inputs to have the same dtype, but got {[arg.dtype for arg in converted]}")
    return converted


def fmul(x: PrimExpr, y: PrimExpr) -> PrimExpr:
    """Multiply with CUDA round-to-nearest semantics.

    Unlike the ``*`` operator, this intrinsic preserves an explicit multiply
    boundary during CUDA lowering. Vectorized loops may lower it to packed
    multiply instructions when the target supports them.
    """
    x, y = _prepare_float_args("fmul", x, y)
    return tirx.call_intrin(x.dtype, tirx.op.Op.get("tl.fmul"), x, y)


def fma(x: PrimExpr, y: PrimExpr, z: PrimExpr) -> PrimExpr:
    """Compute ``x * y + z`` as one CUDA fused multiply-add operation.

    The result uses round-to-nearest semantics. Vectorized loops may lower the
    operation to packed FMA instructions when the target supports them.
    """
    x, y, z = _prepare_float_args("fma", x, y, z)
    return tirx.call_intrin(x.dtype, tirx.op.Op.get("tl.fma"), x, y, z)


__all__ = ["fma", "fmul"]

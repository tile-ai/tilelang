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


def clamp_symmetric(x: PrimExpr, limit: PrimExpr) -> PrimExpr:
    """Symmetric FP32 clipping using CUDA ``min.xorsign.abs.f32`` (SM86+).

    For ``limit > 0`` and a non-NaN input, this returns
    ``copysign(min(abs(x), limit), x)``, preserving the sign of zero.
    Operands must be float32; a scalar limit is broadcast to vector inputs.
    The caller supplies a positive limit for clipping semantics. No runtime
    check is emitted: each scalar lane lowers to one PTX instruction.

    This operation has PTX min-number semantics, not ``T.clamp`` semantics:
    one NaN operand yields the magnitude of the numeric operand, and two
    NaNs yield NaN. For a non-NaN result, its sign is the XOR of the input
    sign bits, including when an operand is NaN. A negative limit therefore
    clips to its absolute magnitude and flips the input sign; it does not
    represent reversed clamp bounds. Subnormals are preserved.
    Each operand is evaluated once.
    """
    x, limit = (tirx.convert(arg) for arg in (x, limit))
    x_dtype, limit_dtype = DataType(x.dtype), DataType(limit.dtype)
    if x_dtype.with_lanes(1) != DataType("float32") or limit_dtype.with_lanes(1) != DataType("float32"):
        raise TypeError("T.clamp_symmetric requires float32 operands")
    if limit_dtype.lanes == 1 and x_dtype.lanes != 1:
        limit = tirx.Broadcast(limit, x_dtype.lanes)
    elif limit_dtype.lanes != x_dtype.lanes:
        raise ValueError("T.clamp_symmetric requires matching lanes or a scalar limit")
    return tirx.call_intrin(x.dtype, tirx.op.Op.get("tl.clamp_symmetric"), x, limit)


__all__ = ["fma", "fmul", "clamp_symmetric"]

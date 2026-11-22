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
    """Symmetric clipping using CUDA ``min.NaN.xorsign.abs`` (SM86+).

    For non-NaN operands, requires ``limit > 0`` and returns
    ``copysign(min(abs(x), limit), x)``, preserving the sign of zero.
    Supports float16, bfloat16, and float32 with matching operand dtypes.
    Python numeric limits are converted to the input's element dtype;
    a scalar limit is broadcast to vector inputs.
    The caller supplies a positive limit for clipping semantics. No runtime
    check is emitted: scalar inputs lower to one PTX instruction. Even-width
    float16/bfloat16 vectors use packed x2 instructions, one per pair.
    Float32 vectors use one instruction per lane.

    If either operand is NaN, the result is a canonical NaN; its original
    sign and payload are not preserved. Nonpositive numeric limits are
    outside this API's contract.
    Subnormals are preserved. Each operand is evaluated once.
    """
    x = tirx.convert(x)
    limit = tirx.const(limit, DataType(x.dtype).with_lanes(1)) if isinstance(limit, (int, float)) else tirx.convert(limit)
    x_dtype, limit_dtype = DataType(x.dtype), DataType(limit.dtype)
    element_dtype = x_dtype.with_lanes(1)
    if element_dtype not in (DataType("float16"), DataType("bfloat16"), DataType("float32")):
        raise TypeError("T.clamp_symmetric requires float16, bfloat16, or float32 operands")
    if limit_dtype.with_lanes(1) != element_dtype:
        raise TypeError("T.clamp_symmetric requires matching operand dtypes")
    if limit_dtype.lanes == 1 and x_dtype.lanes != 1:
        limit = tirx.Broadcast(limit, x_dtype.lanes)
    elif limit_dtype.lanes != x_dtype.lanes:
        raise ValueError("T.clamp_symmetric requires matching lanes or a scalar limit")
    return tirx.call_intrin(x.dtype, tirx.op.Op.get("tl.clamp_symmetric"), x, limit)


__all__ = ["fma", "fmul", "clamp_symmetric"]

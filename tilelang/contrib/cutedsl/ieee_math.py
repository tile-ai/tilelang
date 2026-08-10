# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
IEEE-754 compliant floating-point operations with explicit rounding modes.

These correspond to CUDA __fadd_rn, __fsub_rz, etc.  TileLang keeps the
public API names stable and delegates instruction emission to CUTLASS primitives.
"""

__all__ = [
    "ieee_fadd",
    "ieee_fsub",
    "ieee_fmul",
    "ieee_fmaf",
    "ieee_frcp",
    "ieee_fsqrt",
    "ieee_fdiv",
]

import enum

from cutlass.base_dsl.typing import Float32, Float64, Numeric
from cutlass.experimental import primitives as prims


class FPRound(str, enum.Enum):
    RN = "rn"
    RZ = "rz"
    RM = "rm"
    RP = "rp"


def _rounding(rounding: str | FPRound) -> str:
    if isinstance(rounding, FPRound):
        return rounding.value
    return FPRound(rounding).value


def _scalar_arg_type(arg):
    if isinstance(arg, Numeric):
        return type(arg)
    if hasattr(arg, "type"):
        return Numeric.from_mlir_type(arg.type)
    return None


def _use_f64(*args) -> bool:
    return any(_scalar_arg_type(arg) is Float64 for arg in args)


def _ptx_dtype(dtype) -> str:
    if dtype is Float32:
        return "f32"
    if dtype is Float64:
        return "f64"
    raise TypeError(f"Unsupported IEEE dtype: {dtype}")


def _unary_fp_ptx(op: str, dtype, a, *, rounding: str = "rn", loc=None, ip=None):
    rnd = _rounding(rounding)
    return dtype(
        prims.inline_ptx(
            f"{op}.{rnd}.{_ptx_dtype(dtype)} {{$w0}}, {{$r0}};",
            write_only_types=[dtype],
            read_only_args=[dtype(a)],
            loc=loc,
            ip=ip,
        )
    )


def _binary_fp_ptx(op: str, dtype, a, b, *, rounding: str = "rn", loc=None, ip=None):
    rnd = _rounding(rounding)
    return dtype(
        prims.inline_ptx(
            f"{op}.{rnd}.{_ptx_dtype(dtype)} {{$w0}}, {{$r0}}, {{$r1}};",
            write_only_types=[dtype],
            read_only_args=[dtype(a), dtype(b)],
            loc=loc,
            ip=ip,
        )
    )


def _ternary_fp_ptx(op: str, dtype, a, b, c, *, rounding: str = "rn", loc=None, ip=None):
    rnd = _rounding(rounding)
    return dtype(
        prims.inline_ptx(
            f"{op}.{rnd}.{_ptx_dtype(dtype)} {{$w0}}, {{$r0}}, {{$r1}}, {{$r2}};",
            write_only_types=[dtype],
            read_only_args=[dtype(a), dtype(b), dtype(c)],
            loc=loc,
            ip=ip,
        )
    )


def _fadd_f32(a: Float32, b: Float32, *, rounding: str = "rn", loc=None, ip=None) -> Float32:
    return _binary_fp_ptx("add", Float32, a, b, rounding=rounding, loc=loc, ip=ip)


def _fsub_f32(a: Float32, b: Float32, *, rounding: str = "rn", loc=None, ip=None) -> Float32:
    return _binary_fp_ptx("sub", Float32, a, b, rounding=rounding, loc=loc, ip=ip)


def _fmul_f32(a: Float32, b: Float32, *, rounding: str = "rn", loc=None, ip=None) -> Float32:
    return _binary_fp_ptx("mul", Float32, a, b, rounding=rounding, loc=loc, ip=ip)


def _fmaf_f32(a: Float32, b: Float32, c: Float32, *, rounding: str = "rn", loc=None, ip=None) -> Float32:
    return _ternary_fp_ptx("fma", Float32, a, b, c, rounding=rounding, loc=loc, ip=ip)


def _frcp_f32(a: Float32, *, rounding: str = "rn", loc=None, ip=None) -> Float32:
    return _unary_fp_ptx("rcp", Float32, a, rounding=rounding, loc=loc, ip=ip)


def _fsqrt_f32(a: Float32, *, rounding: str = "rn", loc=None, ip=None) -> Float32:
    return _unary_fp_ptx("sqrt", Float32, a, rounding=rounding, loc=loc, ip=ip)


def _fdiv_f32(a: Float32, b: Float32, *, rounding: str = "rn", loc=None, ip=None) -> Float32:
    return _binary_fp_ptx("div", Float32, a, b, rounding=rounding, loc=loc, ip=ip)


def _dadd_f64(a: Float64, b: Float64, *, rounding: str = "rn", loc=None, ip=None) -> Float64:
    return _binary_fp_ptx("add", Float64, a, b, rounding=rounding, loc=loc, ip=ip)


def _dsub_f64(a: Float64, b: Float64, *, rounding: str = "rn", loc=None, ip=None) -> Float64:
    return _binary_fp_ptx("sub", Float64, a, b, rounding=rounding, loc=loc, ip=ip)


def _dmul_f64(a: Float64, b: Float64, *, rounding: str = "rn", loc=None, ip=None) -> Float64:
    return _binary_fp_ptx("mul", Float64, a, b, rounding=rounding, loc=loc, ip=ip)


def _dmaf_f64(a: Float64, b: Float64, c: Float64, *, rounding: str = "rn", loc=None, ip=None) -> Float64:
    return _ternary_fp_ptx("fma", Float64, a, b, c, rounding=rounding, loc=loc, ip=ip)


def _drcp_f64(a: Float64, *, rounding: str = "rn", loc=None, ip=None) -> Float64:
    return _unary_fp_ptx("rcp", Float64, a, rounding=rounding, loc=loc, ip=ip)


def _dsqrt_f64(a: Float64, *, rounding: str = "rn", loc=None, ip=None) -> Float64:
    return _unary_fp_ptx("sqrt", Float64, a, rounding=rounding, loc=loc, ip=ip)


def _ddiv_f64(a: Float64, b: Float64, *, rounding: str = "rn", loc=None, ip=None) -> Float64:
    return _binary_fp_ptx("div", Float64, a, b, rounding=rounding, loc=loc, ip=ip)


def ieee_fadd(a, b, rounding="rn"):
    """IEEE-754 add with explicit rounding mode."""
    if _use_f64(a, b):
        return _dadd_f64(a, b, rounding=rounding)
    return _fadd_f32(a, b, rounding=rounding)


def ieee_fsub(a, b, rounding="rn"):
    """IEEE-754 subtract with explicit rounding mode."""
    if _use_f64(a, b):
        return _dsub_f64(a, b, rounding=rounding)
    return _fsub_f32(a, b, rounding=rounding)


def ieee_fmul(a, b, rounding="rn"):
    """IEEE-754 multiply with explicit rounding mode."""
    if _use_f64(a, b):
        return _dmul_f64(a, b, rounding=rounding)
    return _fmul_f32(a, b, rounding=rounding)


def ieee_fmaf(a, b, c, rounding="rn"):
    """IEEE-754 fused multiply-add with explicit rounding mode."""
    if _use_f64(a, b, c):
        return _dmaf_f64(a, b, c, rounding=rounding)
    return _fmaf_f32(a, b, c, rounding=rounding)


def ieee_frcp(a, rounding="rn"):
    """IEEE-754 reciprocal with explicit rounding mode."""
    if _use_f64(a):
        return _drcp_f64(a, rounding=rounding)
    return _frcp_f32(a, rounding=rounding)


def ieee_fsqrt(a, rounding="rn"):
    """IEEE-754 square root with explicit rounding mode."""
    if _use_f64(a):
        return _dsqrt_f64(a, rounding=rounding)
    return _fsqrt_f32(a, rounding=rounding)


def ieee_fdiv(a, b, rounding="rn"):
    """IEEE-754 divide with explicit rounding mode."""
    if _use_f64(a, b):
        return _ddiv_f64(a, b, rounding=rounding)
    return _fdiv_f32(a, b, rounding=rounding)

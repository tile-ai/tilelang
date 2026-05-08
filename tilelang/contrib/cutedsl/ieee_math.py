# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
IEEE-754 compliant floating-point operations with explicit rounding modes.

These correspond to CUDA __fadd_rn, __fsub_rz, etc. Implemented via inline PTX
to ensure exact rounding mode compliance.

Rounding modes: rn (nearest), rz (toward zero), rm (toward -inf), rp (toward +inf)
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

from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import Float32, Float64
from cutlass.cutlass_dsl import T, dsl_user_op


# --- f32 binary ops ---


@dsl_user_op
def _fadd_f32(a: Float32, b: Float32, *, rounding: str = "rn", loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(), Float32(b).ir_value()],
            f"add.{rounding}.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fsub_f32(a: Float32, b: Float32, *, rounding: str = "rn", loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(), Float32(b).ir_value()],
            f"sub.{rounding}.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fmul_f32(a: Float32, b: Float32, *, rounding: str = "rn", loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(), Float32(b).ir_value()],
            f"mul.{rounding}.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fmaf_f32(a: Float32, b: Float32, c: Float32, *, rounding: str = "rn", loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(), Float32(b).ir_value(), Float32(c).ir_value()],
            f"fma.{rounding}.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _frcp_f32(a: Float32, *, rounding: str = "rn", loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value()],
            f"rcp.{rounding}.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fsqrt_f32(a: Float32, *, rounding: str = "rn", loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value()],
            f"sqrt.{rounding}.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fdiv_f32(a: Float32, b: Float32, *, rounding: str = "rn", loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(), Float32(b).ir_value()],
            f"div.{rounding}.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# --- f64 binary ops ---


@dsl_user_op
def _dadd_f64(a: Float64, b: Float64, *, rounding: str = "rn", loc=None, ip=None) -> Float64:
    return Float64(
        llvm.inline_asm(
            T.f64(),
            [Float64(a).ir_value(), Float64(b).ir_value()],
            f"add.{rounding}.f64 $0, $1, $2;",
            "=d,d,d",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _dsub_f64(a: Float64, b: Float64, *, rounding: str = "rn", loc=None, ip=None) -> Float64:
    return Float64(
        llvm.inline_asm(
            T.f64(),
            [Float64(a).ir_value(), Float64(b).ir_value()],
            f"sub.{rounding}.f64 $0, $1, $2;",
            "=d,d,d",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _dmul_f64(a: Float64, b: Float64, *, rounding: str = "rn", loc=None, ip=None) -> Float64:
    return Float64(
        llvm.inline_asm(
            T.f64(),
            [Float64(a).ir_value(), Float64(b).ir_value()],
            f"mul.{rounding}.f64 $0, $1, $2;",
            "=d,d,d",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _dmaf_f64(a: Float64, b: Float64, c: Float64, *, rounding: str = "rn", loc=None, ip=None) -> Float64:
    return Float64(
        llvm.inline_asm(
            T.f64(),
            [Float64(a).ir_value(), Float64(b).ir_value(), Float64(c).ir_value()],
            f"fma.{rounding}.f64 $0, $1, $2, $3;",
            "=d,d,d,d",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _drcp_f64(a: Float64, *, rounding: str = "rn", loc=None, ip=None) -> Float64:
    return Float64(
        llvm.inline_asm(
            T.f64(),
            [Float64(a).ir_value()],
            f"rcp.{rounding}.f64 $0, $1;",
            "=d,d",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _dsqrt_f64(a: Float64, *, rounding: str = "rn", loc=None, ip=None) -> Float64:
    return Float64(
        llvm.inline_asm(
            T.f64(),
            [Float64(a).ir_value()],
            f"sqrt.{rounding}.f64 $0, $1;",
            "=d,d",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ddiv_f64(a: Float64, b: Float64, *, rounding: str = "rn", loc=None, ip=None) -> Float64:
    return Float64(
        llvm.inline_asm(
            T.f64(),
            [Float64(a).ir_value(), Float64(b).ir_value()],
            f"div.{rounding}.f64 $0, $1, $2;",
            "=d,d,d",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# --- Public API (dispatches by dtype) ---


def ieee_fadd(a, b, rounding="rn"):
    """IEEE-754 add with explicit rounding mode."""
    return _fadd_f32(a, b, rounding=rounding)


def ieee_fsub(a, b, rounding="rn"):
    """IEEE-754 subtract with explicit rounding mode."""
    return _fsub_f32(a, b, rounding=rounding)


def ieee_fmul(a, b, rounding="rn"):
    """IEEE-754 multiply with explicit rounding mode."""
    return _fmul_f32(a, b, rounding=rounding)


def ieee_fmaf(a, b, c, rounding="rn"):
    """IEEE-754 fused multiply-add with explicit rounding mode."""
    return _fmaf_f32(a, b, c, rounding=rounding)


def ieee_frcp(a, rounding="rn"):
    """IEEE-754 reciprocal with explicit rounding mode."""
    return _frcp_f32(a, rounding=rounding)


def ieee_fsqrt(a, rounding="rn"):
    """IEEE-754 square root with explicit rounding mode."""
    return _fsqrt_f32(a, rounding=rounding)


def ieee_fdiv(a, b, rounding="rn"):
    """IEEE-754 divide with explicit rounding mode."""
    return _fdiv_f32(a, b, rounding=rounding)

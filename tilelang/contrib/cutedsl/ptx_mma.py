"""
PTX MMA operations for CuTeDSL backend.
Based on tl_templates/cuda/instruction/mma.h

These functions provide wrappers around PTX mma.sync instructions
for performing matrix multiply-accumulate operations using Tensor Cores.

Uses inline PTX assembly for direct MMA instruction generation.

Supported dense configurations (from mma.h):
- FP16: m16n8k16 -> f16/f32 accumulator
- BF16: m16n8k16 -> f32 accumulator
- INT8: m16n8k32 -> i32 accumulator
- UINT8: m16n8k32 -> i32 accumulator
- INT4: m16n8k32 -> i32 accumulator (mapped to m16n8k64 in PTX)
- UINT4: m16n8k32 -> i32 accumulator
- FP8 (e4m3/e5m2): m16n8k32 -> f16/f32 accumulator
- TF32: m16n8k4, m16n8k8 -> f32 accumulator
- FP64: m8n8k4 -> f64 accumulator

Sparse (mma.sp) variants mirror the dense ones with halved A registers,
an extra metadata register, and a sparse_selector literal.
"""

__all__ = [
    "ptx_mma_m16n8k16_f16_f16_f32",
    "ptx_mma_m16n8k16_f16_f16_f16",
    "ptx_mma_m16n8k16_bf16_bf16_f32",
    "ptx_mma_m16n8k32_s8_s8_s32",
    "ptx_mma_m16n8k32_u8_u8_s32",
    "ptx_mma_m16n8k32_s4_s4_s32",
    "ptx_mma_m16n8k32_u4_u4_s32",
    "ptx_mma_m16n8k4_tf32_tf32_f32",
    "ptx_mma_m16n8k8_tf32_tf32_f32",
    "ptx_mma_m8n8k4_f64_f64_f64",
    "ptx_mma_m16n8k32_e4m3_e4m3_f32",
    "ptx_mma_m16n8k32_e4m3_e4m3_f16",
    "ptx_mma_m16n8k32_e5m2_e5m2_f32",
    "ptx_mma",
    "ptx_mma_sp",
]

from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import Pointer
import cutlass.cute as cute

_VALID_LAYOUTS = {"row", "col"}

# Flavor configs: (llvm_type_fn, c_constraint, ab_constraint,
#                  ab_wrapper, c_wrapper, ab_recast, c_recast)
_FLAVOR = {
    "f32": (T.f32, "f", "r", cute.Int32, cute.Float32, cute.Int32, None),
    "i32": (T.i32, "r", "r", cute.Int32, cute.Int32, cute.Int32, cute.Int32),
    "f64": (T.f64, "d", "d", cute.Float64, cute.Float64, None, None),
}


# =============================================================================
# Dense MMA factory
# =============================================================================


def _make_ptx_mma(ptx_shape, ptx_dtypes, n_a, n_b, n_c, flavor):
    """Factory to create a @dsl_user_op for ``mma.sync.aligned``.

    Args:
        ptx_shape: PTX shape, e.g. "m16n8k16"
        ptx_dtypes: PTX dtype suffixes, e.g. "f32.f16.f16.f32" (D.A.B.C)
        n_a: Number of A registers
        n_b: Number of B registers
        n_c: Number of C/D registers
        flavor: "f32", "i32", or "f64" — selects register types and constraints
    """
    llvm_type_fn, c_con, ab_con, ab_wrap, c_wrap, ab_recast, c_recast = _FLAVOR[flavor]

    # Pre-build constraints string
    constraints = ",".join([f"={c_con}"] * n_c + [ab_con] * (n_a + n_b) + [c_con] * n_c)

    # Pre-build PTX asm template ({a_layout}/{b_layout} substituted per call)
    d_regs = ", ".join(f"${i}" for i in range(n_c))
    a_regs = ", ".join(f"${n_c + i}" for i in range(n_a))
    b_regs = ", ".join(f"${n_c + n_a + i}" for i in range(n_b))
    c_regs = ", ".join(f"${n_c + n_a + n_b + i}" for i in range(n_c))
    ptx_template = (
        f"mma.sync.aligned.{ptx_shape}.{{a_layout}}.{{b_layout}}.{ptx_dtypes}"
        f" {{{{{d_regs}}}}}, {{{{{a_regs}}}}}, {{{{{b_regs}}}}}, {{{{{c_regs}}}}};"
    )

    @dsl_user_op
    def mma_op(
        a_ptr: Pointer,
        a_offset,
        b_ptr: Pointer,
        b_offset,
        c_ptr: Pointer,
        c_offset,
        a_layout: str = "row",
        b_layout: str = "col",
        *,
        loc=None,
        ip=None,
    ) -> None:
        assert a_layout in _VALID_LAYOUTS, f"invalid a_layout: {a_layout!r}"
        assert b_layout in _VALID_LAYOUTS, f"invalid b_layout: {b_layout!r}"

        # A operand
        a_base = cute.recast_ptr(a_ptr + a_offset, dtype=ab_recast) if ab_recast else (a_ptr + a_offset)
        a_tensor = cute.make_tensor(a_base, (n_a,))
        a_vals = [ab_wrap(a_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(n_a)]

        # B operand
        b_base = cute.recast_ptr(b_ptr + b_offset, dtype=ab_recast) if ab_recast else (b_ptr + b_offset)
        b_tensor = cute.make_tensor(b_base, (n_b,))
        b_vals = [ab_wrap(b_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(n_b)]

        # C operand
        c_base = cute.recast_ptr(c_ptr + c_offset, dtype=c_recast) if c_recast else (c_ptr + c_offset)
        c_tensor = cute.make_tensor(c_base, (n_c,))
        c_vals = [c_wrap(c_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(n_c)]

        # Inline asm
        llvm_elem = llvm_type_fn()
        res_type = llvm.StructType.get_literal([llvm_elem] * n_c)
        ptx_asm = ptx_template.format(a_layout=a_layout, b_layout=b_layout)

        result = llvm.inline_asm(
            res_type,
            a_vals + b_vals + c_vals,
            ptx_asm,
            constraints,
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

        # Write results back
        d_tensor = cute.make_tensor(c_base, (n_c,))
        for i in range(n_c):
            d_tensor[i] = c_wrap(llvm.extractvalue(llvm_elem, result, [i], loc=loc, ip=ip))

    return mma_op


# =============================================================================
# Sparse MMA factory
# =============================================================================


def _make_ptx_mma_sp(ptx_shape, ptx_dtypes, n_a, n_b, n_c, flavor):
    """Factory to create a @dsl_user_op for ``mma.sp.sync.aligned``.

    Args:
        ptx_shape: PTX shape, e.g. "m16n8k32"
        ptx_dtypes: PTX dtype suffixes, e.g. "f32.f16.f16.f32" (D.A.B.C)
        n_a: Number of A registers
        n_b: Number of B registers
        n_c: Number of C/D registers
        flavor: "f32" or "i32" — selects register types and constraints
    """
    llvm_type_fn, c_con, ab_con, ab_wrap, c_wrap, ab_recast, c_recast = _FLAVOR[flavor]

    # Pre-build constraints string (extra "r" for the 1 metadata register)
    constraints = ",".join([f"={c_con}"] * n_c + [ab_con] * (n_a + n_b) + [c_con] * n_c + ["r"])

    # Pre-build PTX asm template ({a_layout}/{b_layout}/{sparse_selector} per call)
    d_regs = ", ".join(f"${i}" for i in range(n_c))
    a_regs = ", ".join(f"${n_c + i}" for i in range(n_a))
    b_regs = ", ".join(f"${n_c + n_a + i}" for i in range(n_b))
    c_regs = ", ".join(f"${n_c + n_a + n_b + i}" for i in range(n_c))
    meta_reg = f"${n_c + n_a + n_b + n_c}"
    ptx_template = (
        f"mma.sp.sync.aligned.{ptx_shape}.{{a_layout}}.{{b_layout}}.{ptx_dtypes}"
        f" {{{{{d_regs}}}}}, {{{{{a_regs}}}}}, {{{{{b_regs}}}}}, {{{{{c_regs}}}}}, {meta_reg}, 0x{{sparse_selector}};"
    )

    @dsl_user_op
    def mma_sp_op(
        a_ptr: Pointer,
        a_offset,
        b_ptr: Pointer,
        b_offset,
        c_ptr: Pointer,
        c_offset,
        meta_ptr: Pointer,
        meta_offset,
        sparse_selector: int = 0,
        a_layout: str = "row",
        b_layout: str = "col",
        *,
        loc=None,
        ip=None,
    ) -> None:
        assert a_layout in _VALID_LAYOUTS, f"invalid a_layout: {a_layout!r}"
        assert b_layout in _VALID_LAYOUTS, f"invalid b_layout: {b_layout!r}"

        # A operand
        a_base = cute.recast_ptr(a_ptr + a_offset, dtype=ab_recast) if ab_recast else (a_ptr + a_offset)
        a_tensor = cute.make_tensor(a_base, (n_a,))
        a_vals = [ab_wrap(a_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(n_a)]

        # B operand
        b_base = cute.recast_ptr(b_ptr + b_offset, dtype=ab_recast) if ab_recast else (b_ptr + b_offset)
        b_tensor = cute.make_tensor(b_base, (n_b,))
        b_vals = [ab_wrap(b_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(n_b)]

        # C operand
        c_base = cute.recast_ptr(c_ptr + c_offset, dtype=c_recast) if c_recast else (c_ptr + c_offset)
        c_tensor = cute.make_tensor(c_base, (n_c,))
        c_vals = [c_wrap(c_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(n_c)]

        # Metadata (1 register, always Int32)
        meta_base = cute.recast_ptr(meta_ptr + meta_offset, dtype=cute.Int32)
        meta_tensor = cute.make_tensor(meta_base, (1,))
        meta_val = cute.Int32(meta_tensor[0]).ir_value(loc=loc, ip=ip)

        # Inline asm
        llvm_elem = llvm_type_fn()
        res_type = llvm.StructType.get_literal([llvm_elem] * n_c)
        ptx_asm = ptx_template.format(
            a_layout=a_layout,
            b_layout=b_layout,
            sparse_selector=sparse_selector,
        )

        result = llvm.inline_asm(
            res_type,
            a_vals + b_vals + c_vals + [meta_val],
            ptx_asm,
            constraints,
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

        # Write results back
        d_tensor = cute.make_tensor(c_base, (n_c,))
        for i in range(n_c):
            d_tensor[i] = c_wrap(llvm.extractvalue(llvm_elem, result, [i], loc=loc, ip=ip))

    return mma_sp_op


# =============================================================================
# Instantiate dense MMA variants
#   args: ptx_shape, ptx_dtypes (D.A.B.C), n_a, n_b, n_c, flavor
#   flavor: "f32" (float accum), "i32" (int/packed-half accum), "f64"
# =============================================================================

# FP16
ptx_mma_m16n8k16_f16_f16_f32 = _make_ptx_mma("m16n8k16", "f32.f16.f16.f32", 4, 2, 4, "f32")
ptx_mma_m16n8k16_f16_f16_f16 = _make_ptx_mma("m16n8k16", "f16.f16.f16.f16", 4, 2, 2, "i32")
# BF16
ptx_mma_m16n8k16_bf16_bf16_f32 = _make_ptx_mma("m16n8k16", "f32.bf16.bf16.f32", 4, 2, 4, "f32")
# INT8
ptx_mma_m16n8k32_s8_s8_s32 = _make_ptx_mma("m16n8k32", "s32.s8.s8.s32", 4, 2, 4, "i32")
ptx_mma_m16n8k32_u8_u8_s32 = _make_ptx_mma("m16n8k32", "s32.u8.u8.s32", 4, 2, 4, "i32")
# INT4 (TileLang m16n8k32 -> PTX m16n8k64)
ptx_mma_m16n8k32_s4_s4_s32 = _make_ptx_mma("m16n8k64", "s32.s4.s4.s32", 4, 2, 4, "i32")
ptx_mma_m16n8k32_u4_u4_s32 = _make_ptx_mma("m16n8k64", "s32.u4.u4.s32", 4, 2, 4, "i32")
# TF32
ptx_mma_m16n8k4_tf32_tf32_f32 = _make_ptx_mma("m16n8k4", "f32.tf32.tf32.f32", 2, 1, 4, "f32")
ptx_mma_m16n8k8_tf32_tf32_f32 = _make_ptx_mma("m16n8k8", "f32.tf32.tf32.f32", 4, 2, 4, "f32")
# FP64
ptx_mma_m8n8k4_f64_f64_f64 = _make_ptx_mma("m8n8k4", "f64.f64.f64.f64", 1, 1, 2, "f64")
# FP8 (SM89+)
ptx_mma_m16n8k32_e4m3_e4m3_f32 = _make_ptx_mma("m16n8k32", "f32.e4m3.e4m3.f32", 4, 2, 4, "f32")
ptx_mma_m16n8k32_e4m3_e4m3_f16 = _make_ptx_mma("m16n8k32", "f16.e4m3.e4m3.f16", 4, 2, 2, "i32")
ptx_mma_m16n8k32_e5m2_e5m2_f32 = _make_ptx_mma("m16n8k32", "f32.e5m2.e5m2.f32", 4, 2, 4, "f32")


# =============================================================================
# Instantiate sparse MMA variants
#   args: ptx_shape, ptx_dtypes (D.A.B.C), n_a, n_b, n_c, flavor
#   flavor: "f32" (float accum), "i32" (int/packed-half accum)
# =============================================================================

# FP16
ptx_mma_sp_m16n8k32_f16_f16_f32 = _make_ptx_mma_sp("m16n8k32", "f32.f16.f16.f32", 4, 4, 4, "f32")
ptx_mma_sp_m16n8k32_f16_f16_f16 = _make_ptx_mma_sp("m16n8k32", "f16.f16.f16.f16", 4, 4, 2, "i32")
ptx_mma_sp_m16n8k16_f16_f16_f32 = _make_ptx_mma_sp("m16n8k16", "f32.f16.f16.f32", 2, 2, 4, "f32")
ptx_mma_sp_m16n8k16_f16_f16_f16 = _make_ptx_mma_sp("m16n8k16", "f16.f16.f16.f16", 2, 2, 2, "i32")
# BF16
ptx_mma_sp_m16n8k32_bf16_bf16_f32 = _make_ptx_mma_sp("m16n8k32", "f32.bf16.bf16.f32", 4, 4, 4, "f32")
ptx_mma_sp_m16n8k16_bf16_bf16_f32 = _make_ptx_mma_sp("m16n8k16", "f32.bf16.bf16.f32", 2, 2, 4, "f32")
# INT8
ptx_mma_sp_m16n8k64_s8_s8_s32 = _make_ptx_mma_sp("m16n8k64", "s32.s8.s8.s32", 4, 4, 4, "i32")
ptx_mma_sp_m16n8k32_s8_s8_s32 = _make_ptx_mma_sp("m16n8k32", "s32.s8.s8.s32", 2, 2, 4, "i32")
# TF32
ptx_mma_sp_m16n8k16_tf32_tf32_f32 = _make_ptx_mma_sp("m16n8k16", "f32.tf32.tf32.f32", 2, 2, 4, "f32")
ptx_mma_sp_m16n8k8_tf32_tf32_f32 = _make_ptx_mma_sp("m16n8k8", "f32.tf32.tf32.f32", 1, 1, 4, "f32")


# =============================================================================
# Dense MMA dispatcher
# =============================================================================


def ptx_mma(
    shape: str,
    a_layout: str,
    b_layout: str,
    a_dtype: str,
    b_dtype: str,
    c_dtype: str,
    a_ptr,
    a_offset,
    b_ptr,
    b_offset,
    c_ptr,
    c_offset,
    saturate: bool = False,
):
    """Generic PTX MMA dispatcher.

    Dispatches to the appropriate specialized MMA function based on
    shape and data types.
    """
    if saturate:
        raise NotImplementedError("saturate=True (.satfinite) is not yet supported in CuTeDSL backend")

    shape = shape.lower()
    a_dtype = a_dtype.lower()
    b_dtype = b_dtype.lower()
    c_dtype = c_dtype.lower()

    _args = (a_ptr, a_offset, b_ptr, b_offset, c_ptr, c_offset, a_layout, b_layout)

    # Dispatch based on shape and types
    if shape == "m16n8k16":
        if a_dtype in ["fp16", "f16", "float16"]:
            if c_dtype in ["fp32", "f32", "float32"]:
                return ptx_mma_m16n8k16_f16_f16_f32(*_args)
            elif c_dtype in ["fp16", "f16", "float16"]:
                return ptx_mma_m16n8k16_f16_f16_f16(*_args)
        elif a_dtype in ["bf16", "bfloat16"] and c_dtype in ["fp32", "f32", "float32"]:
            return ptx_mma_m16n8k16_bf16_bf16_f32(*_args)

    elif shape == "m16n8k32":
        if a_dtype in ["int8", "s8"]:
            return ptx_mma_m16n8k32_s8_s8_s32(*_args)
        elif a_dtype in ["uint8", "u8"]:
            return ptx_mma_m16n8k32_u8_u8_s32(*_args)
        elif a_dtype in ["int4", "s4"]:
            return ptx_mma_m16n8k32_s4_s4_s32(*_args)
        elif a_dtype in ["uint4", "u4"]:
            return ptx_mma_m16n8k32_u4_u4_s32(*_args)
        elif a_dtype in ["e4m3", "float8_e4m3", "fp8_e4m3"]:
            if c_dtype in ["fp32", "f32", "float32"]:
                return ptx_mma_m16n8k32_e4m3_e4m3_f32(*_args)
            elif c_dtype in ["fp16", "f16", "float16"]:
                return ptx_mma_m16n8k32_e4m3_e4m3_f16(*_args)
        elif a_dtype in ["e5m2", "float8_e5m2", "fp8_e5m2"] and c_dtype in ["fp32", "f32", "float32"]:
            return ptx_mma_m16n8k32_e5m2_e5m2_f32(*_args)

    elif shape == "m16n8k4":
        # TF32: accept tf32 or fp32 (TileLang may pass float32 for TF32 GEMM)
        if a_dtype in ["tf32", "tensorfloat32", "fp32", "f32", "float32"]:
            return ptx_mma_m16n8k4_tf32_tf32_f32(*_args)

    elif shape == "m16n8k8":
        # TF32: accept tf32 or fp32 (e.g. deepseek_mhc)
        if a_dtype in ["tf32", "tensorfloat32", "fp32", "f32", "float32"]:
            return ptx_mma_m16n8k8_tf32_tf32_f32(*_args)

    elif shape == "m8n8k4" and a_dtype in ["fp64", "f64", "float64"]:
        return ptx_mma_m8n8k4_f64_f64_f64(*_args)

    raise ValueError(f"Unsupported MMA configuration: shape={shape}, a_dtype={a_dtype}, b_dtype={b_dtype}, c_dtype={c_dtype}")


# =============================================================================
# Sparse MMA dispatcher
# =============================================================================


def ptx_mma_sp(
    shape: str,
    a_layout: str,
    b_layout: str,
    a_dtype: str,
    b_dtype: str,
    c_dtype: str,
    a_ptr,
    a_offset,
    b_ptr,
    b_offset,
    c_ptr,
    c_offset,
    meta_ptr,
    meta_offset,
    sparse_selector: int = 0,
    saturate: bool = False,
):
    """Generic PTX sparse MMA dispatcher.

    Dispatches to the appropriate specialized sparse MMA function based on
    shape and data types.
    """
    if saturate:
        raise NotImplementedError("saturate=True (.satfinite) is not yet supported in CuTeDSL backend")

    shape = shape.lower()
    a_dtype = a_dtype.lower()
    b_dtype = b_dtype.lower()
    c_dtype = c_dtype.lower()

    _args = (a_ptr, a_offset, b_ptr, b_offset, c_ptr, c_offset, meta_ptr, meta_offset, sparse_selector, a_layout, b_layout)

    # Dispatch based on shape and types
    if shape == "m16n8k32":
        if a_dtype in ["fp16", "f16", "float16"]:
            if c_dtype in ["fp32", "f32", "float32"]:
                return ptx_mma_sp_m16n8k32_f16_f16_f32(*_args)
            elif c_dtype in ["fp16", "f16", "float16"]:
                return ptx_mma_sp_m16n8k32_f16_f16_f16(*_args)
        elif a_dtype in ["bf16", "bfloat16"] and c_dtype in ["fp32", "f32", "float32"]:
            return ptx_mma_sp_m16n8k32_bf16_bf16_f32(*_args)
        elif a_dtype in ["int8", "s8"]:
            return ptx_mma_sp_m16n8k32_s8_s8_s32(*_args)

    elif shape == "m16n8k16":
        if a_dtype in ["fp16", "f16", "float16"]:
            if c_dtype in ["fp32", "f32", "float32"]:
                return ptx_mma_sp_m16n8k16_f16_f16_f32(*_args)
            elif c_dtype in ["fp16", "f16", "float16"]:
                return ptx_mma_sp_m16n8k16_f16_f16_f16(*_args)
        elif a_dtype in ["bf16", "bfloat16"] and c_dtype in ["fp32", "f32", "float32"]:
            return ptx_mma_sp_m16n8k16_bf16_bf16_f32(*_args)
        elif a_dtype in ["tf32", "tensorfloat32", "fp32", "f32", "float32"]:
            return ptx_mma_sp_m16n8k16_tf32_tf32_f32(*_args)

    elif shape == "m16n8k64":
        if a_dtype in ["int8", "s8"]:
            return ptx_mma_sp_m16n8k64_s8_s8_s32(*_args)

    elif shape == "m16n8k8" and a_dtype in ["tf32", "tensorfloat32", "fp32", "f32", "float32"]:
        return ptx_mma_sp_m16n8k8_tf32_tf32_f32(*_args)

    raise ValueError(f"Unsupported sparse MMA configuration: shape={shape}, a_dtype={a_dtype}, b_dtype={b_dtype}, c_dtype={c_dtype}")

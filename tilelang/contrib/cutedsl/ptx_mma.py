"""
PTX MMA operations for CuTeDSL backend.
Based on tl_templates/cuda/instruction/mma.h

These functions provide wrappers around PTX mma.sync instructions
for performing matrix multiply-accumulate operations using Tensor Cores.

Uses inline PTX assembly for direct MMA instruction generation.

Supported configurations (from mma.h):
- FP16: m16n8k16 -> f16/f32 accumulator
- BF16: m16n8k16 -> f32 accumulator
- INT8: m16n8k32 -> i32 accumulator
- UINT8: m16n8k32 -> i32 accumulator
- INT4: m16n8k32 -> i32 accumulator (mapped to m16n8k64 in PTX)
- UINT4: m16n8k32 -> i32 accumulator
- FP8 (e4m3/e5m2): m16n8k32 -> f16/f32 accumulator
- TF32: m16n8k4, m16n8k8 -> f32 accumulator
- FP64: m8n8k4 -> f64 accumulator
"""

from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import Pointer
import cutlass.cute as cute


def _to_ir_value(v, loc=None, ip=None):
    """Convert value to MLIR IR, handling both cutlass types and raw MLIR Values"""
    if hasattr(v, "ir_value"):
        return v.ir_value(loc=loc, ip=ip)
    else:
        return v


# =============================================================================
# FP16 MMA operations
# =============================================================================


@dsl_user_op
def ptx_mma_m16n8k16_f16_f16_f32(
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
    """m16n8k16 MMA: f16 inputs, f32 accumulator"""
    # A: 8 f16 = 4 i32, B: 4 f16 = 2 i32, C: 4 f32
    a_base = cute.recast_ptr(a_ptr + a_offset, dtype=cute.Int32)
    a_tensor = cute.make_tensor(a_base, (4,))
    b_base = cute.recast_ptr(b_ptr + b_offset, dtype=cute.Int32)
    b_tensor = cute.make_tensor(b_base, (2,))
    c_base = c_ptr + c_offset
    c_tensor = cute.make_tensor(c_base, (4,))

    a_vals = [cute.Int32(a_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(4)]
    b_vals = [cute.Int32(b_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(2)]
    c_vals = [cute.Float32(c_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(4)]

    res_type = llvm.StructType.get_literal([T.f32()] * 4)
    ptx_asm = """
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$10, $11, $12, $13};
    """
    constraints = "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
    operands = a_vals + b_vals + c_vals

    result = llvm.inline_asm(
        res_type,
        operands,
        ptx_asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    d_tensor = cute.make_tensor(c_base, (4,))
    for i in range(4):
        d_tensor[i] = cute.Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))


@dsl_user_op
def ptx_mma_m16n8k16_f16_f16_f16(
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
    """m16n8k16 MMA: f16 inputs, f16 accumulator"""
    # A: 4 i32, B: 2 i32, C: 2 i32 (4 f16 packed)
    a_base = cute.recast_ptr(a_ptr + a_offset, dtype=cute.Int32)
    a_tensor = cute.make_tensor(a_base, (4,))
    b_base = cute.recast_ptr(b_ptr + b_offset, dtype=cute.Int32)
    b_tensor = cute.make_tensor(b_base, (2,))
    c_base = cute.recast_ptr(c_ptr + c_offset, dtype=cute.Int32)
    c_tensor = cute.make_tensor(c_base, (2,))

    a_vals = [cute.Int32(a_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(4)]
    b_vals = [cute.Int32(b_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(2)]
    c_vals = [cute.Int32(c_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(2)]

    res_type = llvm.StructType.get_literal([T.i32()] * 2)
    ptx_asm = """
    mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
        {$0, $1},
        {$2, $3, $4, $5},
        {$6, $7},
        {$8, $9};
    """
    constraints = "=r,=r,r,r,r,r,r,r,r,r"
    operands = a_vals + b_vals + c_vals

    result = llvm.inline_asm(
        res_type,
        operands,
        ptx_asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    d_tensor = cute.make_tensor(c_base, (2,))
    for i in range(2):
        d_tensor[i] = cute.Int32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))


# =============================================================================
# BF16 MMA operations
# =============================================================================


@dsl_user_op
def ptx_mma_m16n8k16_bf16_bf16_f32(
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
    """m16n8k16 MMA: bf16 inputs, f32 accumulator"""
    a_base = cute.recast_ptr(a_ptr + a_offset, dtype=cute.Int32)
    a_tensor = cute.make_tensor(a_base, (4,))
    b_base = cute.recast_ptr(b_ptr + b_offset, dtype=cute.Int32)
    b_tensor = cute.make_tensor(b_base, (2,))
    c_base = c_ptr + c_offset
    c_tensor = cute.make_tensor(c_base, (4,))

    a_vals = [cute.Int32(a_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(4)]
    b_vals = [cute.Int32(b_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(2)]
    c_vals = [cute.Float32(c_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(4)]

    res_type = llvm.StructType.get_literal([T.f32()] * 4)
    ptx_asm = """
    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$10, $11, $12, $13};
    """
    constraints = "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
    operands = a_vals + b_vals + c_vals

    result = llvm.inline_asm(
        res_type,
        operands,
        ptx_asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    d_tensor = cute.make_tensor(c_base, (4,))
    for i in range(4):
        d_tensor[i] = cute.Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))


# =============================================================================
# INT8/UINT8 MMA operations (m16n8k32)
# =============================================================================


@dsl_user_op
def ptx_mma_m16n8k32_s8_s8_s32(
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
    """m16n8k32 MMA: int8 inputs, int32 accumulator"""
    a_base = cute.recast_ptr(a_ptr + a_offset, dtype=cute.Int32)
    a_tensor = cute.make_tensor(a_base, (4,))
    b_base = cute.recast_ptr(b_ptr + b_offset, dtype=cute.Int32)
    b_tensor = cute.make_tensor(b_base, (2,))
    c_base = cute.recast_ptr(c_ptr + c_offset, dtype=cute.Int32)
    c_tensor = cute.make_tensor(c_base, (4,))

    a_vals = [cute.Int32(a_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(4)]
    b_vals = [cute.Int32(b_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(2)]
    c_vals = [cute.Int32(c_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(4)]

    res_type = llvm.StructType.get_literal([T.i32()] * 4)
    ptx_asm = """
    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$10, $11, $12, $13};
    """
    constraints = "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
    operands = a_vals + b_vals + c_vals

    result = llvm.inline_asm(
        res_type,
        operands,
        ptx_asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    d_tensor = cute.make_tensor(c_base, (4,))
    for i in range(4):
        d_tensor[i] = cute.Int32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))


@dsl_user_op
def ptx_mma_m16n8k32_u8_u8_s32(
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
    """m16n8k32 MMA: uint8 inputs, int32 accumulator"""
    a_base = cute.recast_ptr(a_ptr + a_offset, dtype=cute.Int32)
    a_tensor = cute.make_tensor(a_base, (4,))
    b_base = cute.recast_ptr(b_ptr + b_offset, dtype=cute.Int32)
    b_tensor = cute.make_tensor(b_base, (2,))
    c_base = cute.recast_ptr(c_ptr + c_offset, dtype=cute.Int32)
    c_tensor = cute.make_tensor(c_base, (4,))

    a_vals = [cute.Int32(a_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(4)]
    b_vals = [cute.Int32(b_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(2)]
    c_vals = [cute.Int32(c_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(4)]

    res_type = llvm.StructType.get_literal([T.i32()] * 4)
    ptx_asm = """
    mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$10, $11, $12, $13};
    """
    constraints = "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
    operands = a_vals + b_vals + c_vals

    result = llvm.inline_asm(
        res_type,
        operands,
        ptx_asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    d_tensor = cute.make_tensor(c_base, (4,))
    for i in range(4):
        d_tensor[i] = cute.Int32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))


# =============================================================================
# INT4/UINT4 MMA operations (m16n8k32 in TileLang, m16n8k64 in PTX)
# =============================================================================


@dsl_user_op
def ptx_mma_m16n8k32_s4_s4_s32(
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
    """m16n8k32 (logical) -> m16n8k64 (PTX) MMA: int4 inputs, int32 accumulator"""
    # Note: TileLang uses m16n8k32 but PTX uses m16n8k64 for int4
    a_base = cute.recast_ptr(a_ptr + a_offset, dtype=cute.Int32)
    a_tensor = cute.make_tensor(a_base, (4,))
    b_base = cute.recast_ptr(b_ptr + b_offset, dtype=cute.Int32)
    b_tensor = cute.make_tensor(b_base, (2,))
    c_base = cute.recast_ptr(c_ptr + c_offset, dtype=cute.Int32)
    c_tensor = cute.make_tensor(c_base, (4,))

    a_vals = [cute.Int32(a_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(4)]
    b_vals = [cute.Int32(b_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(2)]
    c_vals = [cute.Int32(c_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(4)]

    res_type = llvm.StructType.get_literal([T.i32()] * 4)
    ptx_asm = """
    mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$10, $11, $12, $13};
    """
    constraints = "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
    operands = a_vals + b_vals + c_vals

    result = llvm.inline_asm(
        res_type,
        operands,
        ptx_asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    d_tensor = cute.make_tensor(c_base, (4,))
    for i in range(4):
        d_tensor[i] = cute.Int32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))


@dsl_user_op
def ptx_mma_m16n8k32_u4_u4_s32(
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
    """m16n8k32 (logical) -> m16n8k64 (PTX) MMA: uint4 inputs, int32 accumulator"""
    a_base = cute.recast_ptr(a_ptr + a_offset, dtype=cute.Int32)
    a_tensor = cute.make_tensor(a_base, (4,))
    b_base = cute.recast_ptr(b_ptr + b_offset, dtype=cute.Int32)
    b_tensor = cute.make_tensor(b_base, (2,))
    c_base = cute.recast_ptr(c_ptr + c_offset, dtype=cute.Int32)
    c_tensor = cute.make_tensor(c_base, (4,))

    a_vals = [cute.Int32(a_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(4)]
    b_vals = [cute.Int32(b_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(2)]
    c_vals = [cute.Int32(c_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(4)]

    res_type = llvm.StructType.get_literal([T.i32()] * 4)
    ptx_asm = """
    mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$10, $11, $12, $13};
    """
    constraints = "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r"
    operands = a_vals + b_vals + c_vals

    result = llvm.inline_asm(
        res_type,
        operands,
        ptx_asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    d_tensor = cute.make_tensor(c_base, (4,))
    for i in range(4):
        d_tensor[i] = cute.Int32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))


# =============================================================================
# TF32 MMA operations (m16n8k4, m16n8k8)
# =============================================================================


@dsl_user_op
def ptx_mma_m16n8k4_tf32_tf32_f32(
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
    """m16n8k4 MMA: tf32 inputs, f32 accumulator"""
    # A: 2 regs, B: 1 reg, C: 4 regs
    a_base = cute.recast_ptr(a_ptr + a_offset, dtype=cute.Int32)
    a_tensor = cute.make_tensor(a_base, (2,))
    b_base = cute.recast_ptr(b_ptr + b_offset, dtype=cute.Int32)
    b_tensor = cute.make_tensor(b_base, (1,))
    c_base = c_ptr + c_offset
    c_tensor = cute.make_tensor(c_base, (4,))

    a_vals = [cute.Int32(a_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(2)]
    b_vals = [cute.Int32(b_tensor[0]).ir_value(loc=loc, ip=ip)]
    c_vals = [cute.Float32(c_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(4)]

    res_type = llvm.StructType.get_literal([T.f32()] * 4)
    ptx_asm = """
    mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32
        {$0, $1, $2, $3},
        {$4, $5},
        {$6},
        {$7, $8, $9, $10};
    """
    constraints = "=f,=f,=f,=f,r,r,r,f,f,f,f"
    operands = a_vals + b_vals + c_vals

    result = llvm.inline_asm(
        res_type,
        operands,
        ptx_asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    d_tensor = cute.make_tensor(c_base, (4,))
    for i in range(4):
        d_tensor[i] = cute.Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))


@dsl_user_op
def ptx_mma_m16n8k8_tf32_tf32_f32(
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
    """m16n8k8 MMA: tf32 inputs, f32 accumulator"""
    # A: 4 regs, B: 2 regs, C: 4 regs
    a_base = cute.recast_ptr(a_ptr + a_offset, dtype=cute.Int32)
    a_tensor = cute.make_tensor(a_base, (4,))
    b_base = cute.recast_ptr(b_ptr + b_offset, dtype=cute.Int32)
    b_tensor = cute.make_tensor(b_base, (2,))
    c_base = c_ptr + c_offset
    c_tensor = cute.make_tensor(c_base, (4,))

    a_vals = [cute.Int32(a_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(4)]
    b_vals = [cute.Int32(b_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(2)]
    c_vals = [cute.Float32(c_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(4)]

    res_type = llvm.StructType.get_literal([T.f32()] * 4)
    ptx_asm = """
    mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$10, $11, $12, $13};
    """
    constraints = "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
    operands = a_vals + b_vals + c_vals

    result = llvm.inline_asm(
        res_type,
        operands,
        ptx_asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    d_tensor = cute.make_tensor(c_base, (4,))
    for i in range(4):
        d_tensor[i] = cute.Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))


# =============================================================================
# FP64 MMA operations (m8n8k4)
# =============================================================================


@dsl_user_op
def ptx_mma_m8n8k4_f64_f64_f64(
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
    """m8n8k4 MMA: f64 inputs, f64 accumulator"""
    # A: 1 f64, B: 1 f64, C: 2 f64
    a_tensor = cute.make_tensor(a_ptr + a_offset, (1,))
    b_tensor = cute.make_tensor(b_ptr + b_offset, (1,))
    c_base = c_ptr + c_offset
    c_tensor = cute.make_tensor(c_base, (2,))

    a_vals = [cute.Float64(a_tensor[0]).ir_value(loc=loc, ip=ip)]
    b_vals = [cute.Float64(b_tensor[0]).ir_value(loc=loc, ip=ip)]
    c_vals = [cute.Float64(c_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(2)]

    res_type = llvm.StructType.get_literal([T.f64()] * 2)
    ptx_asm = """
    mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64
        {$0, $1},
        {$2},
        {$3},
        {$4, $5};
    """
    constraints = "=d,=d,d,d,d,d"
    operands = a_vals + b_vals + c_vals

    result = llvm.inline_asm(
        res_type,
        operands,
        ptx_asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    d_tensor = cute.make_tensor(c_base, (2,))
    for i in range(2):
        d_tensor[i] = cute.Float64(llvm.extractvalue(T.f64(), result, [i], loc=loc, ip=ip))


# =============================================================================
# FP8 MMA operations (m16n8k32) - SM89+
# =============================================================================


@dsl_user_op
def ptx_mma_m16n8k32_e4m3_e4m3_f32(
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
    """m16n8k32 MMA: e4m3 inputs, f32 accumulator"""
    a_base = cute.recast_ptr(a_ptr + a_offset, dtype=cute.Int32)
    a_tensor = cute.make_tensor(a_base, (4,))
    b_base = cute.recast_ptr(b_ptr + b_offset, dtype=cute.Int32)
    b_tensor = cute.make_tensor(b_base, (2,))
    c_base = c_ptr + c_offset
    c_tensor = cute.make_tensor(c_base, (4,))

    a_vals = [cute.Int32(a_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(4)]
    b_vals = [cute.Int32(b_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(2)]
    c_vals = [cute.Float32(c_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(4)]

    res_type = llvm.StructType.get_literal([T.f32()] * 4)
    ptx_asm = """
    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$10, $11, $12, $13};
    """
    constraints = "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
    operands = a_vals + b_vals + c_vals

    result = llvm.inline_asm(
        res_type,
        operands,
        ptx_asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    d_tensor = cute.make_tensor(c_base, (4,))
    for i in range(4):
        d_tensor[i] = cute.Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))


@dsl_user_op
def ptx_mma_m16n8k32_e4m3_e4m3_f16(
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
    """m16n8k32 MMA: e4m3 inputs, f16 accumulator"""
    a_base = cute.recast_ptr(a_ptr + a_offset, dtype=cute.Int32)
    a_tensor = cute.make_tensor(a_base, (4,))
    b_base = cute.recast_ptr(b_ptr + b_offset, dtype=cute.Int32)
    b_tensor = cute.make_tensor(b_base, (2,))
    c_base = cute.recast_ptr(c_ptr + c_offset, dtype=cute.Int32)
    c_tensor = cute.make_tensor(c_base, (2,))

    a_vals = [cute.Int32(a_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(4)]
    b_vals = [cute.Int32(b_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(2)]
    c_vals = [cute.Int32(c_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(2)]

    res_type = llvm.StructType.get_literal([T.i32()] * 2)
    ptx_asm = """
    mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16
        {$0, $1},
        {$2, $3, $4, $5},
        {$6, $7},
        {$8, $9};
    """
    constraints = "=r,=r,r,r,r,r,r,r,r,r"
    operands = a_vals + b_vals + c_vals

    result = llvm.inline_asm(
        res_type,
        operands,
        ptx_asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    d_tensor = cute.make_tensor(c_base, (2,))
    for i in range(2):
        d_tensor[i] = cute.Int32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))


@dsl_user_op
def ptx_mma_m16n8k32_e5m2_e5m2_f32(
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
    """m16n8k32 MMA: e5m2 inputs, f32 accumulator"""
    a_base = cute.recast_ptr(a_ptr + a_offset, dtype=cute.Int32)
    a_tensor = cute.make_tensor(a_base, (4,))
    b_base = cute.recast_ptr(b_ptr + b_offset, dtype=cute.Int32)
    b_tensor = cute.make_tensor(b_base, (2,))
    c_base = c_ptr + c_offset
    c_tensor = cute.make_tensor(c_base, (4,))

    a_vals = [cute.Int32(a_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(4)]
    b_vals = [cute.Int32(b_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(2)]
    c_vals = [cute.Float32(c_tensor[i]).ir_value(loc=loc, ip=ip) for i in range(4)]

    res_type = llvm.StructType.get_literal([T.f32()] * 4)
    ptx_asm = """
    mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$10, $11, $12, $13};
    """
    constraints = "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
    operands = a_vals + b_vals + c_vals

    result = llvm.inline_asm(
        res_type,
        operands,
        ptx_asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    d_tensor = cute.make_tensor(c_base, (4,))
    for i in range(4):
        d_tensor[i] = cute.Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))


# =============================================================================
# Generic dispatcher function
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
    """
    Generic PTX MMA dispatcher.

    Dispatches to the appropriate specialized MMA function based on
    shape and data types.
    """
    shape = shape.lower()
    a_dtype = a_dtype.lower()
    b_dtype = b_dtype.lower()
    c_dtype = c_dtype.lower()

    # Dispatch based on shape and types
    if shape == "m16n8k16":
        if a_dtype in ["fp16", "f16", "float16"]:
            if c_dtype in ["fp32", "f32", "float32"]:
                return ptx_mma_m16n8k16_f16_f16_f32(a_ptr, a_offset, b_ptr, b_offset, c_ptr, c_offset, a_layout, b_layout)
            elif c_dtype in ["fp16", "f16", "float16"]:
                return ptx_mma_m16n8k16_f16_f16_f16(a_ptr, a_offset, b_ptr, b_offset, c_ptr, c_offset, a_layout, b_layout)
        elif a_dtype in ["bf16", "bfloat16"] and c_dtype in ["fp32", "f32", "float32"]:
            return ptx_mma_m16n8k16_bf16_bf16_f32(a_ptr, a_offset, b_ptr, b_offset, c_ptr, c_offset, a_layout, b_layout)

    elif shape == "m16n8k32":
        if a_dtype in ["int8", "s8"]:
            return ptx_mma_m16n8k32_s8_s8_s32(a_ptr, a_offset, b_ptr, b_offset, c_ptr, c_offset, a_layout, b_layout)
        elif a_dtype in ["uint8", "u8"]:
            return ptx_mma_m16n8k32_u8_u8_s32(a_ptr, a_offset, b_ptr, b_offset, c_ptr, c_offset, a_layout, b_layout)
        elif a_dtype in ["int4", "s4"]:
            return ptx_mma_m16n8k32_s4_s4_s32(a_ptr, a_offset, b_ptr, b_offset, c_ptr, c_offset, a_layout, b_layout)
        elif a_dtype in ["uint4", "u4"]:
            return ptx_mma_m16n8k32_u4_u4_s32(a_ptr, a_offset, b_ptr, b_offset, c_ptr, c_offset, a_layout, b_layout)
        elif a_dtype in ["e4m3", "float8_e4m3", "fp8_e4m3"]:
            if c_dtype in ["fp32", "f32", "float32"]:
                return ptx_mma_m16n8k32_e4m3_e4m3_f32(a_ptr, a_offset, b_ptr, b_offset, c_ptr, c_offset, a_layout, b_layout)
            elif c_dtype in ["fp16", "f16", "float16"]:
                return ptx_mma_m16n8k32_e4m3_e4m3_f16(a_ptr, a_offset, b_ptr, b_offset, c_ptr, c_offset, a_layout, b_layout)
        elif a_dtype in ["e5m2", "float8_e5m2", "fp8_e5m2"] and c_dtype in ["fp32", "f32", "float32"]:
            return ptx_mma_m16n8k32_e5m2_e5m2_f32(a_ptr, a_offset, b_ptr, b_offset, c_ptr, c_offset, a_layout, b_layout)

    elif shape == "m16n8k4":
        # TF32 MMA: accept tf32 or fp32 (TileLang may pass float32 for TF32 GEMM)
        if a_dtype in ["tf32", "tensorfloat32", "fp32", "f32", "float32"]:
            return ptx_mma_m16n8k4_tf32_tf32_f32(a_ptr, a_offset, b_ptr, b_offset, c_ptr, c_offset, a_layout, b_layout)

    elif shape == "m16n8k8":
        # TF32 MMA: accept tf32 or fp32 (e.g. deepseek_mhc)
        if a_dtype in ["tf32", "tensorfloat32", "fp32", "f32", "float32"]:
            return ptx_mma_m16n8k8_tf32_tf32_f32(a_ptr, a_offset, b_ptr, b_offset, c_ptr, c_offset, a_layout, b_layout)

    elif shape == "m8n8k4" and a_dtype in ["fp64", "f64", "float64"]:
        return ptx_mma_m8n8k4_f64_f64_f64(a_ptr, a_offset, b_ptr, b_offset, c_ptr, c_offset, a_layout, b_layout)

    raise ValueError(f"Unsupported MMA configuration: shape={shape}, a_dtype={a_dtype}, b_dtype={b_dtype}, c_dtype={c_dtype}")

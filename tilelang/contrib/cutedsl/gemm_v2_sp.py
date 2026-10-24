"""Sparse (2:4 structured) WGMMA helpers for the CuTeDSL backend.

CuTeDSL has no sparse WGMMA primitive, so the ``wgmma.mma_async.sp`` family is
emitted as inline PTX here. The operand layout mirrors the
``cute::SM90::GMMA::SPARSE::GMMA_*_SS``/``_RS`` functors in
``cute/arch/mma_sm90_gmma_sparse.hpp``: the accumulator registers are the tied
D/C operands, ``e`` is the per-instruction 32-bit metadata register taken from
the E fragment, and the sparse selector is an immediate.
"""

__all__ = [
    "wgmma_sp_ss",
    "wgmma_sp_rs",
]

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass._mlir import ir
from cutlass.cutlass_dsl import Constexpr

from .gemm_v2 import (
    GmmaDescriptor,
    _PTX_DTYPE_MAP,
    _wgmma_ab_dtype,
    _wgmma_num_c_regs,
)


def _sparse_tail(ptx_a, num_a_regs, scale_a, scale_b, tnsp_a, tnsp_b):
    """Trailing immediates of ``wgmma.mma_async.sp`` for one dtype form.

    Only the f16/bf16 grammars take transpose immediates; tf32, fp8 and the
    integer forms carry just the two scale immediates (see
    ``cute/arch/mma_sm90_gmma_sparse.hpp``). The RS form drops trans-a because
    sparse register A is always K-major.
    """
    if ptx_a not in ("f16", "bf16"):
        return f"p, {scale_a}, {scale_b};"
    tnsp_a_imm = 1 if tnsp_a else 0
    tnsp_b_imm = 1 if tnsp_b else 0
    if num_a_regs:
        return f"p, {scale_a}, {scale_b}, {tnsp_b_imm};"
    return f"p, {scale_a}, {scale_b}, {tnsp_a_imm}, {tnsp_b_imm};"


def _wgmma_sp_asm(shape, ptx_a, ptx_b, ptx_c, num_c_regs, num_a_regs, tnsp_a, tnsp_b, scale_a, scale_b, spsel, scale_out):
    """Build the ``wgmma.mma_async.sp`` asm string.

    Operand numbering: the num_c_regs write-only D outputs come first (their
    slack inputs follow as tied operands), then the A registers (RS only) or the
    A descriptor (SS only), the B descriptor, and the metadata register. The
    sparse selector, scale-out predicate value and the scale/transpose flags are
    immediates baked into the string.
    """
    d_regs = ", ".join(f"${i}" for i in range(num_c_regs))
    # The accumulators are declared write-only and tied to slack inputs, so the
    # remaining operands start after both groups (see wgmma_rs).
    idx = 2 * num_c_regs
    a_operand = ""
    if num_a_regs:
        a_regs = ", ".join(f"${idx + i}" for i in range(num_a_regs))
        a_operand = f"{{{a_regs}}}, "
        idx += num_a_regs
    else:
        a_operand = f"${idx}, "
        idx += 1
    desc_b_operand = f"${idx}"
    idx += 1
    e_operand = f"${idx}"

    tail = _sparse_tail(ptx_a, num_a_regs, scale_a, scale_b, tnsp_a, tnsp_b)

    return (
        "{\n"
        ".reg .pred p;\n"
        f"setp.ne.b32 p, {1 if scale_out != 0 else 0}, 0;\n"
        f"wgmma.mma_async.sp.sync.aligned.{shape}.{ptx_c}.{ptx_a}.{ptx_b} "
        f"{{{d_regs}}}, "
        f"{a_operand}"
        f"{desc_b_operand}, "
        f"{e_operand}, {spsel}, "
        f"{tail}\n"
        "}\n"
    )


@cute.jit
def wgmma_sp_ss(
    A_dtype: str,
    B_dtype: str,
    C_dtype: str,
    M: Constexpr[int],
    N: Constexpr[int],
    K: Constexpr[int],
    tnspA: Constexpr[bool],
    tnspB: Constexpr[bool],
    scaleA: Constexpr[int],
    scaleB: Constexpr[int],
    spsel: Constexpr[int],
    desc_a: GmmaDescriptor,
    desc_b: GmmaDescriptor,
    C_ptr: cute.Pointer,
    scale_out: Constexpr[int],
    E_ptr: cute.Pointer,
    e_offset,
):
    """Sparse WGMMA shared-shared: A and B come from shared memory descriptors."""
    num_c_regs = _wgmma_num_c_regs(M, N, C_dtype)
    ptx_a = _PTX_DTYPE_MAP[_wgmma_ab_dtype(A_dtype)]
    ptx_b = _PTX_DTYPE_MAP[_wgmma_ab_dtype(B_dtype)]
    ptx_c = _PTX_DTYPE_MAP[C_dtype]

    asm_str = _wgmma_sp_asm(f"m{M}n{N}k{K}", ptx_a, ptx_b, ptx_c, num_c_regs, 0, tnspA, tnspB, scaleA, scaleB, spsel, scale_out)

    is_int_accum = ptx_c == "s32"
    c_constraint = "r" if is_int_accum else "f"
    i32_type = ir.IntegerType.get_signless(32)
    f32_type = ir.F32Type.get()
    c_ir_type = i32_type if is_int_accum else f32_type

    C_vecs = cute.make_tensor(cute.recast_ptr(C_ptr, dtype=cute.Int32), (num_c_regs,))
    # The metadata is one 32-bit register per instruction; read it through the
    # E fragment pointer the same way the CUDA backend does.
    E_vec = cute.make_tensor(cute.recast_ptr(E_ptr + e_offset, dtype=cute.Int32), (1,))

    operands = []
    for i in cutlass.range_constexpr(num_c_regs):
        val = C_vecs[i].ir_value()
        operands.append(val if is_int_accum else llvm.bitcast(f32_type, val))
    operands.append(desc_a.desc_i64[0].ir_value())
    operands.append(desc_b.desc_i64[0].ir_value())
    operands.append(E_vec[0].ir_value())

    constraints = ",".join([f"={c_constraint}"] * num_c_regs) + "," + ",".join([str(i) for i in range(num_c_regs)]) + ",l,l,r"

    result = llvm.inline_asm(
        llvm.StructType.get_literal([c_ir_type] * num_c_regs),
        operands,
        asm_str,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

    for i in cutlass.range_constexpr(num_c_regs):
        extracted = llvm.extractvalue(c_ir_type, result, [i])
        C_vecs[i] = extracted if is_int_accum else llvm.bitcast(i32_type, extracted)


@cute.jit
def wgmma_sp_rs(
    A_dtype: str,
    B_dtype: str,
    C_dtype: str,
    M: Constexpr[int],
    N: Constexpr[int],
    K: Constexpr[int],
    tnspB: Constexpr[bool],
    scaleA: Constexpr[int],
    scaleB: Constexpr[int],
    spsel: Constexpr[int],
    A_ptr: cute.Pointer,
    desc_b: GmmaDescriptor,
    C_ptr: cute.Pointer,
    scale_out: Constexpr[int],
    E_ptr: cute.Pointer,
    e_offset,
):
    """Sparse WGMMA register-shared: A from registers, B from a descriptor.

    Sparse register A is always K-major, so it takes no trans-a operand.
    """
    num_c_regs = _wgmma_num_c_regs(M, N, C_dtype)
    num_a_regs = 4  # Always 4 for M=64, all supported dtypes
    ptx_a = _PTX_DTYPE_MAP[_wgmma_ab_dtype(A_dtype)]
    ptx_b = _PTX_DTYPE_MAP[_wgmma_ab_dtype(B_dtype)]
    ptx_c = _PTX_DTYPE_MAP[C_dtype]

    asm_str = _wgmma_sp_asm(f"m{M}n{N}k{K}", ptx_a, ptx_b, ptx_c, num_c_regs, num_a_regs, False, tnspB, scaleA, scaleB, spsel, scale_out)

    is_int_accum = ptx_c == "s32"
    c_constraint = "r" if is_int_accum else "f"
    i32_type = ir.IntegerType.get_signless(32)
    f32_type = ir.F32Type.get()
    c_ir_type = i32_type if is_int_accum else f32_type

    A_vecs = cute.make_tensor(cute.recast_ptr(A_ptr, dtype=cute.Int32), (num_a_regs,))
    C_vecs = cute.make_tensor(cute.recast_ptr(C_ptr, dtype=cute.Int32), (num_c_regs,))
    E_vec = cute.make_tensor(cute.recast_ptr(E_ptr + e_offset, dtype=cute.Int32), (1,))

    operands = []
    for i in cutlass.range_constexpr(num_c_regs):
        val = C_vecs[i].ir_value()
        operands.append(val if is_int_accum else llvm.bitcast(f32_type, val))
    for i in cutlass.range_constexpr(num_a_regs):
        operands.append(A_vecs[i].ir_value())
    operands.append(desc_b.desc_i64[0].ir_value())
    operands.append(E_vec[0].ir_value())

    constraints = (
        ",".join([f"={c_constraint}"] * num_c_regs)
        + ","
        + ",".join([str(i) for i in range(num_c_regs)])
        + ","
        + ",".join(["r"] * num_a_regs)
        + ",l,r"
    )

    result = llvm.inline_asm(
        llvm.StructType.get_literal([c_ir_type] * num_c_regs),
        operands,
        asm_str,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

    for i in cutlass.range_constexpr(num_c_regs):
        extracted = llvm.extractvalue(c_ir_type, result, [i])
        C_vecs[i] = extracted if is_int_accum else llvm.bitcast(i32_type, extracted)

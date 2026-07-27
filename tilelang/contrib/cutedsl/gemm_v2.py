__all__ = [
    "GmmaDescriptor",
    "initialize_wgmma_descriptor",
    "increase_descriptor_offset",
    "warpgroup_fence_operand",
    "warpgroup_arrive",
    "warpgroup_commit_batch",
    "warpgroup_wait",
    "wgmma_ss",
    "wgmma_rs",
]

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import nvvm, llvm
from cutlass._mlir import ir
from cutlass.cutlass_dsl import T, Constexpr
import cutlass.cute.nvgpu.warpgroup as warpgroup
from .utils import type_map


class GmmaDescriptor:
    def __init__(self, desc_64: cute.Int64 = None):
        self.desc = cute.make_rmem_tensor((2,), dtype=cutlass.Int32)
        self.desc_i64 = cute.make_tensor(cute.recast_ptr(self.desc.iterator, dtype=cute.Int64), (1,))
        if desc_64 is not None:
            self.desc_i64[0] = desc_64

    def __add__(self, offset):
        res = cute.make_rmem_tensor((2,), dtype=cutlass.Int32)
        res_i64 = cute.make_tensor(cute.recast_ptr(res.iterator, dtype=cute.Int64), (1,))
        res[0] = self.desc[0] + offset
        res[1] = self.desc[1]
        return GmmaDescriptor(res_i64[0])


def initialize_wgmma_descriptor(layout_type, leading_byte_offset, stride_byte_offset, desc: GmmaDescriptor, start_address: cute.Pointer):
    # Manually pack the descriptor bits to match the WGMMA descriptor format:
    #   Bits [0:13]  = start_address >> 4
    #   Bits [16:29] = leading_byte_offset
    #   Bits [32:45] = stride_byte_offset
    #   Bits [49:51] = base_offset (0)
    #   Bits [62:63] = layout_type
    ptr_val = start_address.toint() >> 4
    # Low 32 bits: start_address[0:13] | leading[16:29]
    desc.desc[0] = cutlass.Int32(ptr_val) | cutlass.Int32(cutlass.Int32(leading_byte_offset) << 16)
    # High 32 bits: stride[0:13] | layout_type[30:31]
    desc.desc[1] = cutlass.Int32(stride_byte_offset) | cutlass.Int32(cutlass.Int32(layout_type) << 30)


def increase_descriptor_offset(desc: GmmaDescriptor, offset):
    desc.desc[0] += offset >> 4


def warpgroup_fence_operand(*args):
    # No-op in CuTeDSL: warpgroup synchronization is handled by CUTLASS cuTe
    # primitives (warpgroup.fence(), warpgroup.commit_group(), etc.) and by the
    # has_side_effects=True flag on inline PTX asm in wgmma_rs/wgmma_ss.
    # The codegen emits calls to this (see codegen_cutedsl.cc) but they are
    # intentionally empty.
    pass


def warpgroup_arrive():
    warpgroup.fence()


def warpgroup_commit_batch():
    warpgroup.commit_group()


def warpgroup_wait(N):
    warpgroup.wait_group(N)


# PTX dtype suffix mapping for WGMMA instructions
_PTX_DTYPE_MAP = {
    "fp16": "f16",
    "bf16": "bf16",
    "tf32": "tf32",
    "fp32": "f32",
    "e4m3": "e4m3",
    "e5m2": "e5m2",
    "s8": "s8",
    "u8": "u8",
    "float16": "f16",
    "bfloat16": "bf16",
    "float32": "f32",
    "float8_e4m3": "e4m3",
    "float8_e4m3fn": "e4m3",
    "float8_e5m2": "e5m2",
    "int8": "s8",
    "uint8": "u8",
}

# For WGMMA A/B operands, fp32 must be treated as tf32 on SM90
_FP32_TO_TF32 = {"fp32": "tf32", "f32": "tf32", "float32": "tf32"}

# Canonical PTX dtype -> cutlass scalar type (for output dtypes only).
_PTX_TO_CUTLASS_TYPE = {
    "f16": cutlass.Float16,
    "bf16": cutlass.BFloat16,
    "f32": cutlass.Float32,
    "s32": cutlass.Int32,
}


def _wgmma_num_c_regs(M, N, C_dtype):
    """Number of i32 result registers per thread for a WGMMA op.

    Each i32 register holds ``32 // elem_bits`` packed elements.
    """
    canonical = _PTX_DTYPE_MAP.get(C_dtype, C_dtype)
    elem_bits = _PTX_TO_CUTLASS_TYPE[canonical].width
    return M * N * elem_bits // (128 * 32)


def _wgmma_ab_dtype(dtype_str):
    """Map A/B operand dtype for WGMMA: fp32 -> tf32 (SM90 compatibility)."""
    return _FP32_TO_TF32.get(dtype_str, dtype_str)


@cute.jit
def wgmma_ss(
    A_dtype: str,
    B_dtype: str,
    C_dtype: str,
    M: Constexpr[int],
    N: Constexpr[int],
    K: Constexpr[int],
    tnspA: bool,
    tnspB: bool,
    scaleA: int,
    scaleB: int,
    desc_a: GmmaDescriptor,
    desc_b: GmmaDescriptor,
    C_ptr: cute.Pointer,
    scale_out: Constexpr[int],
):
    num_elems_per_thread = _wgmma_num_c_regs(M, N, C_dtype)

    C_types = llvm.StructType.get_literal([T.i32()] * num_elems_per_thread)

    C_vecs = cute.make_tensor(cute.recast_ptr(C_ptr, dtype=cute.Int32), (num_elems_per_thread,))

    # Pack current accumulator values into a struct
    inouts_struct = llvm.mlir_undef(C_types)
    for i in cutlass.range_constexpr(num_elems_per_thread):
        inouts_struct = llvm.insertvalue(inouts_struct, C_vecs[i].ir_value(), [i])

    shape_attr = ir.Attribute.parse(f"#nvvm.shape<m={M},n={N},k={K}>")

    new_C_vecs = nvvm.wgmma_mma_async(
        results_=C_types,
        inouts=inouts_struct,
        descriptor_a=desc_a.desc_i64[0].ir_value(),
        descriptor_b=desc_b.desc_i64[0].ir_value(),
        shape=shape_attr,
        type_a=type_map[_wgmma_ab_dtype(A_dtype)],
        type_b=type_map[_wgmma_ab_dtype(B_dtype)],
        type_d=type_map[C_dtype],
        scale_d=nvvm.WGMMAScaleOut.zero if scale_out == 0 else nvvm.WGMMAScaleOut.one,
        scale_a=nvvm.WGMMAScaleIn.one if scaleA == 1 else nvvm.WGMMAScaleIn.neg,
        scale_b=nvvm.WGMMAScaleIn.one if scaleB == 1 else nvvm.WGMMAScaleIn.neg,
        layout_a=nvvm.MMALayout.col if tnspA else nvvm.MMALayout.row,
        layout_b=nvvm.MMALayout.row if tnspB else nvvm.MMALayout.col,
    )
    for i in cutlass.range_constexpr(num_elems_per_thread):
        C_vecs[i] = llvm.extractvalue(T.i32(), new_C_vecs, [i])


@cute.jit
def wgmma_rs(
    A_dtype: str,
    B_dtype: str,
    C_dtype: str,
    M: Constexpr[int],
    N: Constexpr[int],
    K: Constexpr[int],
    tnspB: Constexpr[bool],
    scaleA: Constexpr[int],
    scaleB: Constexpr[int],
    A_ptr: cute.Pointer,
    desc_b: GmmaDescriptor,
    C_ptr: cute.Pointer,
    scale_out: Constexpr[int],
):
    """WGMMA register-shared variant using PTX inline asm.

    A operand comes from registers, B from shared memory descriptor.
    M is always 64. A is always K-major (not transposed).
    """
    num_a_regs = 4  # Always 4 for M=64, all supported dtypes
    num_c_regs = _wgmma_num_c_regs(M, N, C_dtype)

    ptx_a = _PTX_DTYPE_MAP[_wgmma_ab_dtype(A_dtype)]
    ptx_b = _PTX_DTYPE_MAP[_wgmma_ab_dtype(B_dtype)]
    ptx_c = _PTX_DTYPE_MAP[C_dtype]

    # Create tensor views over register data
    A_vecs = cute.make_tensor(cute.recast_ptr(A_ptr, dtype=cute.Int32), (num_a_regs,))
    C_vecs = cute.make_tensor(cute.recast_ptr(C_ptr, dtype=cute.Int32), (num_c_regs,))

    # Operand numbering in the inline asm:
    #   $0 .. $(num_c_regs-1)                          : output D regs (f32)
    #   $(num_c_regs) .. $(2*num_c_regs-1)             : tied C inputs
    #   $(2*num_c_regs) .. $(2*num_c_regs+3)           : A regs (i32)
    #   $(2*num_c_regs+4)                              : desc_b (i64)
    a_base = 2 * num_c_regs
    desc_b_idx = a_base + num_a_regs

    # Build PTX asm string
    d_regs_str = ", ".join(f"${i}" for i in range(num_c_regs))
    a_regs_str = ", ".join(f"${a_base + i}" for i in range(num_a_regs))
    tnsp_b_imm = 1 if tnspB else 0
    # Embed scale_out as immediate in predicate setup
    scale_const = 1 if scale_out != 0 else 0

    # TF32 WGMMA does not have a tnspB parameter (B is always K-major)
    tail_args = f"p, {scaleA}, {scaleB};" if ptx_a == "tf32" else f"p, {scaleA}, {scaleB}, {tnsp_b_imm};"

    asm_str = (
        "{\n"
        ".reg .pred p;\n"
        f"setp.ne.b32 p, {scale_const}, 0;\n"
        f"wgmma.mma_async.sync.aligned.m{M}n{N}k{K}"
        f".{ptx_c}.{ptx_a}.{ptx_b} "
        f"{{{d_regs_str}}}, "
        f"{{{a_regs_str}}}, "
        f"${desc_b_idx}, "
        f"{tail_args}\n"
        "}\n"
    )

    # Determine if C/D is float or integer based on canonical PTX dtype.
    # All type/constraint decisions are made here (no branching in the DSL loop).
    is_int_accum = ptx_c == "s32"
    c_constraint = "r" if is_int_accum else "f"
    i32_type = ir.IntegerType.get_signless(32)
    f32_type = ir.F32Type.get()
    c_ir_type = i32_type if is_int_accum else f32_type

    # Build constraint string
    out_constraints = ",".join([f"={c_constraint}"] * num_c_regs)
    tied_constraints = ",".join([str(i) for i in range(num_c_regs)])
    a_constraints = ",".join(["r"] * num_a_regs)
    constraints = f"{out_constraints},{tied_constraints},{a_constraints},l"

    # Prepare operands list
    operands = []

    # Tied C inputs: for 'f' constraint bitcast i32→f32, for 'r' pass i32 as-is
    for i in cutlass.range_constexpr(num_c_regs):
        val = C_vecs[i].ir_value()
        operands.append(val if is_int_accum else llvm.bitcast(f32_type, val))

    # A inputs (i32)
    for i in cutlass.range_constexpr(num_a_regs):
        operands.append(A_vecs[i].ir_value())

    # desc_b (i64)
    operands.append(desc_b.desc_i64[0].ir_value())

    # Result type
    result_type = llvm.StructType.get_literal([c_ir_type] * num_c_regs)

    # Execute inline asm
    result = llvm.inline_asm(
        result_type,
        operands,
        asm_str,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

    # Extract results and store back: for 'f' bitcast f32→i32, for 'r' direct
    for i in cutlass.range_constexpr(num_c_regs):
        extracted = llvm.extractvalue(c_ir_type, result, [i])
        C_vecs[i] = extracted if is_int_accum else llvm.bitcast(i32_type, extracted)

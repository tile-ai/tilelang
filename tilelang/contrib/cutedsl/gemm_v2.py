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
from cutlass._mlir.dialects import llvm
from cutlass._mlir import ir
from cutlass.cutlass_dsl import Constexpr
import cutlass.cute.nvgpu.warpgroup as warpgroup


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


def initialize_wgmma_descriptor(layout_type, leading_byte_offset, stride_byte_offset, desc: GmmaDescriptor, start_address):
    # Manually pack the descriptor bits to match the WGMMA descriptor format:
    #   Bits [0:13]  = start_address >> 4
    #   Bits [16:29] = leading_byte_offset
    #   Bits [32:45] = stride_byte_offset
    #   Bits [49:51] = base_offset (0)
    #   Bits [62:63] = layout_type
    ptr = start_address.iterator if isinstance(start_address, cute.Tensor) else start_address
    ptr_val = ptr.toint() >> 4
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


def wgmma_ss(
    A_dtype, B_dtype, C_dtype, M, N, K, tnspA, tnspB, scaleA, scaleB, desc_a, desc_b, C_ptr, scale_out, E_ptr=None, sparse_selector=0
):
    return _wgmma(
        A_dtype, B_dtype, C_dtype, M, N, K, tnspA, tnspB, scaleA, scaleB, desc_a, desc_b, C_ptr, scale_out, E_ptr, sparse_selector
    )


def wgmma_rs(A_dtype, B_dtype, C_dtype, M, N, K, tnspB, scaleA, scaleB, A_ptr, desc_b, C_ptr, scale_out, E_ptr=None, sparse_selector=0):
    return _wgmma(A_dtype, B_dtype, C_dtype, M, N, K, False, tnspB, scaleA, scaleB, A_ptr, desc_b, C_ptr, scale_out, E_ptr, sparse_selector)


@cute.jit
def _wgmma(
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
    A_ptr,
    desc_b: GmmaDescriptor,
    C_ptr: cute.Pointer,
    scale_out,
    E_ptr=None,
    sparse_selector: Constexpr[int] = 0,
):
    """Shared PTX emission keeps scale-out a runtime predicate for both variants."""
    is_ss = isinstance(A_ptr, GmmaDescriptor)
    num_a_regs = 1 if is_ss else 4
    num_c_regs = _wgmma_num_c_regs(M, N, C_dtype)

    ptx_a = _PTX_DTYPE_MAP[_wgmma_ab_dtype(A_dtype)]
    ptx_b = _PTX_DTYPE_MAP[_wgmma_ab_dtype(B_dtype)]
    ptx_c = _PTX_DTYPE_MAP[C_dtype]

    # Create tensor views over register data
    if cutlass.const_expr(not is_ss):
        A_vecs = cute.make_tensor(cute.recast_ptr(A_ptr, dtype=cute.Int32), (num_a_regs,))
    C_vecs = cute.make_tensor(cute.recast_ptr(C_ptr, dtype=cute.Int32), (num_c_regs,))

    a_base = 2 * num_c_regs
    desc_b_idx = a_base + num_a_regs
    scale_idx = desc_b_idx + 1

    d_regs_str = ", ".join(f"${i}" for i in range(num_c_regs))
    a_regs_str = f"${a_base}" if is_ss else "{" + ", ".join(f"${a_base + i}" for i in range(num_a_regs)) + "}"
    sparse = E_ptr is not None
    tail_args = f"${scale_idx + 1}, {sparse_selector}, p" if sparse else "p"
    if cutlass.const_expr(ptx_a not in ("s8", "u8")):
        tail_args += f", {scaleA}, {scaleB}"
    if cutlass.const_expr(ptx_a in ("f16", "bf16")):
        if cutlass.const_expr(is_ss):
            tail_args += f", {int(tnspA)}"
        tail_args += f", {int(tnspB)}"

    asm_str = (
        "{\n"
        ".reg .pred p;\n"
        f"setp.ne.b32 p, ${scale_idx}, 0;\n"
        f"wgmma.mma_async{'.sp' if sparse else ''}.sync.aligned.m{M}n{N}k{K}"
        f".{ptx_c}.{ptx_a}.{ptx_b} "
        f"{{{d_regs_str}}}, "
        f"{a_regs_str}, "
        f"${desc_b_idx}, "
        f"{tail_args};\n"
        "}\n"
    )

    # Integer and packed FP16 accumulators use i32 registers.
    use_float_regs = ptx_c == "f32"
    c_constraint = "f" if use_float_regs else "r"
    i32_type = ir.IntegerType.get_signless(32)
    f32_type = ir.F32Type.get()
    c_ir_type = f32_type if use_float_regs else i32_type

    # Build constraint string
    out_constraints = ",".join([f"={c_constraint}"] * num_c_regs)
    tied_constraints = ",".join([str(i) for i in range(num_c_regs)])
    a_constraints = "l" if is_ss else ",".join(["r"] * num_a_regs)
    constraints = f"{out_constraints},{tied_constraints},{a_constraints},l,r"

    if cutlass.const_expr(sparse):
        constraints += ",r"

    # Prepare operands list
    operands = []

    # Tied C inputs: for 'f' constraint bitcast i32→f32, for 'r' pass i32 as-is
    for i in cutlass.range_constexpr(num_c_regs):
        val = C_vecs[i].ir_value()
        operands.append(llvm.bitcast(f32_type, val) if use_float_regs else val)

    if cutlass.const_expr(is_ss):
        operands.append(A_ptr.desc_i64[0].ir_value())
    else:
        for i in cutlass.range_constexpr(num_a_regs):
            operands.append(A_vecs[i].ir_value())

    # desc_b (i64)
    operands.append(desc_b.desc_i64[0].ir_value())
    operands.append(cutlass.Int32(scale_out).ir_value())

    if cutlass.const_expr(sparse):
        metadata = cute.make_tensor(cute.recast_ptr(E_ptr, dtype=cute.Int32), (1,))
        operands.append(metadata[0].ir_value())

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
        C_vecs[i] = llvm.bitcast(i32_type, extracted) if use_float_regs else extracted

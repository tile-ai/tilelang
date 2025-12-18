"""
LDMATRIX and STMATRIX operations for CuTeDSL backend.
Based on tl_templates/cuda/ldsm.h

These functions provide wrappers around PTX ldmatrix/stmatrix instructions
for loading/storing 8x8 matrix fragments between shared memory and registers.
"""

from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import nvvm, llvm
from cutlass._mlir import ir  # noqa: F401
from cutlass.cute.typing import Pointer, Int32  # noqa: F401
import cutlass.cute as cute


def _to_ir_value(v, loc=None, ip=None):
    """Convert value to MLIR IR, handling both cutlass types and raw MLIR Values"""
    if hasattr(v, "ir_value"):
        return v.ir_value(loc=loc, ip=ip)
    else:
        # Already an MLIR Value
        return v


def _ldmatrix(smem_ptr, local_ptr, num, transpose, loc=None, ip=None):
    """Internal helper for ldmatrix operations"""
    layout = nvvm.MMALayout.col if transpose else nvvm.MMALayout.row
    assert num in [2, 4]
    ret_type = llvm.StructType.get_literal([T.i32()] * num)
    out_i32 = nvvm.ldmatrix(ret_type, smem_ptr.llvm_ptr, num=num, layout=layout, loc=loc, ip=ip)
    out = cute.make_tensor(cute.recast_ptr(local_ptr, dtype=cute.Int32), num)
    for i in range(num):
        out[i] = cute.Int32(llvm.extractvalue(T.i32(), out_i32, [i], loc=loc, ip=ip))


def _stmatrix(smem_ptr, values, transpose, loc=None, ip=None):
    """Internal helper for stmatrix operations"""
    layout = nvvm.MMALayout.col if transpose else nvvm.MMALayout.row
    ir_values = [_to_ir_value(v, loc, ip) for v in values]
    nvvm.stmatrix(smem_ptr.llvm_ptr, ir_values, layout=layout, loc=loc, ip=ip)


# ============================================================================
# LDMATRIX operations (load from shared memory to registers)
# ============================================================================


@dsl_user_op
def ptx_ldmatrix_x1(smem_ptr: Pointer, local_ptr: Pointer, *, loc=None, ip=None) -> None:
    """Load 1 matrix (8x8) from shared memory"""
    # _ldmatrix(smem_ptr, local_ptr, 1, False, loc, ip)
    out_i32 = nvvm.ldmatrix(T.i32(), smem_ptr.llvm_ptr, num=1, layout=nvvm.MMALayout.row, loc=loc, ip=ip)
    out = cute.make_tensor(cute.recast_ptr(local_ptr, dtype=cute.Int32), 1)
    out[0] = cute.Int32(out_i32)


@dsl_user_op
def ptx_ldmatrix_x2(smem_ptr: Pointer, local_ptr: Pointer, *, loc=None, ip=None) -> None:
    """Load 2 matrices (8x8 each) from shared memory"""
    _ldmatrix(smem_ptr, local_ptr, 2, False, loc, ip)


@dsl_user_op
def ptx_ldmatrix_x4(smem_ptr: Pointer, local_ptr: Pointer, *, loc=None, ip=None) -> None:
    """Load 4 matrices (8x8 each) from shared memory"""
    _ldmatrix(smem_ptr, local_ptr, 4, False, loc, ip)


@dsl_user_op
def ptx_ldmatrix_x1_trans(smem_ptr: Pointer, local_ptr: Pointer, *, loc=None, ip=None) -> None:
    """Load 1 matrix (8x8) with transpose from shared memory"""
    out_i32 = nvvm.ldmatrix(T.i32(), smem_ptr.llvm_ptr, num=1, layout=nvvm.MMALayout.col, loc=loc, ip=ip)
    out = cute.make_tensor(cute.recast_ptr(local_ptr, dtype=cute.Int32), 1)
    out[0] = cute.Int32(out_i32)


@dsl_user_op
def ptx_ldmatrix_x2_trans(smem_ptr: Pointer, local_ptr: Pointer, *, loc=None, ip=None) -> None:
    """Load 2 matrices (8x8 each) with transpose from shared memory"""
    _ldmatrix(smem_ptr, local_ptr, 2, True, loc, ip)


@dsl_user_op
def ptx_ldmatrix_x4_trans(smem_ptr: Pointer, local_ptr: Pointer, *, loc=None, ip=None) -> None:
    """Load 4 matrices (8x8 each) with transpose from shared memory"""
    _ldmatrix(smem_ptr, local_ptr, 4, True, loc, ip)


# ============================================================================
# STMATRIX operations (store from registers to shared memory)
# ============================================================================


@dsl_user_op
def ptx_stmatrix_x1(smem_ptr: Pointer, value0, *, loc=None, ip=None) -> None:
    """Store 1 matrix (8x8) to shared memory"""
    _stmatrix(smem_ptr, [value0], False, loc, ip)


@dsl_user_op
def ptx_stmatrix_x2(smem_ptr: Pointer, value0, value1, *, loc=None, ip=None) -> None:
    """Store 2 matrices (8x8 each) to shared memory"""
    _stmatrix(smem_ptr, [value0, value1], False, loc, ip)


@dsl_user_op
def ptx_stmatrix_x4(smem_ptr: Pointer, value0, value1, value2, value3, *, loc=None, ip=None) -> None:
    """Store 4 matrices (8x8 each) to shared memory"""
    _stmatrix(smem_ptr, [value0, value1, value2, value3], False, loc, ip)


@dsl_user_op
def ptx_stmatrix_x1_trans(smem_ptr: Pointer, value0, *, loc=None, ip=None) -> None:
    """Store 1 matrix (8x8) with transpose to shared memory"""
    _stmatrix(smem_ptr, [value0], True, loc, ip)


@dsl_user_op
def ptx_stmatrix_x2_trans(smem_ptr: Pointer, value0, value1, *, loc=None, ip=None) -> None:
    """Store 2 matrices (8x8 each) with transpose to shared memory"""
    _stmatrix(smem_ptr, [value0, value1], True, loc, ip)


@dsl_user_op
def ptx_stmatrix_x4_trans(smem_ptr: Pointer, value0, value1, value2, value3, *, loc=None, ip=None) -> None:
    """Store 4 matrices (8x8 each) with transpose to shared memory"""
    _stmatrix(smem_ptr, [value0, value1, value2, value3], True, loc, ip)

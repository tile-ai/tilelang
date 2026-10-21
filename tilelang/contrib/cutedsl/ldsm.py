"""
LDMATRIX and STMATRIX operations for CuTeDSL backend.
Based on tl_templates/cuda/ldsm.h

These functions provide wrappers around CUTLASS primitive ldmatrix/stmatrix operations
for loading/storing 8x8 matrix fragments between shared memory and registers.
"""

__all__ = [
    "ldmatrix_x1",
    "ldmatrix_x2",
    "ldmatrix_x4",
    "ldmatrix_x1_trans",
    "ldmatrix_x2_trans",
    "ldmatrix_x4_trans",
    "stmatrix_x1",
    "stmatrix_x2",
    "stmatrix_x4",
    "stmatrix_x1_trans",
    "stmatrix_x2_trans",
    "stmatrix_x4_trans",
    "ptx_ldmatrix_x1",
    "ptx_ldmatrix_x2",
    "ptx_ldmatrix_x4",
    "ptx_ldmatrix_x1_trans",
    "ptx_ldmatrix_x2_trans",
    "ptx_ldmatrix_x4_trans",
    "ptx_stmatrix_x1",
    "ptx_stmatrix_x2",
    "ptx_stmatrix_x4",
    "ptx_stmatrix_x1_trans",
    "ptx_stmatrix_x2_trans",
    "ptx_stmatrix_x4_trans",
]

from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims
from cutlass._mlir import ir  # noqa: F401
from cutlass.cute.typing import Pointer, Int32  # noqa: F401
import cutlass.cute as cute


def _ldmatrix(smem_ptr, local_ptr, num, transpose, loc=None, ip=None):
    """Internal helper for ldmatrix operations"""
    layout = prims.MMALayout.COL if transpose else prims.MMALayout.ROW
    assert num in [1, 2, 4]
    ptr = smem_ptr.llvm_ptr if hasattr(smem_ptr, "llvm_ptr") else smem_ptr
    out_i32 = prims.ldmatrix(ptr, num=num, layout=layout, loc=loc, ip=ip)
    out = cute.make_tensor(cute.recast_ptr(local_ptr, dtype=cute.Int32), num)
    if num == 1:
        out[0] = cute.Int32(out_i32)
    else:
        for i in range(num):
            out[i] = cute.Int32(out_i32[i])


def _stmatrix(smem_ptr, values, transpose, loc=None, ip=None):
    """Internal helper for stmatrix operations"""
    layout = prims.MMALayout.COL if transpose else prims.MMALayout.ROW
    ptr = smem_ptr.llvm_ptr if hasattr(smem_ptr, "llvm_ptr") else smem_ptr
    num = len(values)
    assert num in [1, 2, 4]
    prims.stmatrix(ptr, values, layout, loc=loc, ip=ip)


# ============================================================================
# LDMATRIX operations (load from shared memory to registers)
# ============================================================================


@dsl_user_op
def ldmatrix_x1(smem_ptr: Pointer, local_ptr: Pointer, *, loc=None, ip=None) -> None:
    """Load 1 matrix (8x8) from shared memory"""
    _ldmatrix(smem_ptr, local_ptr, 1, False, loc, ip)


@dsl_user_op
def ldmatrix_x2(smem_ptr: Pointer, local_ptr: Pointer, *, loc=None, ip=None) -> None:
    """Load 2 matrices (8x8 each) from shared memory"""
    _ldmatrix(smem_ptr, local_ptr, 2, False, loc, ip)


@dsl_user_op
def ldmatrix_x4(smem_ptr: Pointer, local_ptr: Pointer, *, loc=None, ip=None) -> None:
    """Load 4 matrices (8x8 each) from shared memory"""
    _ldmatrix(smem_ptr, local_ptr, 4, False, loc, ip)


@dsl_user_op
def ldmatrix_x1_trans(smem_ptr: Pointer, local_ptr: Pointer, *, loc=None, ip=None) -> None:
    """Load 1 matrix (8x8) with transpose from shared memory"""
    _ldmatrix(smem_ptr, local_ptr, 1, True, loc, ip)


@dsl_user_op
def ldmatrix_x2_trans(smem_ptr: Pointer, local_ptr: Pointer, *, loc=None, ip=None) -> None:
    """Load 2 matrices (8x8 each) with transpose from shared memory"""
    _ldmatrix(smem_ptr, local_ptr, 2, True, loc, ip)


@dsl_user_op
def ldmatrix_x4_trans(smem_ptr: Pointer, local_ptr: Pointer, *, loc=None, ip=None) -> None:
    """Load 4 matrices (8x8 each) with transpose from shared memory"""
    _ldmatrix(smem_ptr, local_ptr, 4, True, loc, ip)


# ============================================================================
# STMATRIX operations (store from registers to shared memory)
# ============================================================================


@dsl_user_op
def stmatrix_x1(smem_ptr: Pointer, value0, *, loc=None, ip=None) -> None:
    """Store 1 matrix (8x8) to shared memory"""
    _stmatrix(smem_ptr, [value0], False, loc, ip)


@dsl_user_op
def stmatrix_x2(smem_ptr: Pointer, value0, value1, *, loc=None, ip=None) -> None:
    """Store 2 matrices (8x8 each) to shared memory"""
    _stmatrix(smem_ptr, [value0, value1], False, loc, ip)


@dsl_user_op
def stmatrix_x4(smem_ptr: Pointer, value0, value1, value2, value3, *, loc=None, ip=None) -> None:
    """Store 4 matrices (8x8 each) to shared memory"""
    _stmatrix(smem_ptr, [value0, value1, value2, value3], False, loc, ip)


@dsl_user_op
def stmatrix_x1_trans(smem_ptr: Pointer, value0, *, loc=None, ip=None) -> None:
    """Store 1 matrix (8x8) with transpose to shared memory"""
    _stmatrix(smem_ptr, [value0], True, loc, ip)


@dsl_user_op
def stmatrix_x2_trans(smem_ptr: Pointer, value0, value1, *, loc=None, ip=None) -> None:
    """Store 2 matrices (8x8 each) with transpose to shared memory"""
    _stmatrix(smem_ptr, [value0, value1], True, loc, ip)


@dsl_user_op
def stmatrix_x4_trans(smem_ptr: Pointer, value0, value1, value2, value3, *, loc=None, ip=None) -> None:
    """Store 4 matrices (8x8 each) with transpose to shared memory"""
    _stmatrix(smem_ptr, [value0, value1, value2, value3], True, loc, ip)


ptx_ldmatrix_x1 = ldmatrix_x1
ptx_ldmatrix_x2 = ldmatrix_x2
ptx_ldmatrix_x4 = ldmatrix_x4
ptx_ldmatrix_x1_trans = ldmatrix_x1_trans
ptx_ldmatrix_x2_trans = ldmatrix_x2_trans
ptx_ldmatrix_x4_trans = ldmatrix_x4_trans
ptx_stmatrix_x1 = stmatrix_x1
ptx_stmatrix_x2 = stmatrix_x2
ptx_stmatrix_x4 = stmatrix_x4
ptx_stmatrix_x1_trans = stmatrix_x1_trans
ptx_stmatrix_x2_trans = stmatrix_x2_trans
ptx_stmatrix_x4_trans = stmatrix_x4_trans

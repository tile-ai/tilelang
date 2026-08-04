"""ROCm-specific low-level TIR language operators."""

from __future__ import annotations

import tvm.tirx.op as _tvm_op

from tilelang.language.tir.exports import ROCM_ONLY_TIR_EXPORTS, SHARED_LEGACY_TIR_EXPORTS
from tilelang.language.tir.ir import _dtype_forward
from tilelang.language.tir.ir import (  # noqa: F401
    ptx_arrive_barrier,
    ptx_arrive_barrier_expect_tx,
    ptx_commit_group,
    ptx_cp_async,
    ptx_cp_async_barrier,
    ptx_init_barrier_thread_count,
    ptx_wait_group,
)
from tilelang.language.tir.op import call_intrin as _call_intrin


@_dtype_forward
def tvm_mfma(
    dtype,
    shape,
    A_layout,
    B_layout,
    A_dtype,
    B_dtype,
    C_dtype,
    multiplicand_a,
    a_index,
    multiplicand_b,
    b_index,
    accumulator,
    c_index,
):
    """TVM intrinsic for amd matrix core mfma instructions
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma

    Parameters
    ----------
    dtype : str
        The data type of the result.

    shape : str
        The shape of mma fragment.

    A_layout : Literal["row", "col"]
        The layout of multiplicand fragment A.

    B_layout : Literal["row", "col"]
        The layout of multiplicand fragment B.

    A_dtype : str
        The data type of multiplicand fragment A.

    B_dtype : str
        The data type of multiplicand fragment B.

    C_dtype : str
        The data type of accumulator fragment C.

    multiplicand_a : Var
        The multiplicand fragment A variable.

    a_index : Expr
        The index of multiplicand fragment A.

    multiplicand_b : Var
        The multiplicand fragment B variable.

    b_index : Expr
        The index of multiplicand fragment A.

    accumulator : Var
        The accumulator fragment C variable.

    c_index : Expr
        The index of accumulator fragment C.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _call_intrin(
        dtype,
        _tvm_op.Op.get("tl.tvm_mfma"),
        shape,
        A_layout,
        B_layout,
        A_dtype,
        B_dtype,
        C_dtype,
        multiplicand_a,
        a_index,
        multiplicand_b,
        b_index,
        accumulator,
        c_index,
    )


@_dtype_forward
def tvm_mfma_store(dtype, m, n, dst_ptr, src_ptr, src_offset, dst_stride):
    """TVM intrinsic for storing the result of PTX MMA into a destination pointer

    Parameters
    ----------
    dtype : str
        The data type of the result.

    m : IntImm
        The shape of mma fragment.

    n : IntImm
        The shape of mma fragment.

    dst_ptr : Var
        The destination pointer variable.

    src_ptr : Var
        The source pointer variable.

    src_offset : Expr
        The source offset.

    dst_stride : Var
        The destination stride.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _call_intrin(
        dtype,
        _tvm_op.Op.get("tl.tvm_mfma_store"),
        m,
        n,
        dst_ptr,
        src_ptr,
        src_offset,
        dst_stride,
    )


@_dtype_forward
def tvm_rdna_wmma(
    dtype,
    shape,
    A_layout,
    B_layout,
    A_dtype,
    B_dtype,
    C_dtype,
    multiplicand_a,
    a_index,
    multiplicand_b,
    b_index,
    accumulator,
    c_index,
):
    """TVM intrinsic for amd matrix core mfma instructions
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma

    Parameters
    ----------
    dtype : str
        The data type of the result.

    shape : str
        The shape of mma fragment.

    A_layout : Literal["row", "col"]
        The layout of multiplicand fragment A.

    B_layout : Literal["row", "col"]
        The layout of multiplicand fragment B.

    A_dtype : str
        The data type of multiplicand fragment A.

    B_dtype : str
        The data type of multiplicand fragment B.

    C_dtype : str
        The data type of accumulator fragment C.

    multiplicand_a : Var
        The multiplicand fragment A variable.

    a_index : Expr
        The index of multiplicand fragment A.

    multiplicand_b : Var
        The multiplicand fragment B variable.

    b_index : Expr
        The index of multiplicand fragment A.

    accumulator : Var
        The accumulator fragment C variable.

    c_index : Expr
        The index of accumulator fragment C.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _call_intrin(
        dtype,
        _tvm_op.Op.get("tl.tvm_rdna_wmma"),
        shape,
        A_layout,
        B_layout,
        A_dtype,
        B_dtype,
        C_dtype,
        multiplicand_a,
        a_index,
        multiplicand_b,
        b_index,
        accumulator,
        c_index,
    )


@_dtype_forward
def tvm_rdna_wmma_store(dtype, m, n, dst_ptr, src_ptr, src_offset, dst_stride):
    """TVM intrinsic for storing the result of PTX MMA into a destination pointer

    Parameters
    ----------
    dtype : str
        The data type of the result.

    m : IntImm
        The shape of mma fragment.

    n : IntImm
        The shape of mma fragment.

    dst_ptr : Var
        The destination pointer variable.

    src_ptr : Var
        The source pointer variable.

    src_offset : Expr
        The source offset.

    dst_stride : Var
        The destination stride.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _call_intrin(
        dtype,
        _tvm_op.Op.get("tl.tvm_rdna_wmma_store"),
        m,
        n,
        dst_ptr,
        src_ptr,
        src_offset,
        dst_stride,
    )


__all__ = tuple(sorted(ROCM_ONLY_TIR_EXPORTS | SHARED_LEGACY_TIR_EXPORTS))

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
    """Build an AMD MFMA matrix-core call."""
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
    """Build an AMD MFMA accumulator store call."""
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
    """Build an AMD RDNA WMMA matrix-core call."""
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
    """Build an AMD RDNA WMMA accumulator store call."""
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

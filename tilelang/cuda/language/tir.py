"""CUDA-specific low-level TIR language operators."""

import tilelang.language.tir.op as _tir_op
from tilelang.language.tir.exports import CUDA_ONLY_TIR_EXPORTS, SHARED_LEGACY_TIR_EXPORTS
from tilelang.language.tir.ir import (
    _dtype_forward,
    _op_wrapper,
    ptx_arrive_barrier as ptx_arrive_barrier,
    ptx_arrive_barrier_expect_tx as ptx_arrive_barrier_expect_tx,
    ptx_commit_group as ptx_commit_group,
    ptx_cp_async as ptx_cp_async,
    ptx_cp_async_barrier as ptx_cp_async_barrier,
    ptx_init_barrier_thread_count as ptx_init_barrier_thread_count,
    ptx_wait_group as ptx_wait_group,
)

ptx_cp_async_bulk = _dtype_forward(_tir_op.ptx_cp_async_bulk)
ptx_fence_barrier_init = _op_wrapper(_tir_op.ptx_fence_barrier_init)
ptx_ldmatrix = _tir_op.ptx_ldmatrix
ptx_mma = _dtype_forward(_tir_op.ptx_mma)
ptx_mma_sp = _dtype_forward(_tir_op.ptx_mma_sp)
ptx_tcgen05_mma_blockscaled_ss = _dtype_forward(_tir_op.ptx_tcgen05_mma_blockscaled_ss)
ptx_tcgen05_mma_ss = _dtype_forward(_tir_op.ptx_tcgen05_mma_ss)
ptx_tcgen05_mma_ts = _dtype_forward(_tir_op.ptx_tcgen05_mma_ts)
ptx_wait_barrier = _op_wrapper(_tir_op.ptx_wait_barrier)
ptx_wgmma_rs = _dtype_forward(_tir_op.ptx_wgmma_rs)
ptx_wgmma_sp_rs = _dtype_forward(_tir_op.ptx_wgmma_sp_rs)
ptx_wgmma_sp_ss = _dtype_forward(_tir_op.ptx_wgmma_sp_ss)
ptx_wgmma_ss = _dtype_forward(_tir_op.ptx_wgmma_ss)

__all__ = tuple(sorted(CUDA_ONLY_TIR_EXPORTS | SHARED_LEGACY_TIR_EXPORTS))

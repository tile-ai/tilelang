"""ROCm-specific low-level TIR language operators."""

import tilelang.language.tir.op as _tir_op
from tilelang.language.tir.exports import SHARED_LEGACY_TIR_EXPORTS
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

tvm_mfma = _dtype_forward(_tir_op.tvm_mfma)
tvm_mfma_store = _dtype_forward(_tir_op.tvm_mfma_store)
tvm_rdna_wmma = _dtype_forward(_tir_op.tvm_rdna_wmma)
tvm_rdna_wmma_store = _dtype_forward(_tir_op.tvm_rdna_wmma_store)

__all__ = tuple(
    sorted(
        (
            *SHARED_LEGACY_TIR_EXPORTS,
            "tvm_mfma",
            "tvm_mfma_store",
            "tvm_rdna_wmma",
            "tvm_rdna_wmma_store",
        )
    )
)

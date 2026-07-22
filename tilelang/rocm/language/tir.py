"""ROCm-specific low-level TIR language operators."""

import tilelang.language.tir.op as _tir_op
from tilelang.language.tir.ir import _dtype_forward

tvm_mfma = _dtype_forward(_tir_op.tvm_mfma)
tvm_mfma_store = _dtype_forward(_tir_op.tvm_mfma_store)
tvm_rdna_wmma = _dtype_forward(_tir_op.tvm_rdna_wmma)
tvm_rdna_wmma_store = _dtype_forward(_tir_op.tvm_rdna_wmma_store)

__all__ = [
    "tvm_mfma",
    "tvm_mfma_store",
    "tvm_rdna_wmma",
    "tvm_rdna_wmma_store",
]

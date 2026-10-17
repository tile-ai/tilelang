"""ROCm/HIP-specific language operators and intrinsic helpers."""

from __future__ import annotations

from tilelang.language.builtin import ds_read_tr16_b64, ds_read_tr8_b64
from tilelang.rocm.intrinsics import (
    MatrixCoreIntrinEmitter,
    MatrixCorePreshuffleIntrinEmitter,
    WMMAIntrinEmitter,
    get_mma_micro_size,
    make_mfma_swizzle_layout,
    mfma_store_index_map,
    mfma_store_index_map_32x32,
)

from .tir import tvm_mfma, tvm_mfma_store, tvm_rdna_wmma, tvm_rdna_wmma_store

mfma = tvm_mfma
mfma_store = tvm_mfma_store
rdna_wmma = tvm_rdna_wmma
rdna_wmma_store = tvm_rdna_wmma_store

__all__ = [
    "MatrixCoreIntrinEmitter",
    "MatrixCorePreshuffleIntrinEmitter",
    "WMMAIntrinEmitter",
    "ds_read_tr16_b64",
    "ds_read_tr8_b64",
    "get_mma_micro_size",
    "make_mfma_swizzle_layout",
    "mfma",
    "mfma_store",
    "mfma_store_index_map",
    "mfma_store_index_map_32x32",
    "rdna_wmma",
    "rdna_wmma_store",
    "tvm_mfma",
    "tvm_mfma_store",
    "tvm_rdna_wmma",
    "tvm_rdna_wmma_store",
]

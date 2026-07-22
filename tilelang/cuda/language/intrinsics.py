"""CUDA-specific language operators and intrinsic helpers."""

from __future__ import annotations

from tilelang.cuda.debug import device_assert
from tilelang.cuda.intrinsics import (
    TensorCoreIntrinEmitter,
    TensorCoreIntrinEmitterWithLadderTransform,
    get_ldmatrix_offset,
    get_mma_micro_size,
    get_swizzle_layout,
    make_mma_swizzle_layout,
    mma_store_index_map,
)
from tilelang.cuda.intrinsics.macro.mma_sp_macro_generator import SparseTensorCoreIntrinEmitter
from tilelang.cuda.intrinsics.macro.tcgen05_macro_generator import (
    TCGEN05DescriptorParams,
    TensorCoreIntrinEmitter as TCGEN05TensorCoreIntrinEmitter,
    compute_umma_descriptor,
)
from tilelang.cuda.intrinsics.macro.wgmma_macro_generator import (
    WGMMADescriptorParams,
    TensorCoreIntrinEmitter as WGMMATensorCoreIntrinEmitter,
    compute_gmma_descriptor,
)
from tilelang.cuda.intrinsics.macro.wgmma_sp_macro_generator import WGSparseTensorCoreIntrinEmitter
from tilelang.language.allocate import (
    alloc_tcgen05_instr_desc,
    alloc_tcgen05_instruction_desc,
    alloc_tcgen05_smem_desc,
    alloc_wgmma_desc,
)
from tilelang.language.builtin import (
    __ffs,
    __fns,
    __ldg,
    cp_async_barrier_noinc,
    initialize_tcgen05_descriptor,
    initialize_wgmma_descriptor,
    tcgen05_after_thread_sync,
    tcgen05_before_thread_sync,
    tcgen05_cp_warpx4,
    tcgen05_mma_arrive,
    tcgen05_sf_warp_transpose,
    wait_wgmma,
    warpgroup_arrive,
    warpgroup_commit_batch,
    warpgroup_fence_operand,
    warpgroup_wait,
)
from tilelang.language.experimental.gemm_sp_op import tcgen05_gemm_sp, wgmma_gemm_sp
from tilelang.language.gemm_op import (
    make_blockscaled_gemm_layout,
    tcgen05_gemm,
    tcgen05_gemm_blockscaled,
    wgmma_gemm,
)
from tilelang.language.tir.ir import (
    ptx_cp_async,
    ptx_cp_async_barrier,
    ptx_cp_async_bulk,
    ptx_ldmatrix,
    ptx_mma,
    ptx_mma_sp,
    ptx_tcgen05_mma_blockscaled_ss,
    ptx_tcgen05_mma_ss,
    ptx_tcgen05_mma_ts,
    ptx_wgmma_rs,
    ptx_wgmma_sp_rs,
    ptx_wgmma_sp_ss,
    ptx_wgmma_ss,
)

tcgen05_mma = tcgen05_gemm
tcgen05_mma_blockscaled = tcgen05_gemm_blockscaled
wgmma_mma = wgmma_gemm

__all__ = [
    "SparseTensorCoreIntrinEmitter",
    "TCGEN05DescriptorParams",
    "TCGEN05TensorCoreIntrinEmitter",
    "TensorCoreIntrinEmitter",
    "TensorCoreIntrinEmitterWithLadderTransform",
    "WGMMADescriptorParams",
    "WGMMATensorCoreIntrinEmitter",
    "WGSparseTensorCoreIntrinEmitter",
    "__ffs",
    "__fns",
    "__ldg",
    "alloc_tcgen05_instr_desc",
    "alloc_tcgen05_instruction_desc",
    "alloc_tcgen05_smem_desc",
    "alloc_wgmma_desc",
    "compute_gmma_descriptor",
    "compute_umma_descriptor",
    "cp_async_barrier_noinc",
    "device_assert",
    "get_ldmatrix_offset",
    "get_mma_micro_size",
    "get_swizzle_layout",
    "initialize_tcgen05_descriptor",
    "initialize_wgmma_descriptor",
    "make_blockscaled_gemm_layout",
    "make_mma_swizzle_layout",
    "mma_store_index_map",
    "ptx_cp_async",
    "ptx_cp_async_barrier",
    "ptx_cp_async_bulk",
    "ptx_ldmatrix",
    "ptx_mma",
    "ptx_mma_sp",
    "ptx_tcgen05_mma_blockscaled_ss",
    "ptx_tcgen05_mma_ss",
    "ptx_tcgen05_mma_ts",
    "ptx_wgmma_rs",
    "ptx_wgmma_sp_rs",
    "ptx_wgmma_sp_ss",
    "ptx_wgmma_ss",
    "tcgen05_after_thread_sync",
    "tcgen05_before_thread_sync",
    "tcgen05_cp_warpx4",
    "tcgen05_gemm",
    "tcgen05_gemm_blockscaled",
    "tcgen05_gemm_sp",
    "tcgen05_mma",
    "tcgen05_mma_arrive",
    "tcgen05_mma_blockscaled",
    "tcgen05_sf_warp_transpose",
    "wait_wgmma",
    "warpgroup_arrive",
    "warpgroup_commit_batch",
    "warpgroup_fence_operand",
    "warpgroup_wait",
    "wgmma_gemm",
    "wgmma_gemm_sp",
    "wgmma_mma",
]

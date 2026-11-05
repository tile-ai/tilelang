"""CUDA-specific language operators and intrinsic helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tilelang.cuda.debug import device_assert
    from tilelang.cuda.intrinsics import (
        TensorCoreIntrinEmitter,
        TensorCoreIntrinEmitterSM120,
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
from .gemm_op import (
    make_blockscaled_gemm_layout,
    mma_gemm_blockscaled,
    tcgen05_gemm,
    tcgen05_gemm_blockscaled,
    wgmma_gemm,
)
from .tir import (
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
    "TensorCoreIntrinEmitterSM120",
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
    "mma_gemm_blockscaled",
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


_HELPER_EXPORTS = {
    "device_assert": ("tilelang.cuda.debug", "device_assert"),
    "TensorCoreIntrinEmitter": ("tilelang.cuda.intrinsics", "TensorCoreIntrinEmitter"),
    "TensorCoreIntrinEmitterSM120": ("tilelang.cuda.intrinsics", "TensorCoreIntrinEmitterSM120"),
    "TensorCoreIntrinEmitterWithLadderTransform": ("tilelang.cuda.intrinsics", "TensorCoreIntrinEmitterWithLadderTransform"),
    "get_ldmatrix_offset": ("tilelang.cuda.intrinsics", "get_ldmatrix_offset"),
    "get_mma_micro_size": ("tilelang.cuda.intrinsics", "get_mma_micro_size"),
    "get_swizzle_layout": ("tilelang.cuda.intrinsics", "get_swizzle_layout"),
    "make_mma_swizzle_layout": ("tilelang.cuda.intrinsics", "make_mma_swizzle_layout"),
    "mma_store_index_map": ("tilelang.cuda.intrinsics", "mma_store_index_map"),
    "SparseTensorCoreIntrinEmitter": ("tilelang.cuda.intrinsics.macro.mma_sp_macro_generator", "SparseTensorCoreIntrinEmitter"),
    "TCGEN05DescriptorParams": ("tilelang.cuda.intrinsics.macro.tcgen05_macro_generator", "TCGEN05DescriptorParams"),
    "TCGEN05TensorCoreIntrinEmitter": ("tilelang.cuda.intrinsics.macro.tcgen05_macro_generator", "TensorCoreIntrinEmitter"),
    "compute_umma_descriptor": ("tilelang.cuda.intrinsics.macro.tcgen05_macro_generator", "compute_umma_descriptor"),
    "WGMMADescriptorParams": ("tilelang.cuda.intrinsics.macro.wgmma_macro_generator", "WGMMADescriptorParams"),
    "WGMMATensorCoreIntrinEmitter": ("tilelang.cuda.intrinsics.macro.wgmma_macro_generator", "TensorCoreIntrinEmitter"),
    "compute_gmma_descriptor": ("tilelang.cuda.intrinsics.macro.wgmma_macro_generator", "compute_gmma_descriptor"),
    "WGSparseTensorCoreIntrinEmitter": ("tilelang.cuda.intrinsics.macro.wgmma_sp_macro_generator", "WGSparseTensorCoreIntrinEmitter"),
}


def __getattr__(name):
    if name not in _HELPER_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module, attribute = _HELPER_EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value

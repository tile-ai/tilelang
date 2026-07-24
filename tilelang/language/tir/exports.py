"""Ownership classification for target-specific TIR script exports."""

CUDA_ONLY_TIR_EXPORTS = frozenset(
    {
        "ptx_cp_async_bulk",
        "ptx_fence_barrier_init",
        "ptx_ldmatrix",
        "ptx_mma",
        "ptx_mma_sp",
        "ptx_tcgen05_mma_blockscaled_ss",
        "ptx_tcgen05_mma_ss",
        "ptx_tcgen05_mma_ts",
        "ptx_wait_barrier",
        "ptx_wgmma_rs",
        "ptx_wgmma_sp_rs",
        "ptx_wgmma_sp_ss",
        "ptx_wgmma_ss",
    }
)

METAL_ONLY_TIR_EXPORTS = frozenset(
    {
        "make_filled_simdgroup_matrix",
        "simdgroup_load",
        "simdgroup_multiply_accumulate",
        "simdgroup_store",
    }
)

# These names are historical PTX spellings, but CUDA and HIP both lower them.
# Keep them common until backend-neutral spellings replace their public use.
SHARED_LEGACY_TIR_EXPORTS = frozenset(
    {
        "ptx_arrive_barrier",
        "ptx_arrive_barrier_expect_tx",
        "ptx_commit_group",
        "ptx_cp_async",
        "ptx_cp_async_barrier",
        "ptx_init_barrier_thread_count",
        "ptx_wait_group",
    }
)

BACKEND_ONLY_TIR_EXPORTS = CUDA_ONLY_TIR_EXPORTS | METAL_ONLY_TIR_EXPORTS
CLASSIFIED_VENDOR_TIR_EXPORTS = BACKEND_ONLY_TIR_EXPORTS | SHARED_LEGACY_TIR_EXPORTS

__all__ = [
    "BACKEND_ONLY_TIR_EXPORTS",
    "CLASSIFIED_VENDOR_TIR_EXPORTS",
    "CUDA_ONLY_TIR_EXPORTS",
    "METAL_ONLY_TIR_EXPORTS",
    "SHARED_LEGACY_TIR_EXPORTS",
]

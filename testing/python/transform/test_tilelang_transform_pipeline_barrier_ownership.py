"""Regression tests for pipeline barrier ownership.

Plain pipelined T.copy should stay on the synchronous path in non-WS kernels.
Explicit TMA-style producers such as im2col still own pipeline barriers when
their lowering requires them.
"""

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing


def _check_hopper():
    """Return True if running on Hopper (sm_90)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        props = torch.cuda.get_device_properties(0)
        return (props.major, props.minor) == (9, 0)
    except Exception:
        return False


@pytest.mark.skipif(not _check_hopper(), reason="Requires Hopper GPU (sm_90)")
def test_nonws_plain_copy_gemm_num_stages_3_stays_sync():
    """Non-WS pipelined GEMM should not auto-upgrade plain T.copy to TMA."""
    M, N, K = 512, 512, 512
    block_M, block_N, block_K = 128, 128, 32

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), T.float16)
            B_s = T.alloc_shared((block_K, block_N), T.float16)
            C_l = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_l)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_s)
                T.copy(B[ko * block_K, bx * block_N], B_s)
                T.gemm(A_s, B_s, C_l)
            T.copy(C_l, C[by * block_M, bx * block_N])

    kernel = tilelang.compile(
        gemm,
        out_idx=-1,
        execution_backend="tvm_ffi",
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    src = kernel.get_kernel_source()
    assert "tl::tma_load" not in src, "Non-WS plain T.copy should stay synchronous"
    assert "pipeline_mbar_mem" not in src, "Non-WS plain T.copy should not allocate pipeline TMA barriers"


@pytest.mark.skipif(not _check_hopper(), reason="Requires Hopper GPU (sm_90)")
def test_nonws_plain_copy_gemm_num_stages_1_stays_sync():
    """num_stages=1 should also keep non-WS plain T.copy on the sync path."""
    M, N, K = 512, 512, 512
    block_M, block_N, block_K = 128, 128, 32

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), T.float16)
            B_s = T.alloc_shared((block_K, block_N), T.float16)
            C_l = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_l)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.copy(A[by * block_M, ko * block_K], A_s)
                T.copy(B[ko * block_K, bx * block_N], B_s)
                T.gemm(A_s, B_s, C_l)
            T.copy(C_l, C[by * block_M, bx * block_N])

    kernel = tilelang.compile(
        gemm,
        out_idx=-1,
        execution_backend="tvm_ffi",
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    src = kernel.get_kernel_source()
    assert "tl::tma_load" not in src, "Non-WS plain T.copy should stay synchronous"
    assert "pipeline_mbar_mem" not in src, "Non-WS plain T.copy should not allocate pipeline TMA barriers"


@pytest.mark.skipif(not _check_hopper(), reason="Requires Hopper GPU (sm_90)")
def test_nonws_im2col_tma_num_stages_3_uses_pipeline_barrier():
    """Non-WS pipelined im2col TMA with num_stages=3 must use pipeline_mbar[3]."""
    N, C, H, W, F, K_size = 4, 64, 32, 32, 64, 3
    S, D, P = 1, 1, 1
    block_M, block_N, block_K = 64, 128, 32
    KH, KW = K_size, K_size
    OH = (H + 2 * P - D * (K_size - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K_size - 1) - 1) // S + 1
    num_stages = 3

    @T.prim_func
    def conv(
        data: T.Tensor((N, H, W, C), T.float16),
        weight: T.Tensor((KH, KW, C, F), T.float16),
        out: T.Tensor((N, OH, OW, F), T.float16),
    ):
        with T.Kernel(T.ceildiv(F, block_N), T.ceildiv(N * OH * OW, block_M), threads=256) as (bx, by):
            data_shared = T.alloc_shared((block_M, block_K), T.float16)
            weight_shared = T.alloc_shared((block_K, block_N), T.float16)
            out_local = T.alloc_fragment((block_M, block_N), T.float32)
            out_shared = T.alloc_shared((block_M, block_N), T.float16)
            kernel_flat = T.Tensor((KH * KW * C, F), T.float16, weight.data)
            out_flat = T.Tensor((N * OH * OW, F), T.float16, out.data)
            T.clear(out_local)
            for k_iter in T.Pipelined(T.ceildiv(KH * KW * C, block_K), num_stages=num_stages):
                T.c2d_im2col(data, data_shared, by, k_iter, KH, S, D, P)
                T.copy(kernel_flat[k_iter * block_K, bx * block_N], weight_shared)
                T.gemm(data_shared, weight_shared, out_local)
            T.copy(out_local, out_shared)
            T.copy(out_shared, out_flat[by * block_M, bx * block_N])

    kernel = tilelang.compile(
        conv,
        out_idx=-1,
        execution_backend="tvm_ffi",
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    src = kernel.get_kernel_source()
    assert f"pipeline_mbar_mem[{num_stages}]" in src, f"Expected pipeline_mbar_mem[{num_stages}] for non-WS im2col TMA pipeline"
    # tma_load_im2col must appear (im2col was lowered through TMA path)
    assert "tma_load_im2col" in src, "Expected tma_load_im2col in generated code"
    # No fallback internal barriers for im2col
    assert "mbarrier_1" not in src, "Should not have fallback mbarrier_1 when im2col uses pipeline barrier"


if __name__ == "__main__":
    test_nonws_plain_copy_gemm_num_stages_3_stays_sync()
    test_nonws_plain_copy_gemm_num_stages_1_stays_sync()
    test_nonws_im2col_tma_num_stages_3_uses_pipeline_barrier()
    print("All pipeline barrier ownership tests passed!")

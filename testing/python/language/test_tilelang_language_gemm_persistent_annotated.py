import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def _assert_persistent_gemm_close(kernel, M: int, N: int, K: int) -> None:
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = kernel(a, b)
    torch.cuda.synchronize()
    ref = (a.float() @ b.float()).half()
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)


def _compile_persistent_gemm(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    *,
    target: str,
    num_stages=3,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.PersistentKernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return tilelang.compile(gemm, out_idx=[-1], target=target)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("num_stages", [2, 3])
def test_persistent_annotated_gemm_compiles_for_supported_pipeline_stages(num_stages):
    M = N = K = 1024
    block_M = block_N = 128
    block_K = 32
    kernel = _compile_persistent_gemm(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        num_stages=num_stages,
        target="cuda",
    )

    source = kernel.get_kernel_source()
    assert "gemm_kernel" in source
    _assert_persistent_gemm_close(kernel, M, N, K)


@tilelang.testing.requires_cuda
def test_persistent_annotated_gemm_stage3_waits_keep_persistent_phase_progress():
    kernel_codegen = _compile_persistent_gemm(
        8192,
        8192,
        8192,
        128,
        128,
        32,
        num_stages=3,
        target="cuda",
    )

    source = kernel_codegen.get_kernel_source()
    assert "mbarrier[4].wait((w_tile_sched & 1));" in source
    assert "mbarrier[5].wait((w_tile_sched & 1));" in source
    assert "mbarrier[0].wait(1);" in source
    assert "mbarrier[1].wait(1);" in source

    M = N = K = 1024
    kernel_check = _compile_persistent_gemm(
        M,
        N,
        K,
        128,
        128,
        32,
        num_stages=3,
        target="cuda",
    )
    _assert_persistent_gemm_close(kernel_check, M, N, K)


if __name__ == "__main__":
    tilelang.testing.main()

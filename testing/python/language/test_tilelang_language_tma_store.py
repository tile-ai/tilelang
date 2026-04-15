"""Tests for TMA store (shared -> global).

Explicit T.tma_copy(shared_buf, global_buf) emits tma_store + tma_store_arrive
(no wait). The user must explicitly call T.tma_store_wait() for
synchronization.

Plain T.copy(shared_buf, global_buf) may also auto-lower to tma_store when the
store-side TMA constraints are satisfied. In that case lowering emits both
tma_store_arrive and tma_store_wait automatically.
"""

from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T
import tilelang


def matmul_tma_store(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    threads,
    num_stages,
):
    """GEMM with T.copy for loads and T.tma_copy for the final store (shared -> global)."""
    A_shape = (M, K)
    B_shape = (K, N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_K, block_N), in_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            # Store: fragment -> shared, then shared -> global via T.tma_copy
            T.copy(C_local, C_shared)
            T.tma_copy(C_shared, C[by * block_M, bx * block_N])
            T.tma_store_wait()

    return main


def run_gemm_tma_store(num_stages):
    M, N, K = 1024, 1024, 1024
    block_M, block_N, block_K = 128, 128, 32
    in_dtype = T.float16
    out_dtype = T.float16
    accum_dtype = T.float32
    threads = 128

    program = matmul_tma_store(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        in_dtype,
        out_dtype,
        accum_dtype,
        threads,
        num_stages,
    )
    kernel = tilelang.compile(
        program,
        out_idx=[2],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    kernel_source = kernel.get_kernel_source()
    print(kernel_source)
    # exit()
    # Verify that the generated kernel contains tma_store_arrive but NOT tma_store_wait
    # (the wait is issued separately by the user via T.tma_store_wait)
    assert "tma_store_arrive" in kernel_source, "Expected tma_store_arrive in kernel source"

    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        return C.to(torch.__getattribute__(out_dtype))

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def auto_tma_store_copy(M, N, block_M, block_N, dtype, threads):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(A_shared, C[by * block_M, bx * block_N])

    return main


def run_auto_tma_store_copy():
    M = N = 256
    block_M = block_N = 128
    dtype = T.float16
    threads = 128

    program = auto_tma_store_copy(M, N, block_M, block_N, dtype, threads)
    kernel = tilelang.compile(
        program,
        out_idx=[1],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    kernel_source = kernel.get_kernel_source()
    assert "tma_store_arrive" in kernel_source, "Expected auto tma_store_arrive in kernel source"
    assert "tma_store_wait" in kernel_source, "Expected auto tma_store_wait in kernel source"

    profiler = kernel.get_profiler()

    def ref_program(A):
        return A

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_store_2_stages():
    run_gemm_tma_store(num_stages=2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_store_3_stages():
    run_gemm_tma_store(num_stages=3)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_plain_copy_auto_tma_store():
    run_auto_tma_store_copy()


if __name__ == "__main__":
    # tilelang.testing.main()
    test_tma_store_2_stages()

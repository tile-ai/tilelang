"""Test T.tma_copy() with user-managed mbarrier synchronization.

T.tma_copy() emits only expect_tx + tma_load (no arrive, no wait).
The user must explicitly call T.barrier_arrive() and T.mbarrier_wait_parity().
This allows multiple tma_copy operations to share a single barrier arrive.
MultiVersionBuffer expands the barrier to num_stages versions automatically.
"""

from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T
import tilelang


def matmul_tma_copy(
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
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            mbar_A = T.alloc_barrier(128)
            mbar_B = T.alloc_barrier(128)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.tma_copy(A[by * block_M, k * block_K], A_shared, barrier=mbar_A)
                T.barrier_arrive(mbar_A)
                T.tma_copy(B[k * block_K, bx * block_N], B_shared, barrier=mbar_B)
                T.barrier_arrive(mbar_B)
                T.mbarrier_wait_parity(mbar_A, k % 2)
                T.mbarrier_wait_parity(mbar_B, k % 2)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_tma_copy(num_stages):
    M, N, K = 1024, 1024, 1024
    block_M, block_N, block_K = 128, 128, 32
    in_dtype = T.float16
    out_dtype = T.float16
    accum_dtype = T.float32
    threads = 128

    program = matmul_tma_copy(
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
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    print(kernel.get_kernel_source())
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        return C.to(torch.__getattribute__(out_dtype))

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_copy_pipeline_2_stages():
    run_gemm_tma_copy(num_stages=2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_copy_pipeline_3_stages():
    run_gemm_tma_copy(num_stages=3)


if __name__ == "__main__":
    tilelang.testing.main()

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.testing.requires_cuda_compute_version_eq(8, 0)
def test_copy_and_async_copy_gemm_codegen_equivalent_sm80():
    """For SM80, T.copy(global->shared) may lower to cp.async.

    This test checks that the explicit form:
      T.async_copy(...) + T.ptx_wait_group(0)
    produces identical CUDA source as:
      T.copy(...)

    This is intentionally a codegen equivalence test (not a perf test).
    """

    M = 256
    N = 256
    K = 128
    block_M = 128
    block_N = 128
    block_K = 32

    @T.prim_func
    def matmul_relu_kernel(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M)) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.async_copy(A[by * block_M, ko * block_K], A_shared)
                T.ptx_wait_group(0)

                T.async_copy(B[ko * block_K, bx * block_N], B_shared)
                T.ptx_wait_group(0)

                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)

            T.copy(C_local, C[by * block_M, bx * block_N])

    async_matmul_relu = matmul_relu_kernel

    @T.prim_func
    def matmul_relu_kernel(  # noqa: F811
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M)) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)

            T.copy(C_local, C[by * block_M, bx * block_N])

    sync_matmul_relu = matmul_relu_kernel

    # Compile both and compare the generated CUDA source.
    async_kernel = tilelang.compile(async_matmul_relu, target="cuda")
    sync_kernel = tilelang.compile(sync_matmul_relu, target="cuda")

    async_src = async_kernel.get_kernel_source()
    sync_src = sync_kernel.get_kernel_source()

    assert async_src == sync_src


if __name__ == "__main__":
    tilelang.testing.main()

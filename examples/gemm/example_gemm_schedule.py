import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm_schedule(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Enable rasterization for better L2 Cache Locality
            T.use_swizzle(panel_size=10)

            # Clear the local buffer
            T.clear(C_local)

            # Auto pipeline the computation
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)

                # Instead of using
                # T.copy(B[k * block_K, bx * block_N], B_shared)
                # we can also use Parallel to auto map the thread
                # bindings and vectorize the copy operation.
                for k, j in T.Parallel(block_K, block_N):
                    B_shared[k, j] = B[ko * block_K + k, bx * block_N + j]

                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_schedule


def main():
    kernel = matmul(1024, 1024, 1024, 128, 128, 32)

    import torch

    a = torch.randn(1024, 1024).cuda().half()
    b = torch.randn(1024, 1024).cuda().half()

    c = kernel(a, b)

    ref_c = a @ b

    print("c:")
    print(c)
    print("ref_c:")
    print(ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")

    # Get CUDA Source
    print("CUDA Source:")
    print(kernel.get_kernel_source())


def run_regression_perf():
    kernel = matmul(1024, 1024, 1024, 128, 128, 32)
    import torch

    a = torch.randn(1024, 1024).cuda().half()
    b = torch.randn(1024, 1024).cuda().half()
    from tilelang.profiler import do_bench

    def run_kernel_only():
        kernel(a, b)

    return do_bench(run_kernel_only, warmup=10, rep=100, backend="cupti")


if __name__ == "__main__":
    main()

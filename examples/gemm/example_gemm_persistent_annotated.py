import tilelang
import tilelang.language as T
tilelang.disable_cache()

@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
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
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def main():
    kernel = matmul(8192, 8192, 8192, 128, 128, 32)
    print(kernel.get_kernel_source())
    import torch

    print("alloc inputs")
    a = torch.randn(8192, 8192).cuda().half()
    b = torch.randn(8192, 8192).cuda().half()

    print("start torch ref")
    ref_c = (a @ b).half()
    torch.cuda.synchronize()
    print("finish torch ref")

    print("launch tilelang kernel")
    c = kernel(a, b)
    print("launched tilelang kernel")
    print("synchronize tilelang kernel")
    torch.cuda.synchronize()
    print("finish tilelang synchronize")

    print("c:")
    print(c)
    print("ref_c:")
    print(ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")

    # Get CUDA Source
    print("CUDA Source:")
    print(kernel.get_kernel_source())

    # benchmark
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(backend="cupti")
    # latency = profiler.do_bench()
    print(f"tilelang Latency: {latency}ms")


def run_regression_perf():
    kernel = matmul(8192, 8192, 8192, 128, 128, 32)
    profiler = kernel.get_profiler()
    return profiler.do_bench(backend="cupti")


if __name__ == "__main__":
    main()

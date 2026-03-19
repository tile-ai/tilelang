import tilelang
import tilelang.language as T


@tilelang.jit
def matmul(A, B, block_M: int = 128, block_N: int = 128, block_K: int = 32, dtype: T.dtype = T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]
    C = T.empty([M, N], dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


def main():
    import torch

    a = torch.randn(1024, 1024).cuda().half()
    b = torch.randn(1024, 1024).cuda().half()

    c = matmul(a, b)

    ref_c = a @ b

    print("c:")
    print(c)
    print("ref_c:")
    print(ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")

    # Get CUDA Source
    print("CUDA Source:")
    print(matmul.get_kernel_source(a, b))

    # benchmark
    kernel = matmul.compile(a, b)
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(backend="cupti")
    # latency = profiler.do_bench()
    print(f"tilelang Latency: {latency}ms")


def run_regression_perf():
    import torch

    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    kernel = matmul.compile(a, b)
    profiler = kernel.get_profiler()
    return profiler.do_bench(backend="cupti")


if __name__ == "__main__":
    main()

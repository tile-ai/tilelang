import tilelang
import tilelang.language as T


# add decorator @tilelang.jit if you want to return a torch function
# @tilelang.jit
@tilelang.jit
def matmul(A, B, block_M: int = 128, block_N: int = 128, block_K: int = 64, dtype: T.dtype = T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]
    C = T.empty([M, N], dtype)

    # Initialize Kernel Context
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        # create mbarrier for tma
        data_is_ready = T.alloc_barrier(arrive_count=128)
        compute_is_done = T.alloc_barrier(arrive_count=128)

        with T.ws(0):
            T.clear(C_local)

        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
            with T.ws(1):
                T.barrier_wait(compute_is_done, (ko + 1) % 2)
                T.tma_copy(A[by * block_M, ko * block_K], A_shared, barrier=data_is_ready)
                T.tma_copy(B[ko * block_K, bx * block_N], B_shared, barrier=data_is_ready)
                T.barrier_arrive(data_is_ready)
            with T.ws(0):
                T.barrier_wait(data_is_ready, ko % 2)
                T.gemm(A_shared, B_shared, C_local)
                T.barrier_arrive(compute_is_done)

        with T.ws(0):
            T.copy(C_local, C[by * block_M, bx * block_N])

    return C


def main(M=16384, N=16384, K=16384):
    import torch

    # Create random input tensors on the GPU
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    # Run the kernel through the Profiler
    c = matmul(a, b)
    # Reference multiplication using PyTorch
    ref_c = a @ b

    # Validate correctness
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Kernel output matches PyTorch reference.")

    # 4. Retrieve and inspect the generated CUDA source (optional)
    # cuda_source = matmul.get_kernel_source(a, b)
    # print("Generated CUDA kernel:\n", cuda_source)

    # 5.Profile latency with kernel
    kernel = matmul.compile(a, b)
    profiler = kernel.get_profiler()

    latency = profiler.do_bench()

    print(f"Latency: {latency} ms")


def run_regression_perf(M=16384, N=16384, K=16384):
    import torch

    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    c = matmul(a, b)

    ref_c = a @ b

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    kernel = matmul.compile(a, b)
    profiler = kernel.get_profiler()

    return profiler.do_bench(backend="cupti")


if __name__ == "__main__":
    main()

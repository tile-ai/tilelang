import tilelang
import tilelang.language as T


@tilelang.jit
def matmul_warp_specialize_copy_0_gemm_1(A, B, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    # Initialize Kernel Context
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        data_is_ready = T.alloc_barrier(arrive_count=128)
        compute_is_done = T.alloc_barrier(arrive_count=128)

        with T.ws(1):
            T.clear(C_local)

        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
            with T.ws(0):
                T.barrier_wait(compute_is_done, (ko + 1) % 2)
                T.tma_copy(A[by * block_M, ko * block_K], A_shared, barrier=data_is_ready)
                T.tma_copy(B[ko * block_K, bx * block_N], B_shared, barrier=data_is_ready)
                T.barrier_arrive(data_is_ready)
            with T.ws(1):
                T.barrier_wait(data_is_ready, ko % 2)
                T.gemm(A_shared, B_shared, C_local)
                T.barrier_arrive(compute_is_done)

        with T.ws(1):
            T.copy(C_local, C[by * block_M, bx * block_N])

    return C


def main(M=1024, N=1024, K=1024):
    block_M = 128
    block_N = 128
    block_K = 64

    import torch

    # Create random input tensors on the GPU
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    # Run the kernel
    c = matmul_warp_specialize_copy_0_gemm_1(a, b, block_M, block_N, block_K)
    # Reference multiplication using PyTorch
    ref_c = a @ b

    # Validate correctness
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Kernel output matches PyTorch reference.")

    # Retrieve and inspect the generated CUDA source (optional)
    # cuda_source = matmul_warp_specialize_copy_0_gemm_1.get_kernel_source(a, b, block_M, block_N, block_K)
    # print("Generated CUDA kernel:\n", cuda_source)

    # Profile latency with kernel
    from tilelang.profiler import do_bench

    latency = do_bench(lambda: matmul_warp_specialize_copy_0_gemm_1(a, b, block_M, block_N, block_K))

    print(f"Latency: {latency} ms")


def run_regression_perf(M=4096, N=4096, K=4096):
    block_M = 128
    block_N = 128
    block_K = 64

    import torch

    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    c = matmul_warp_specialize_copy_0_gemm_1(a, b, block_M, block_N, block_K)
    ref_c = a @ b

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    from tilelang.profiler import do_bench

    return do_bench(lambda: matmul_warp_specialize_copy_0_gemm_1(a, b, block_M, block_N, block_K), backend="cupti")


if __name__ == "__main__":
    main()

import tilelang
import tilelang.language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def matmul_warp_specialize_copy_gemm_0_1(A, B, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    warp_group_num = 2
    threads = 128 * warp_group_num

    # Initialize Kernel Context
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype, "shared")
        B_shared_g0 = T.alloc_shared((block_K, block_N // warp_group_num), dtype, "shared")
        B_shared_g1 = T.alloc_shared((block_K, block_N // warp_group_num), dtype, "shared")

        C_local_g0 = T.alloc_fragment((block_M, block_N // warp_group_num), accum_dtype)
        C_local_g1 = T.alloc_fragment((block_M, block_N // warp_group_num), accum_dtype)

        with T.ws(1):
            T.clear(C_local_g1)
        with T.ws(0):
            T.clear(C_local_g0)

        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            with T.ws(1):
                T.copy(B[ko * block_K, bx * block_N], B_shared_g1)
                T.gemm(A_shared, B_shared_g1, C_local_g1)
            with T.ws(0):
                T.copy(B[ko * block_K, bx * block_N + block_N // warp_group_num], B_shared_g0)
                T.gemm(A_shared, B_shared_g0, C_local_g0)

        with T.ws(1):
            T.copy(C_local_g1, C[by * block_M, bx * block_N])
        with T.ws(0):
            T.copy(C_local_g0, C[by * block_M, bx * block_N + block_N // warp_group_num])

    return C


def main():
    M = 128
    N = 128
    K = 64
    block_M = 128
    block_N = 128
    block_K = 64

    import torch

    # Create random input tensors on the GPU
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    # Run the kernel
    c = matmul_warp_specialize_copy_gemm_0_1(a, b, block_M, block_N, block_K)
    print(c)

    # Reference multiplication using PyTorch
    ref_c = a @ b
    print(ref_c)

    # Validate correctness
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Kernel output matches PyTorch reference.")

    # Retrieve and inspect the generated CUDA source (optional)
    # cuda_source = matmul_warp_specialize_copy_gemm_0_1.get_kernel_source(a, b, block_M, block_N, block_K)
    # print("Generated CUDA kernel:\n", cuda_source)

    # Profile latency with kernel
    from tilelang.profiler import do_bench

    latency = do_bench(lambda: matmul_warp_specialize_copy_gemm_0_1(a, b, block_M, block_N, block_K))

    print(f"Latency: {latency} ms")


if __name__ == "__main__":
    main()

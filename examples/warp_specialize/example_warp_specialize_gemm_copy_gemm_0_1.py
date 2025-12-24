import tilelang
import tilelang.language as T


# add decorator @tilelang.jit if you want to return a torch function
# @tilelang.jit
@tilelang.jit(
    out_idx=[2],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def matmul_warp_specialize_copy_1_gemm_0(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    warp_group_num = 2
    threads = 128 * warp_group_num

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
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

    return main


def main():
    M = 128
    N = 128
    K = 64
    block_M = 128
    block_N = 128
    block_K = 64

    jit_kernel = matmul_warp_specialize_copy_1_gemm_0(M, N, K, block_M, block_N, block_K)
    print(jit_kernel.get_kernel_source())
    # 3. Test the kernel in Python with PyTorch data
    import torch

    # Create random input tensors on the GPU
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    # Run the kernel through the Profiler
    c = jit_kernel(a, b)
    print(c)

    # Reference multiplication using PyTorch
    ref_c = a @ b
    print(ref_c)

    # Validate correctness
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Kernel output matches PyTorch reference.")

    # 4. Retrieve and inspect the generated CUDA source (optional)
    # cuda_source = jit_kernel.get_kernel_source()
    # print("Generated CUDA kernel:\n", cuda_source)

    # 5.Profile latency with kernel
    profiler = jit_kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

    latency = profiler.do_bench()

    print(f"Latency: {latency} ms")


if __name__ == "__main__":
    main()

# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import tilelang
import tilelang.language as T


# add decorator @tilelang.jit if you want to return a torch function
# @tilelang.jit
def matmul_warp_specialize_copy_0_gemm_1(M,
                                         N,
                                         K,
                                         block_M,
                                         block_N,
                                         block_K,
                                         dtype="float16",
                                         accum_dtype="float"):

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # create mbarrier for tma
            T.create_list_of_mbarrier(128, 128)

            with T.ws(1):
                T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                with T.ws(0):
                    T.mbarrier_wait_parity(1, (ko & 1) ^ 1)
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                    T.copy(B[ko * block_K, bx * block_N], B_shared)
                    T.mbarrier_arrive(0)
                with T.ws(1):
                    T.mbarrier_wait_parity(0, ko & 1)
                    T.gemm(A_shared, B_shared, C_local)
                    T.mbarrier_arrive(1)

            with T.ws(1):
                T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def main():
    M = 16384
    N = 16384
    K = 16384
    block_M = 128
    block_N = 128
    block_K = 64

    # 1. Define the kernel (matmul) and compile/lower it into an executable module
    func = matmul_warp_specialize_copy_0_gemm_1(M, N, K, block_M, block_N, block_K)

    # 2. Compile the kernel into a torch function
    # out_idx specifies the index of the output buffer in the argument list
    # if out_idx is specified, the tensor will be created during runtime
    # target currently can be "cuda" or "hip" or "cpu".
    jit_kernel = tilelang.compile(
        func,
        out_idx=[2],
    )

    # 3. Test the kernel in Python with PyTorch data
    import torch

    # Create random input tensors on the GPU
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    # Run the kernel through the Profiler
    c = jit_kernel(a, b)

    # Reference multiplication using PyTorch
    ref_c = a @ b

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

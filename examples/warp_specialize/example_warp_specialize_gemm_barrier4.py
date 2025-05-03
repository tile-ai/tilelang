# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import tilelang
import tilelang.language as T

tilelang.disable_cache()


def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):

    num_stages = 2
    mbarrier_list = [128, 128] * num_stages
    # add decorator @tilelang.jit if you want to return a torch function
    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
            A_shared = T.alloc_shared((num_stages, block_M, block_K), dtype, "shared")
            B_shared = T.alloc_shared((num_stages, block_K, block_N), dtype, "shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # create mbarrier for tma
            T.create_list_of_mbarrier(mbarrier_list)

            with T.ws(1):
                for ko in range(T.ceildiv(K, block_K)):
                    T.mbarrier_wait_parity(ko % num_stages + num_stages,
                                           ((ko // num_stages) % num_stages) ^ 1)
                    T.copy(A[by * block_M:(by + 1) * block_M, ko * block_K:(ko + 1) * block_K],
                           A_shared[ko % num_stages, :, :])
                    T.copy(B[ko * block_K:(ko + 1) * block_K, bx * block_N:(bx + 1) * block_N],
                           B_shared[ko % num_stages, :, :])
                    T.mbarrier_arrive(ko % num_stages)
            with T.ws(0):
                T.clear(C_local)
                for ko in range(T.ceildiv(K, block_K)):
                    T.mbarrier_wait_parity(ko % num_stages, (ko // num_stages) % num_stages)
                    T.gemm(A_shared[ko % num_stages, :, :], B_shared[ko % num_stages, :, :],
                           C_local)
                    T.mbarrier_arrive(ko % num_stages + num_stages)
                T.copy(C_local, C[by * block_M, bx * block_N])

    return main


K = 64
# 1. Define the kernel (matmul) and compile/lower it into an executable module
func = matmul(128, 128, K, 128, 128, 32)
print(func.script())
# 2. Compile the kernel into a torch function
# out_idx specifies the index of the output buffer in the argument list
# if out_idx is specified, the tensor will be created during runtime
# target currently can be "cuda" or "hip" or "cpu".
tilelang.disable_cache()
jit_kernel = tilelang.compile(
    func,
    out_idx=[2],
    target="cuda",
    execution_backend="cython",
    pass_configs={
        "tl.disable_warp_specialized": True,
        # "tl.disable_tma_lower": True,
    })
tilelang.enable_cache()
print(jit_kernel.get_kernel_source())
# 3. Test the kernel in Python with PyTorch data
import torch

# Create random input tensors on the GPU
a = torch.randn(128, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, 128, device="cuda", dtype=torch.float16)

# Run the kernel through the Profiler
c = jit_kernel(a, b)

print(c)
# Reference multiplication using PyTorch
ref_c = a @ b

# Validate correctness
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel output matches PyTorch reference.")

# # 4. Retrieve and inspect the generated CUDA source (optional)
# # cuda_source = jit_kernel.get_kernel_source()
# # print("Generated CUDA kernel:\n", cuda_source)

# # 5.Profile latency with kernel
# profiler = jit_kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

# latency = profiler.do_bench()

# print(f"Latency: {latency} ms")

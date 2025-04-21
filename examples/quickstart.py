# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import tilelang
import tilelang.language as T
# `make_mma_swizzle_layout` is a python defined layout function
# specifically designed for MMA operations
# which ensures the consistency with the nvidia CUTLASS Library.
# to avoid bank conflicts and maximize the performance.
from tilelang.intrinsics import (
    make_mma_swizzle_layout as make_swizzle_layout,)  # noqa: F401


def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    # add decorator @tilelang.jit if you want to return a torch function
    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Apply layout optimizations or define your own layout (Optional)
            # If not specified, we will deduce the layout automatically
            # T.annotate_layout({
            #     A_shared: make_swizzle_layout(A_shared),
            #     B_shared: make_swizzle_layout(B_shared),
            # })

            # Enable rasterization for better L2 cache locality (Optional)
            # T.use_swizzle(panel_size=10, enable=True)

            # Clear local accumulation
            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # Copy tile of A
                # This is a sugar syntax for parallelized copy
                # for i, k in T.Parallel(M, block_K):
                #     A_shared[i, k] = A[by * block_M + i, ko * block_K + k]
                T.copy(A[by * block_M, ko * block_K], A_shared)

                # Copy tile of B
                T.copy(B[ko * block_K, bx * block_N], B_shared)

                # Perform a tile-level GEMM on the shared buffers
                # Currently we dispatch to the cute/hip on Nvidia/AMD GPUs
                T.gemm(A_shared, B_shared, C_local)

            # Copy result back to global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


M = 1024  # M = T.symbolic("m") if you want to use dynamic shape
N = 1024
K = 1024
block_M = 128
block_N = 128
block_K = 32

# 1. Define the kernel (matmul) and compile/lower it into an executable module
func = matmul(M, N, K, block_M, block_N, block_K)

# 2. Compile the kernel into a torch function
# out_idx specifies the index of the output buffer in the argument list
# if out_idx is specified, the tensor will be created during runtime
# target currently can be "cuda" or "hip" or "cpu".
jit_kernel = tilelang.compile(func, out_idx=[2], target="cuda", execution_backend="cython")
# jit_kernel = tilelang.compile(func, out_idx=[2], target="cuda", execution_backend="dlpack")

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

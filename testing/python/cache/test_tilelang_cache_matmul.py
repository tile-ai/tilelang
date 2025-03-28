# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from tilelang import tvm as tvm
import tilelang.testing
from tilelang import cached
import tilelang.language as T


def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    """
    Defines a matrix multiplication primitive function using tilelang.

    This function constructs a tilelang primitive function for matrix multiplication,
    optimized for execution on hardware accelerators. It utilizes shared memory and
    fragment memory for performance.

    Args:
        M (int): Number of rows in matrix A and C.
        N (int): Number of columns in matrix B and C.
        K (int): Number of columns in matrix A and rows in matrix B.
        block_M (int): Block size for M dimension in shared memory and fragment.
        block_N (int): Block size for N dimension in shared memory and fragment.
        block_K (int): Block size for K dimension in shared memory.
        dtype (str, optional): Data type for input matrices A and B, and output C. Defaults to "float16".
        accum_dtype (str, optional): Accumulation data type for internal computations. Defaults to "float".

    Returns:
        T.PrimFunc: A tilelang primitive function representing the matrix multiplication.
    """

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
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

    return main


def run_cache_matmul():
    """
    Demonstrates the usage of the cached matrix multiplication kernel.

    This function defines a reference PyTorch matrix multiplication,
    creates a cached kernel from the tilelang matmul function,
    runs the kernel with random input tensors, compares the output with the reference,
    and prints the CUDA kernel source code.
    """

    def ref_program(A, B):
        """
        Reference PyTorch matrix multiplication for comparison.
        """
        import torch
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.half)  # Assuming dtype="float16" in matmul
        return C

    func = matmul(1024, 1024, 1024, 128, 128, 32)
    kernel = cached(func, [2], execution_backend="cython")
    import torch

    a = torch.randn(1024, 1024).cuda().half()
    b = torch.randn(1024, 1024).cuda().half()

    c = kernel(a, b)
    print("\nOutput from Cached Kernel:")
    print(c)

    ref_c = ref_program(a, b)
    print("\nReference PyTorch Output:")
    print(ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("\nOutputs are close (within tolerance).")

    # Get CUDA Source
    print("\nCUDA Kernel Source:")
    print(kernel.get_kernel_source())


def test_cache_matmul_f16f16f16_nn():
    """
    Test function for cached matrix multiplication (float16 inputs, float16 output, no transpose).
    """
    run_cache_matmul()


if __name__ == "__main__":
    tilelang.testing.main()

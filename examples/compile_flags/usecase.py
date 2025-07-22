import tilelang
import tilelang.language as T


# @tilelang.jit(compile_flags=["-O3", "--use_fast_math", "--expt-relaxed-constexpr"])
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):

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
            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


M = 1024
N = 1024
K = 1024
block_M = 128
block_N = 128
block_K = 32

func = matmul(M, N, K, block_M, block_N, block_K)

jit_kernel = tilelang.compile(
    func, out_idx=[2], target="cuda", compile_flags="-O3 --use_fast_math --expt-relaxed-constexpr")
# or jit_kernel = tilelang.compile(func, out_idx=[2], target="cuda", compile_flags=["-O3", "--use_fast_math", "--expt-relaxed-constexpr"])
# or jit_kernel = tilelang.compile(func, out_idx=[2], target="cuda", compile_flags=["-O3 --use_fast_math --expt-relaxed-constexpr"])

import torch

a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)

c = jit_kernel(a, b)

print(c)

ref_c = a @ b

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel output matches PyTorch reference.")

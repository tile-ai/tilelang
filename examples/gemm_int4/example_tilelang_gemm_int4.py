"""Frontend int4 GEMM example for the T.gemm int4 path.

This file intentionally models the desired TileLang frontend API:
- A/B are declared as T.int4 tensors
- the matmul is expressed with T.gemm(...)

The example compiles the kernel and prints the generated CUDA source.
"""

import tilelang
import tilelang.language as T

tilelang.disable_cache()


def matmul_nt_int4(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.int4),
        B: T.Tensor((N, K), T.int4),
        C: T.Tensor((M, N), T.int32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.int4)
            B_shared = T.alloc_shared((block_N, block_K), T.int4)
            C_local = T.alloc_fragment((block_M, block_N), T.int32)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                # Frontend expectation: T.gemm should accept int4 operands directly.
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def compile_int4_gemm(
    M=1024,
    N=1024,
    K=1024,
    block_M=128,
    block_N=128,
    block_K=64,
):
    func = matmul_nt_int4(M, N, K, block_M, block_N, block_K)
    kernel = tilelang.compile(func, out_idx=-1)
    print("Compilation succeeded.")
    print(kernel.get_kernel_source())
    return func, kernel


def main():
    compile_int4_gemm()


if __name__ == "__main__":
    main()

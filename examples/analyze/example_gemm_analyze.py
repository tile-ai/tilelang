import tilelang.language as T
from tilelang.tools import Analyzer
from tilelang.carver.arch import CUDA
from tilelang.carver.arch import CDNA
import torch

M = N = K = 1024


def kernel(
    block_M=None,
    block_N=None,
    block_K=None,
    num_stages=None,
    thread_num=None,
    enable_rasteration=None,
):
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def matmul(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)
            T.use_swizzle(panel_size=10, enable=enable_rasteration)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(
                    A_shared,
                    B_shared,
                    C_local,
                    transpose_B=True,
                )
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return matmul


def main():
    my_func = kernel(128, 128, 32, 3, 128, True)

    cuda_device = CUDA("cuda") if torch.version.hip is None else CDNA("hip")
    result = Analyzer.analysis(my_func, cuda_device)

    print(f"Analyzed FLOPs: {result.total_flops}")
    print(f"Expected FLOPs: {2 * M * N * K}")


if __name__ == "__main__":
    main()

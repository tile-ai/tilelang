import tilelang
import tilelang.language as T
from tilelang.tools import Analyzer
from tilelang.carver.arch import CUDA
from tilelang.carver.arch import CDNA
import torch

N = 64
C = 256
H = 512
W = 512
F = 512
K = 3
S = 1
D = 1
P = 1


@tilelang.jit
def kernel(data, kernel, S, D, P, block_M, block_N, block_K, num_stages, threads, dtype=T.float16, accum_dtype=T.float32):
    N, C, H, W, F, K = T.const("N, C, H, W, F, K")
    KH, KW = K, K
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
    dtype = T.float16
    accum_dtype = T.float32

    data: T.Tensor((N, H, W, C), dtype)
    kernel: T.Tensor((KH, KW, C, F), dtype)
    out = T.empty((N, OH, OW, F), dtype)

    with T.Kernel(T.ceildiv(F, block_N), T.ceildiv(N * OH * OW, block_M), threads=threads) as (bx, by):
        data_shared = T.alloc_shared((block_M, block_K), dtype)
        kernel_shared = T.alloc_shared((block_K, block_N), dtype)
        out_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        out_shared = T.alloc_shared((block_M, block_N), dtype)

        kernel_flat = T.Tensor((KH * KW * C, F), dtype, kernel.data)
        out_flat = T.Tensor((N * OH * OW, F), dtype, out.data)

        T.clear(out_local)
        for k_iter in T.Pipelined(T.ceildiv(KH * KW * C, block_K), num_stages=num_stages):
            T.im2col(data, data_shared, by, k_iter, KH, S, D, P)
            T.copy(kernel_flat[k_iter * block_K, bx * block_N], kernel_shared)
            T.gemm(data_shared, kernel_shared, out_local)

        T.copy(out_local, out_shared)
        T.copy(out_shared, out_flat[by * block_M, bx * block_N])

    return out


def main():
    my_func = kernel.get_tir(
        N=N,
        C=C,
        H=H,
        W=W,
        F=F,
        K=K,
        S=S,
        D=D,
        P=P,
        block_M=64,
        block_N=128,
        block_K=32,
        num_stages=3,
        threads=256,
    )
    cuda_device = CUDA("cuda") if torch.version.hip is None else CDNA("hip")
    result = Analyzer.analysis(my_func, cuda_device)
    print(result)
    print(f"Analyzed FLOPs: {result.total_flops}")


if __name__ == "__main__":
    main()

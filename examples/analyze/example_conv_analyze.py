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


def check_hopper():
    # if not torch.cuda.is_available():
    #     return None
    # props = torch.cuda.get_device_properties(0)
    # compute_capability = props.major, props.minor
    # return compute_capability == (9, 0)
    return False


def kernel(N, C, H, W, F, K, S, D, P, block_M, block_N, block_K, num_stages, threads, dtype=T.float16, accum_dtype=T.float32):
    KH, KW = K, K
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
    dtype = T.float16
    accum_dtype = T.float32
    is_hopper = check_hopper()

    @T.prim_func
    def conv(
        data: T.Tensor((N, H, W, C), dtype),
        kernel: T.Tensor((KH, KW, C, F), dtype),
        out: T.Tensor((N, OH, OW, F), dtype),
    ):
        with T.Kernel(T.ceildiv(F, block_N), T.ceildiv(N * OH * OW, block_M), threads=threads) as (bx, by):
            data_shared = T.alloc_shared((block_M, block_K), dtype)
            kernel_shared = T.alloc_shared((block_K, block_N), dtype)
            out_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            out_shared = T.alloc_shared((block_M, block_N), dtype)

            kernel_flat = T.Tensor((KH * KW * C, F), dtype, kernel.data)
            out_flat = T.Tensor((N * OH * OW, F), dtype, out.data)

            T.clear(out_local)
            for k_iter in T.Pipelined(T.ceildiv(KH * KW * C, block_K), num_stages=num_stages):
                if is_hopper:
                    T.c2d_im2col(data, data_shared, by, k_iter, KH, S, D, P)
                else:
                    for i, j in T.Parallel(block_M, block_K):
                        k = k_iter * block_K + j
                        m = by * block_M + i
                        access_h = m % (OH * OW) // OW * S + k // (KW * C) * D - P
                        access_w = m % OW * S + k // C % KW * D - P
                        in_bound = (access_h >= 0) and (access_w >= 0) and (access_h < H) and (access_w < W)
                        data_shared[i, j] = T.if_then_else(in_bound, data[m // (OH * OW), access_h, access_w, k % C], 0)
                T.copy(kernel_flat[k_iter * block_K, bx * block_N], kernel_shared)
                T.gemm(data_shared, kernel_shared, out_local)

            T.copy(out_local, out_shared)
            T.copy(out_shared, out_flat[by * block_M, bx * block_N])

    return conv


def main():
    my_func = kernel(N, C, H, W, F, K, S, D, P, 64, 128, 32, 3, 256)
    cuda_device = CUDA("cuda") if torch.version.hip is None else CDNA("hip")
    result = Analyzer.analysis(my_func, cuda_device)
    print(result)
    print(f"Analyzed FLOPs: {result.total_flops}")


if __name__ == "__main__":
    main()

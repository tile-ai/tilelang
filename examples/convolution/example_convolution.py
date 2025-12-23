import torch
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import argparse


def check_hopper():
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    compute_capability = props.major, props.minor
    return compute_capability == (9, 0)


def ref_program(stride, padding, dilation):
    def main(A, B):
        A = A.permute(0, 3, 1, 2)  # N, H, W, C -> N, C, H, W
        B = B.permute(3, 2, 0, 1)  # H, W, C, F -> F, C, H, W
        C = torch.conv2d(A, B, stride=stride, padding=padding, dilation=dilation)
        C = C.permute(0, 2, 3, 1)  # N, C, H, W -> N, H, W, C
        return C

    return main


@tilelang.jit(out_idx=[2])
def convolution(N, C, H, W, F, K, S, D, P, block_M, block_N, block_K, num_stages, threads, dtype=T.float16, accum_dtype=T.float32):
    KH, KW = K, K
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
    dtype = T.float16
    accum_dtype = T.float32
    is_hopper = check_hopper()

    @T.prim_func
    def main(
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

    return main


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128, help="n")
    parser.add_argument("--c", type=int, default=128, help="c")
    parser.add_argument("--h", type=int, default=64, help="h")
    parser.add_argument("--w", type=int, default=64, help="w")
    parser.add_argument("--f", type=int, default=128, help="f")
    parser.add_argument("--k", type=int, default=3, help="k")
    parser.add_argument("--s", type=int, default=1, help="s")
    parser.add_argument("--d", type=int, default=1, help="d")
    parser.add_argument("--p", type=int, default=1, help="p")

    args = parser.parse_args(argv)
    N, C, H, W, F, K, S, D, P = args.n, args.c, args.h, args.w, args.f, args.k, args.s, args.d, args.p
    a = torch.randn(N, H, W, C).cuda().half()
    b = torch.randn(K, K, C, F).cuda().half()

    block_m = 64
    block_n = 128
    block_k = 32
    num_stages = 3
    threads = 256
    kernel = convolution(N, C, H, W, F, K, S, D, P, block_m, block_n, block_k, num_stages, threads)

    out_c = kernel(a, b)
    ref_c = ref_program(S, P, D)(a, b)
    torch.testing.assert_close(out_c, ref_c, rtol=1e-2, atol=1e-2)
    print("All checks passed.✅")


def run_regression_perf(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128, help="n")
    parser.add_argument("--c", type=int, default=128, help="c")
    parser.add_argument("--h", type=int, default=64, help="h")
    parser.add_argument("--w", type=int, default=64, help="w")
    parser.add_argument("--f", type=int, default=128, help="f")
    parser.add_argument("--k", type=int, default=3, help="k")
    parser.add_argument("--s", type=int, default=1, help="s")
    parser.add_argument("--d", type=int, default=1, help="d")
    parser.add_argument("--p", type=int, default=1, help="p")

    args = parser.parse_args(argv)
    N, C, H, W, F, K, S, D, P = args.n, args.c, args.h, args.w, args.f, args.k, args.s, args.d, args.p

    block_m = 64
    block_n = 128
    block_k = 32
    num_stages = 3
    threads = 256
    kernel = convolution(N, C, H, W, F, K, S, D, P, block_m, block_n, block_k, num_stages, threads)
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)
    return profiler.do_bench(backend="cupti")


if __name__ == "__main__":
    main()

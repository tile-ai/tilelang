# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import torch
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import itertools
import argparse
from tilelang.carver.template import ConvTemplate
from tilelang.carver.arch import CUDA
from tilelang.carver.roller.rasterization import NoRasterization


def check_hopper():
    # if not torch.cuda.is_available():
    #     return None
    # props = torch.cuda.get_device_properties(0)
    # compute_capability = props.major, props.minor
    # return compute_capability == (9, 0)
    return False


def get_configs(N, C, H, W, F, K, S, D, P, with_roller=False):
    if with_roller:
        arch = CUDA("cuda")
        topk = 10
        carve_template = ConvTemplate(
            N=N,
            C=C,
            H=H,
            W=W,
            F=F,
            K=K,
            S=S,
            D=D,
            P=P,
            in_dtype="float16",
            out_dtype="float16",
            accum_dtype="float",
        ).with_arch(arch)

        func = carve_template.equivalent_function()
        assert func is not None, "Function is None"
        roller_hints = carve_template.recommend_hints(topk=topk)
        if roller_hints is None:
            raise ValueError("No Roller Hints Found for TensorCore Scheduling")
        configs = []
        for hint in roller_hints:
            config = {}
            block_m, block_n = hint.block
            warp_m, warp_n = hint.warp
            block_rows, block_cols = block_m // warp_m, block_n // warp_n
            config["block_M"] = block_m
            config["block_N"] = block_n
            config["block_K"] = hint.rstep[0]
            config["num_stages"] = hint.pipeline_stage
            config["thread_num"] = block_rows * block_cols * 32
            config["enable_rasteration"] = hint.rasterization_plan is not NoRasterization
            configs.append(config)
        for config in configs:
            print(config)
    else:
        block_M = [64, 128, 256]
        block_N = [64, 128, 256]
        block_K = [32, 64]
        num_stages = [0, 1, 2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(block_M, block_N, block_K, num_stages, threads))

        configs = [{
            'block_M': c[0],
            'block_N': c[1],
            'block_K': c[2],
            'num_stages': c[3],
            'thread_num': c[4]
        } for c in _configs]
    return configs


def ref_program(stride, padding, dilation):

    def main(A, B):
        A = A.permute(0, 3, 1, 2)  # N, H, W, C -> N, C, H, W
        B = B.permute(3, 2, 0, 1)  # H, W, C, F -> F, C, H, W
        C = torch.conv2d(A, B, stride=stride, padding=padding, dilation=dilation)
        C = C.permute(0, 2, 3, 1)  # N, C, H, W -> N, H, W, C
        return C

    return main


def get_best_config(N, C, H, W, F, K, S, D, P, with_roller):
    KH, KW = K, K
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1

    dtype = "float16"
    accum_dtype = "float"
    is_hopper = check_hopper()

    def kernel(
        block_M=None,
        block_N=None,
        block_K=None,
        num_stages=None,
        thread_num=None,
    ):

        @T.prim_func
        def main(
                data: T.Tensor((N, H, W, C), dtype),
                kernel: T.Tensor((KH, KW, C, F), dtype),
                out: T.Tensor((N, OH, OW, F), dtype),
        ):
            with T.Kernel(
                    T.ceildiv(F, block_N), T.ceildiv(N * OH * OW, block_M),
                    threads=thread_num) as (bx, by):
                data_shared = T.alloc_shared((block_M, block_K), dtype)
                kernel_shared = T.alloc_shared((block_K, block_N), dtype)
                out_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                out_shared = T.alloc_shared((block_M, block_N), dtype)

                kernel_flat = T.Tensor((KH * KW * C, F), dtype, kernel.data)
                out_flat = T.Tensor((N * OH * OW, F), dtype, out.data)

                T.annotate_layout({
                    out_shared: tilelang.layout.make_swizzled_layout(out_shared),
                    data_shared: tilelang.layout.make_swizzled_layout(data_shared),
                    kernel_shared: tilelang.layout.make_swizzled_layout(kernel_shared),
                })

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
                            in_bound = ((access_h >= 0) and (access_w >= 0) and (access_h < H) and
                                        (access_w < W))
                            data_shared[i, j] = T.if_then_else(
                                in_bound, data[m // (OH * OW), access_h, access_w, k % C], 0)
                    T.copy(kernel_flat[k_iter * block_K, bx * block_N], kernel_shared)
                    T.gemm(data_shared, kernel_shared, out_local)

                T.copy(out_local, out_shared)
                T.copy(out_shared, out_flat[by * block_M, bx * block_N])

        return main

    autotuner = AutoTuner.from_kernel(
        kernel=kernel, configs=get_configs(N, C, H, W, F, K, S, D, P,
                                           with_roller)).set_compile_args(
                                               out_idx=[2],
                                               supply_type=tilelang.TensorSupplyType.Integer,
                                               ref_prog=ref_program(S, P, D),
                                               skip_check=False,
                                               target="auto",
                                           )
    return autotuner.run(warmup=10, rep=10)


def convolution(N,
                C,
                H,
                W,
                F,
                K,
                S,
                D,
                P,
                block_M,
                block_N,
                block_K,
                num_stages,
                threads,
                dtype="float16",
                accum_dtype="float"):
    KH, KW = K, K
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
    dtype = "float16"
    accum_dtype = "float"
    is_hopper = check_hopper()

    @T.prim_func
    def main(
            data: T.Tensor((N, H, W, C), dtype),
            kernel: T.Tensor((KH, KW, C, F), dtype),
            out: T.Tensor((N, OH, OW, F), dtype),
    ):
        with T.Kernel(
                T.ceildiv(F, block_N), T.ceildiv(N * OH * OW, block_M),
                threads=threads) as (bx, by):
            data_shared = T.alloc_shared((block_M, block_K), dtype)
            kernel_shared = T.alloc_shared((block_K, block_N), dtype)
            out_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            out_shared = T.alloc_shared((block_M, block_N), dtype)

            kernel_flat = T.Tensor((KH * KW * C, F), dtype, kernel.data)
            out_flat = T.Tensor((N * OH * OW, F), dtype, out.data)

            T.annotate_layout({
                out_shared: tilelang.layout.make_swizzled_layout(out_shared),
                data_shared: tilelang.layout.make_swizzled_layout(data_shared),
                kernel_shared: tilelang.layout.make_swizzled_layout(kernel_shared),
            })

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
                        in_bound = ((access_h >= 0) and (access_w >= 0) and (access_h < H) and
                                    (access_w < W))
                        data_shared[i, j] = T.if_then_else(
                            in_bound, data[m // (OH * OW), access_h, access_w, k % C], 0)
                T.copy(kernel_flat[k_iter * block_K, bx * block_N], kernel_shared)
                T.gemm(data_shared, kernel_shared, out_local)

            T.copy(out_local, out_shared)
            T.copy(out_shared, out_flat[by * block_M, bx * block_N])

    return main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=128, help='n')
    parser.add_argument('--c', type=int, default=128, help='c')
    parser.add_argument('--h', type=int, default=64, help='h')
    parser.add_argument('--w', type=int, default=64, help='w')
    parser.add_argument('--f', type=int, default=128, help='f')
    parser.add_argument('--k', type=int, default=3, help='k')
    parser.add_argument('--s', type=int, default=1, help='s')
    parser.add_argument('--d', type=int, default=1, help='d')
    parser.add_argument('--p', type=int, default=1, help='p')
    parser.add_argument(
        "--use_autotune",
        action="store_true",
        default=True,
        help="Whether to use autotune for matmul configs")
    parser.add_argument(
        "--with_roller",
        action="store_true",
        default=True,
        help="Whether to enable BitBLAS roller for search space")

    args = parser.parse_args()
    N, C, H, W, F, K, S, D, P = args.n, args.c, args.h, args.w, args.f, args.k, args.s, args.d, args.p
    a = torch.randn(N, H, W, C).cuda().half()
    b = torch.randn(K, K, C, F).cuda().half()
    use_autotune = args.use_autotune
    with_roller = args.with_roller
    if use_autotune:
        result = get_best_config(N, C, H, W, F, K, S, D, P, with_roller)
        print(f"best latency {result.latency}")
        kernel = result.kernel
    else:
        kernel = tilelang.compile(
            convolution(N, C, H, W, F, K, S, D, P, 64, 128, 32, 3, 256), out_idx=[2])

    out_c = kernel(a, b)
    ref_c = ref_program(S, P, D)(a, b)
    torch.testing.assert_close(out_c, ref_c, rtol=1e-2, atol=1e-2)

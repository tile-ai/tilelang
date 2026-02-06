import torch
import argparse
import itertools
import tilelang
import tilelang.language as T


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


def get_configs():
    block_M = [64, 128, 256]
    block_N = [64, 128, 256]
    block_K = [32, 64]
    num_stages = [0, 1, 2, 3]
    thread_num = [128, 256]
    enable_rasterization = [True, False]
    _configs = list(
        itertools.product(
            block_M,
            block_N,
            block_K,
            num_stages,
            thread_num,
            enable_rasterization,
        )
    )

    configs = [
        {
            "block_M": c[0],
            "block_N": c[1],
            "block_K": c[2],
            "num_stages": c[3],
            "thread_num": c[4],
            "enable_rasteration": c[5],  # keep param name for backward-compat
        }
        for c in _configs
    ]
    return configs


def get_heuristic_config() -> dict:
    # Get CUDA device properties
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.cuda.current_device()
    sm_major, sm_minor = torch.cuda.get_device_capability(device)
    sm_version = sm_major * 10 + sm_minor
    print(f"CUDA device capability: {sm_version}")
    if sm_version in {80}:
        return {"block_M": 128, "block_N": 256, "block_K": 32, "num_stages": 2, "thread_num": 128, "enable_rasteration": True}
    elif sm_version in {90}:
        return {"block_M": 128, "block_N": 256, "block_K": 64, "num_stages": 3, "thread_num": 256, "enable_rasteration": True}
    else:
        return {"block_M": 128, "block_N": 256, "block_K": 32, "num_stages": 0, "thread_num": 128, "enable_rasteration": True}


@tilelang.autotune(configs=get_configs())
@tilelang.jit(out_idx=[2])
def convolution(
    N, C, H, W, F, K, S, D, P, block_M, block_N, block_K, num_stages, thread_num, enable_rasteration, dtype=T.float16, accum_dtype=T.float32
):
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
        with T.Kernel(T.ceildiv(F, block_N), T.ceildiv(N * OH * OW, block_M), threads=thread_num) as (bx, by):
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

            if is_hopper:
                T.copy(out_local, out_shared)
                T.copy(out_shared, out_flat[by * block_M, bx * block_N])
            else:
                T.copy(out_local, out_flat[by * block_M, bx * block_N])

    return main


def main(
    n: int = 128,
    c: int = 128,
    h: int = 64,
    w: int = 64,
    f: int = 128,
    k: int = 3,
    s: int = 1,
    d: int = 1,
    p: int = 1,
    use_autotune: bool = False,
    with_roller: bool = True,
):
    N, C, H, W, F, K, S, D, P = n, c, h, w, f, k, s, d, p
    ref_prog = ref_program(S, P, D)

    if use_autotune:
        kernel = convolution(N, C, H, W, F, K, S, D, P)
    else:
        config = get_heuristic_config()
        kernel = convolution(N, C, H, W, F, K, S, D, P, **config)

    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)
    tilelang_latency = profiler.do_bench()
    ref_latency = profiler.do_bench(ref_prog)
    profiler.assert_allclose(ref_prog, atol=1e-2, rtol=1e-2)
    print(f"TileLang latency: {tilelang_latency}")
    print(f"Ref latency: {ref_latency}")


def run_regression_perf(
    n: int = 128,
    c: int = 128,
    h: int = 64,
    w: int = 64,
    f: int = 128,
    k: int = 3,
    s: int = 1,
    d: int = 1,
    p: int = 1,
    use_autotune: bool = False,
    with_roller: bool = True,
):
    N, C, H, W, F, K, S, D, P = n, c, h, w, f, k, s, d, p
    config = get_heuristic_config()
    kernel = convolution(N, C, H, W, F, K, S, D, P, **config)
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)
    return profiler.do_bench(backend="cupti")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autotuned MatMul Benchmark")
    parser.add_argument("--n", type=int, default=128, help="n")
    parser.add_argument("--c", type=int, default=128, help="c")
    parser.add_argument("--h", type=int, default=64, help="h")
    parser.add_argument("--w", type=int, default=64, help="w")
    parser.add_argument("--f", type=int, default=128, help="f")
    parser.add_argument("--k", type=int, default=3, help="k")
    parser.add_argument("--s", type=int, default=1, help="s")
    parser.add_argument("--d", type=int, default=1, help="d")
    parser.add_argument("--p", type=int, default=1, help="p")
    parser.add_argument("--use_autotune", action="store_true", default=False, help="Whether to use autotune for matmul configs")
    parser.add_argument("--with_roller", action="store_true", default=True, help="Whether to enable BitBLAS roller for search space")
    args = parser.parse_args()
    main(args.n, args.c, args.h, args.w, args.f, args.k, args.s, args.d, args.p, args.use_autotune, args.with_roller)

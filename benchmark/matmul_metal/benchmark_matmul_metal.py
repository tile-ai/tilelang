import argparse
import logging
import time

import torch

import tilelang
import tilelang.language as T

logging.getLogger("tilelang").setLevel(logging.WARNING)

BLOCK_CONFIGS = [
    (16, 16, 16),
    (32, 32, 16),
    (32, 32, 32),
    (64, 64, 32),
]


@tilelang.jit
def matmul_simdgroup(M, N, K, block_M=64, block_N=64, block_K=32, dtype=T.float16, accum_dtype=T.float32):

    @T.prim_func
    def gemm_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_kernel


def _tflops(M, N, K, seconds):
    return 2.0 * M * N * K / seconds / 1e12


def _bench(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / repeats


def bench_torch_mps(M, N, K, warmup, repeats):
    a = torch.randn(M, K, dtype=torch.float16, device="mps")
    b = torch.randn(K, N, dtype=torch.float16, device="mps")
    avg_s = _bench(lambda: torch.mm(a, b), warmup, repeats)
    return _tflops(M, N, K, avg_s)


def bench_tilelang(M, N, K, block_M, block_N, block_K, warmup, repeats):
    kernel = matmul_simdgroup(M, N, K, block_M, block_N, block_K)
    a = torch.randn(M, K, dtype=torch.float16, device="mps")
    b = torch.randn(K, N, dtype=torch.float16, device="mps")
    c = torch.zeros(M, N, dtype=torch.float32, device="mps")
    avg_s = _bench(lambda: kernel(a, b, c), warmup, repeats)
    return _tflops(M, N, K, avg_s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metal GEMM Benchmark (simdgroup)")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--sweep", action="store_true", help="Sweep all block configs instead of using default (64,64,32)")
    args = parser.parse_args()

    M, N, K = args.m, args.n, args.k

    print(f"torch:    {torch.__version__}")
    print(f"tilelang: {tilelang.__version__}")
    print(f"MPS:      {torch.backends.mps.is_available()}")
    print(f"M={M}, N={N}, K={K}, warmup={args.warmup}, repeats={args.repeats}")
    print()

    ref_tflops = bench_torch_mps(M, N, K, args.warmup, args.repeats)
    print(f"PyTorch MPS (torch.mm fp16): {ref_tflops:.1f} TFLOPS")
    print()

    configs = BLOCK_CONFIGS if args.sweep else [(64, 64, 32)]

    print(f"{'block (M,N,K)':>16s} | {'TileLang':>14s} | {'Ratio':>6s}")
    print("-" * 44)

    best_tflops = 0.0
    best_config = configs[0]
    for bM, bN, bK in configs:
        try:
            tl = bench_tilelang(M, N, K, bM, bN, bK, args.warmup, args.repeats)
            ratio = tl / ref_tflops * 100
            tag = ""
            if tl > best_tflops:
                best_tflops = tl
                best_config = (bM, bN, bK)
            print(f"{f'({bM},{bN},{bK})':>16s} | {tl:>10.1f} TFLOPS | {ratio:>5.0f}%")
        except Exception as e:
            print(f"{f'({bM},{bN},{bK})':>16s} | {'FAILED':>14s} | {e}")

    if args.sweep:
        print()
        print(f"Best config: {best_config}")
        print(f"Best TFlops: {best_tflops:.1f}")
        print(f"Reference TFlops (PyTorch MPS): {ref_tflops:.1f}")

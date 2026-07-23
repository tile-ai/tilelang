"""AutoTuning GEMM with early_stop.

Demonstrates the early_stop feature that skips clearly suboptimal configs
during tuning, significantly reducing total tuning wall time.

Two usage modes:
- API mode: AutoTuner.from_kernel() with early_stop=True
- Decorator mode: @autotune(..., early_stop=True) + @tilelang.jit
"""

import os
import time
import tilelang as tl
import tilelang.language as T
from tilelang.autotuner import AutoTuner, autotune

os.environ["TILELANG_DISABLE_CACHE"] = "1"

M, N, K = 4096, 4096, 4096


def get_configs():
    """Configs with varying tile sizes to produce large performance gaps."""
    configs = []
    tile_sizes = [(128, 128), (64, 64), (32, 64), (64, 32), (32, 32)]
    for bm, bn in tile_sizes:
        for ns in [2, 1]:
            configs.append(
                {
                    "block_M": bm,
                    "block_N": bn,
                    "block_K": 32,
                    "num_stages": ns,
                    "thread_num": 128,
                }
            )
    return configs


def make_kernel():
    """GEMM kernel factory for API mode autotuning."""

    def kernel(block_M=None, block_N=None, block_K=None, num_stages=None, thread_num=None):
        dtype = T.float16
        accum_dtype = T.float32

        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.use_swizzle(panel_size=10)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                T.copy(C_local, C[by * block_M, bx * block_N])

        return main

    return kernel


# --- API mode ---


def run_api_mode():
    """Compare tuning wall time with and without early_stop."""
    configs = get_configs()
    print(f"Matrix: {M}x{N}x{K}, configs: {len(configs)}")
    print()

    results = {}
    for early_stop in [False, True]:
        label = "early_stop=True" if early_stop else "baseline"
        autotuner = (
            AutoTuner.from_kernel(kernel=make_kernel(), configs=configs).set_compile_args(out_idx=[-1], target="auto").set_profile_args()
        )
        start = time.perf_counter()
        result = autotuner.run(warmup=5, rep=50, early_stop=early_stop, early_stop_factor=1.2)
        wall = time.perf_counter() - start
        results[label] = (result, wall)
        print(f"[{label}] latency={result.latency:.4f}ms, wall={wall:.2f}s, config={result.config}")

    print()
    t_base = results["baseline"][1]
    t_early = results["early_stop=True"][1]
    print(f"Time saved: {t_base - t_early:.2f}s ({(t_base - t_early) / t_base * 100:.1f}%)")


# --- Decorator mode ---


@autotune(
    configs=get_configs(),
    warmup=5,
    rep=50,
    early_stop=True,
    early_stop_factor=2.0,
)
@tl.jit(out_idx=[-1])
def matmul_early_stop(
    M,
    N,
    K,
    block_M=128,
    block_N=128,
    block_K=32,
    num_stages=2,
    thread_num=128,
):
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_decorator_mode():
    """Run decorator mode autotuning with early_stop."""
    print("Decorator mode: early_stop=True, early_stop_factor=2.0")
    start = time.perf_counter()
    kernel = matmul_early_stop(M, N, K)
    wall = time.perf_counter() - start
    print(f"  Wall time: {wall:.2f}s")
    print(f"  Kernel source length: {len(kernel.get_kernel_source())} chars")


if __name__ == "__main__":
    print("=" * 60)
    print("API mode")
    print("=" * 60)
    run_api_mode()
    print()
    print("=" * 60)
    print("Decorator mode")
    print("=" * 60)
    run_decorator_mode()

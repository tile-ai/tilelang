"""Example: AutoTuning with early_stop.

Demonstrates the early_stop feature with configs of varying tile sizes
to produce large performance gaps (5-20x), making early stop effective.

Two usage modes are shown:
1. API mode: using AutoTuner.from_kernel() explicitly
2. Decorator (wrapper) mode: using @autotune + @tilelang.jit decorators
"""

import argparse
import os
import time
import tilelang as tl
import tilelang.language as T
from tilelang.autotuner import AutoTuner, autotune

os.environ["TILELANG_DISABLE_CACHE"] = "1"


def ref_program(A, B):
    """Reference: C = A @ B^T."""
    return A @ B.T


def get_configs():
    """Generate configs with large performance gaps across tile sizes."""
    configs = []

    for ns in [2, 1]:
        configs.append({"block_M": 128, "block_N": 128, "block_K": 32, "num_stages": ns, "thread_num": 128})

    for ns in [2, 1]:
        configs.append({"block_M": 64, "block_N": 64, "block_K": 32, "num_stages": ns, "thread_num": 128})

    for ns in [2, 1]:
        configs.append({"block_M": 32, "block_N": 64, "block_K": 32, "num_stages": ns, "thread_num": 128})

    for ns in [2, 1]:
        configs.append({"block_M": 64, "block_N": 32, "block_K": 32, "num_stages": ns, "thread_num": 128})

    for ns in [2, 1]:
        configs.append({"block_M": 32, "block_N": 32, "block_K": 32, "num_stages": ns, "thread_num": 128})

    return configs


def make_kernel(M, N, K):
    """Create a GEMM kernel factory for autotuning."""

    def kernel(
        block_M=None,
        block_N=None,
        block_K=None,
        num_stages=None,
        thread_num=None,
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

    return kernel


def run_autotune(M, N, K, early_stop: bool, early_stop_factor: float = 1.2):
    """Run autotune with or without early_stop, return (result, wall_time)."""
    configs = get_configs()
    kernel = make_kernel(M, N, K)

    autotuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=configs)
        .set_compile_args(out_idx=[-1], target="auto")
        .set_profile_args(
            supply_type=tl.TensorSupplyType.Integer,
            ref_prog=ref_program,
            skip_check=True,
        )
    )

    start = time.perf_counter()
    result = autotuner.run(warmup=5, rep=50, early_stop=early_stop, early_stop_factor=early_stop_factor)
    wall_time = time.perf_counter() - start

    return result, wall_time


# ======================== Decorator (wrapper) mode ========================


@autotune(
    configs=get_configs(),
    warmup=5,
    rep=50,
    early_stop=True,
    early_stop_factor=2.0,
    supply_type=tl.TensorSupplyType.Integer,
    ref_prog=ref_program,
    skip_check=True,
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
    """GEMM kernel with early_stop enabled via decorator mode."""
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


def main_decorator(M: int = 4096, N: int = 4096, K: int = 4096):
    """Run decorator mode autotuning with early_stop and print results."""
    print("Running decorator mode with early_stop=True, early_stop_factor=2.0 ...")
    start = time.perf_counter()
    kernel = matmul_early_stop(M, N, K)
    wall_time = time.perf_counter() - start
    print(f"  Wall time:     {wall_time:.2f} s")
    print(f"  Kernel source: {len(kernel.get_kernel_source())} chars")


def main(M: int = 4096, N: int = 4096, K: int = 4096):
    """Compare autotune wall time with and without early_stop."""
    configs = get_configs()
    print(f"Matrix size: {M}x{N}x{K}")
    print(f"Total configs: {len(configs)}")
    print()

    print("=" * 60)
    print("Run 1: early_stop=False (baseline)")
    print("=" * 60)
    result_baseline, time_baseline = run_autotune(M, N, K, early_stop=False)
    print(f"\n  Best latency: {result_baseline.latency:.4f} ms")
    print(f"  Best config:  {result_baseline.config}")
    print(f"  Wall time:    {time_baseline:.2f} s")
    print()

    print("=" * 60)
    print("Run 2: early_stop=True")
    print("=" * 60)
    result_early, time_early = run_autotune(M, N, K, early_stop=True)
    print(f"\n  Best latency: {result_early.latency:.4f} ms")
    print(f"  Best config:  {result_early.config}")
    print(f"  Wall time:    {time_early:.2f} s")
    print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    speedup = time_baseline / time_early if time_early > 0 else float("inf")
    saved = time_baseline - time_early
    print(f"  Baseline wall time:    {time_baseline:.2f} s")
    print(f"  Early stop wall time:  {time_early:.2f} s")
    print(f"  Time saved:            {saved:.2f} s ({saved / time_baseline * 100:.1f}%)")
    print(f"  Speedup:               {speedup:.2f}x")
    print(f"  Best latency match:    {abs(result_baseline.latency - result_early.latency) < 0.1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoTune MatMul with early_stop")
    parser.add_argument("--m", type=int, default=4096, help="Matrix dimension M")
    parser.add_argument("--n", type=int, default=4096, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=4096, help="Matrix dimension K")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["api", "decorator", "both"],
        help="Which mode to run",
    )
    args = parser.parse_args()
    if args.mode in ("api", "both"):
        print("=" * 60)
        print("API mode")
        print("=" * 60)
        main(args.m, args.n, args.k)
    if args.mode in ("decorator", "both"):
        print("=" * 60)
        print("Decorator mode")
        print("=" * 60)
        main_decorator(args.m, args.n, args.k)

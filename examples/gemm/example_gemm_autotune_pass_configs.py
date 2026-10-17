"""Example: AutoTuning with per-config pass_configs.

This example demonstrates how to include pass_configs (compiler pass configurations)
as part of the auto-tuning search space, allowing the tuner to find the best
combination of kernel parameters AND compiler options.

Two usage modes are shown:
1. API mode: using AutoTuner.from_kernel() explicitly
2. Decorator (wrapper) mode: using @autotune + @tilelang.jit decorators
"""

import argparse
import itertools
import tilelang as tl
import tilelang.language as T
from tilelang.autotuner import AutoTuner, autotune
from tilelang.transform import PassConfigKey


def ref_program(A, B):
    """Reference: C = A @ B^T."""
    return A @ B.T


def get_configs():
    """Generate configs that include pass_configs variations."""
    block_M = [128]
    block_N = [128]
    block_K = [32]
    num_stages = [1, 2]
    thread_num = [128]
    warp_spec = [True, False]

    configs = []
    for bm, bn, bk, ns, tn, ws in itertools.product(block_M, block_N, block_K, num_stages, thread_num, warp_spec):
        config = {
            "block_M": bm,
            "block_N": bn,
            "block_K": bk,
            "num_stages": ns,
            "thread_num": tn,
            "pass_configs": {PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: ws},
        }
        configs.append(config)
    return configs


def get_best_config(M, N, K):
    """Run autotuning to find the best kernel config (API mode)."""

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

    autotuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=get_configs())
        .set_compile_args(
            out_idx=[-1],
            target="auto",
        )
        .set_profile_args(
            supply_type=tl.TensorSupplyType.Integer,
            ref_prog=ref_program,
            skip_check=False,
        )
    )
    return autotuner.run(warmup=3, rep=20, enable_grouped_compile=True)


def main(M: int = 4096, N: int = 4096, K: int = 4096):
    """Run API mode autotuning and print results."""
    result = get_best_config(M, N, K)
    print(f"Best config: {result.config}")
    print(f"Best latency: {result.latency} ms")
    print(f"TFlops: {2 * M * N * K / result.latency * 1e-9}")

    if result.ref_latency:
        print(f"Ref latency: {result.ref_latency} ms")
        print(f"Ref TFlops: {2 * M * N * K / result.ref_latency * 1e-9}")


# ======================== Decorator (wrapper) mode ========================


@autotune(configs=get_configs(), warmup=3, rep=20)
@tl.jit(out_idx=[-1])
def matmul_decorator(
    M,
    N,
    K,
    block_M=128,
    block_N=128,
    block_K=32,
    num_stages=2,
    thread_num=128,
):
    """GEMM kernel with per-config pass_configs in autotune search space (decorator mode)."""
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
    """Run decorator mode autotuning and print results."""
    kernel = matmul_decorator(M, N, K)
    print("[Decorator mode] Kernel compiled successfully")
    print(f"[Decorator mode] Kernel source length: {len(kernel.get_kernel_source())} chars")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoTune MatMul with pass_configs")
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

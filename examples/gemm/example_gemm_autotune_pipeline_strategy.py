"""Benchmark: Pipeline Copy Strategy autotune comparison.

Compares three pipeline copy strategies (occupancy/latency/balance) across
different workloads using per-config pass_configs. Each case is designed to
favor a specific strategy.

Usage:
    python example_gemm_autotune_pipeline_strategy.py
"""

import tilelang as tl
import tilelang.language as T
from tilelang.autotuner import AutoTuner
from tilelang.transform import PassConfigKey


def get_strategy_configs():
    return [{"pass_configs": {PassConfigKey.TL_PIPELINE_COPY_STRATEGY: s}} for s in ("occupancy", "latency", "balance")]


# ==================== Case 1: balance wins ====================


def balance_favored_kernel():
    """Small tiles + many A-only ops before B is used.

    Balance's Step 2 rotation consolidates copies and reduces stage count,
    improving occupancy on H200 (~10%+ faster than alternatives).
    """
    M, N, K = 4096, 4096, 4096
    block_M, block_N, block_K = 32, 32, 32
    num_stages, thread_num = 2, 128
    dtype, accum_dtype = T.float16, T.float32

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
            Acc1 = T.alloc_fragment((block_M, block_K), accum_dtype)
            Acc2 = T.alloc_fragment((block_M, block_K), accum_dtype)
            Acc3 = T.alloc_fragment((block_M, block_K), accum_dtype)
            Acc4 = T.alloc_fragment((block_M, block_K), accum_dtype)
            T.clear(C_local)
            T.clear(Acc1)
            T.clear(Acc2)
            T.clear(Acc3)
            T.clear(Acc4)

            for k in T.Pipelined(K // block_K, num_stages=num_stages):
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.copy(A[by * block_M, k * block_K], A_shared)
                for i, j in T.Parallel(block_M, block_K):
                    Acc1[i, j] += T.cast(A_shared[i, j], accum_dtype)
                for i, j in T.Parallel(block_M, block_K):
                    Acc2[i, j] += T.cast(A_shared[i, j], accum_dtype) * 0.5
                for i, j in T.Parallel(block_M, block_K):
                    Acc3[i, j] += T.cast(A_shared[i, j], accum_dtype) * 0.25
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                for i, j in T.Parallel(block_M, block_K):
                    Acc4[i, j] += T.cast(A_shared[i, j], accum_dtype) * 0.125

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


# ==================== Case 2: occupancy wins ====================


def occupancy_favored_kernel():
    """Large tiles + deep pipeline (num_stages=3).

    Occupancy preserves full prefetch depth while balance's rotation
    reduces it, hurting throughput on large workloads (~5% faster).
    """
    M, N, K = 8192, 8192, 8192
    block_M, block_N, block_K = 128, 128, 64
    num_stages, thread_num = 3, 128
    dtype, accum_dtype = T.float16, T.float32

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
            Acc1 = T.alloc_fragment((block_M, block_K), accum_dtype)
            Acc2 = T.alloc_fragment((block_M, block_K), accum_dtype)
            T.clear(C_local)
            T.clear(Acc1)
            T.clear(Acc2)

            for k in T.Pipelined(K // block_K, num_stages=num_stages):
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.copy(A[by * block_M, k * block_K], A_shared)
                for i, j in T.Parallel(block_M, block_K):
                    Acc1[i, j] += T.cast(A_shared[i, j], accum_dtype)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                for i, j in T.Parallel(block_M, block_K):
                    Acc2[i, j] += T.cast(A_shared[i, j], accum_dtype) * 0.5

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


# ==================== Case 3: latency wins ====================


def latency_favored_kernel():
    """Large tiles + num_stages=1 (no pipeline prefetch).

    Without pipeline staging, copy execution order directly determines
    when data arrives. Latency places A first (matching use order, ~2.5%).
    """
    M, N, K = 4096, 4096, 16384
    block_M, block_N, block_K = 128, 128, 64
    num_stages, thread_num = 1, 128
    dtype, accum_dtype = T.float16, T.float32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, K), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_K), accum_dtype)
            Acc1 = T.alloc_fragment((block_N, block_K), accum_dtype)
            Acc2 = T.alloc_fragment((block_M, block_K), accum_dtype)
            T.clear(C_local)
            T.clear(Acc1)
            T.clear(Acc2)

            for k in T.Pipelined(K // block_K, num_stages=num_stages):
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.copy(A[by * block_M, k * block_K], A_shared)
                for i, j in T.Parallel(block_M, block_K):
                    C_local[i, j] += T.cast(A_shared[i, j], accum_dtype)
                for i, j in T.Parallel(block_N, block_K):
                    Acc1[i, j] += T.cast(B_shared[i, j], accum_dtype)
                for i, j in T.Parallel(block_M, block_K):
                    Acc2[i, j] += T.cast(A_shared[i, j], accum_dtype) * 0.125

            T.copy(C_local, C[by * block_M, bx * block_K])

    return main


# ==================== Runner ====================

CASES = [
    ("balance-favored", balance_favored_kernel, -1),
    ("occupancy-favored", occupancy_favored_kernel, -1),
    ("latency-favored", latency_favored_kernel, -1),
]


def run_case(name, kernel_fn, out_idx):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    autotuner = (
        AutoTuner.from_kernel(kernel=kernel_fn, configs=get_strategy_configs())
        .set_compile_args(
            out_idx=[out_idx] if isinstance(out_idx, int) else out_idx,
            target="auto",
            pass_configs={
                PassConfigKey.TL_DISABLE_TMA_LOWER: True,
                PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            },
        )
        .set_profile_args(
            supply_type=tl.TensorSupplyType.Integer,
        )
    )
    result = autotuner.run(warmup=10, rep=50)
    print(f"  Winner: {result.config.get('pass_configs', {}).get(PassConfigKey.TL_PIPELINE_COPY_STRATEGY, '?')}")
    print(f"  Latency: {result.latency:.4f} ms")
    return result


if __name__ == "__main__":
    for name, kernel_fn, out_idx in CASES:
        run_case(name, kernel_fn, out_idx)

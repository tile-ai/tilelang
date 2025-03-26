# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import argparse
import torch
import tilelang as tl
import tilelang.language as T
from tvm import DataType
from functools import partial
from tilelang.autotuner import autotune, jit
from tilelang.carver.template import GEMVTemplate
from tilelang.carver.arch import CUDA
from tilelang.carver.roller.rasterization import NoRasterization


def naive_gemv(
    N: int,
    K: int,
    BLOCK_N: int,
    BLOCK_K: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    """
    Naive GEMV following GEMM tiling strategy in SIMD manner.
    """

    @T.prim_func
    def main(
            A: T.Buffer((K,), dtype),
            B: T.Buffer((N, K), dtype),
            C: T.Buffer((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N)) as bn:
            tn = T.get_thread_binding(0)  # tn = threadIdx.x
            A_shared = T.alloc_shared((BLOCK_K,), dtype)
            B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
            C_reg = T.alloc_local((1,), accum_dtype)
            T.clear(C_reg)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for tk in T.serial(BLOCK_K):
                    A_shared[tk] = A[bk * BLOCK_K + tk]
                    B_shared[tn, tk] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk]
                for tk in T.serial(BLOCK_K):
                    C_reg[0] += A_shared[tk].astype(accum_dtype) * B_shared[tn,
                                                                            tk].astype(accum_dtype)
            C[bn * BLOCK_N + tn] = C_reg[0]

    return main


def naive_splitk_gemv(
    N: int,
    K: int,
    BLOCK_N: int,
    BLOCK_K: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    """
    Naive GEMV following GEMM tiling strategy in SIMD manner.
    """

    @T.prim_func
    def main(
            A: T.Buffer((K,), dtype),
            B: T.Buffer((N, K), dtype),
            C: T.Buffer((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, BLOCK_K)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_shared = T.alloc_shared((BLOCK_K,), dtype)
            B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
            C_shared = T.alloc_shared((BLOCK_N,), accum_dtype)
            if tk == 0:
                C_shared[tn] = 0
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                A_shared[tk] = A[bk * BLOCK_K + tk]
                B_shared[tn, tk] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk]
                T.atomic_add(C_shared[tn],
                             A_shared[tk].astype(accum_dtype) * B_shared[tn, tk].astype(accum_dtype))  # AtomicAdd as defined in src/tl_templates/cuda/common.h
            C[bn * BLOCK_N + tn] = C_shared[tn]

    return main


def splitk_gemv(
    N: int,
    K: int,
    BLOCK_N: int,
    BLOCK_K: int,
    reduce_threads: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    """
    Naive GEMV following GEMM tiling strategy in SIMD manner.
    """

    @T.prim_func
    def main(
            A: T.Buffer((K,), dtype),
            B: T.Buffer((N, K), dtype),
            C: T.Buffer((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            TILE_K = T.ceildiv(BLOCK_K, reduce_threads)
            A_shared = T.alloc_shared((BLOCK_K,), dtype)
            B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
            C_shared = T.alloc_shared((BLOCK_N,), accum_dtype)
            if tk == 0:
                C_shared[tn] = 0
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for k in T.serial(TILE_K):
                    A_shared[tk * TILE_K + k] = A[bk * BLOCK_K + tk * TILE_K + k]
                    B_shared[tn, tk * TILE_K + k] = B[bn * BLOCK_N + tn,
                                                      bk * BLOCK_K + tk * TILE_K + k]
                C_reg = T.alloc_local((1,), accum_dtype)
                T.clear(C_reg)
                for k in T.serial(TILE_K):
                    C_reg[0] += A_shared[tk * TILE_K +
                                         k].astype(accum_dtype) * B_shared[tn, tk * TILE_K +
                                                                           k].astype(accum_dtype)
                T.atomic_add(C_shared[tn], C_reg[0])
            C[bn * BLOCK_N + tn] = C_shared[tn]

    return main


def splitk_gemv_vectorized(
    N: int,
    K: int,
    BLOCK_N: int,
    reduce_threads: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
    BLOCK_K = reduce_threads * TILE_K

    @T.prim_func
    def main(
            A: T.Buffer((K,), dtype),
            B: T.Buffer((N, K), dtype),
            C: T.Buffer((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            TILE_K = T.ceildiv(BLOCK_K, reduce_threads)
            A_shared = T.alloc_shared((BLOCK_K,), dtype)
            B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
            C_shared = T.alloc_shared((BLOCK_N,), accum_dtype)
            if tk == 0:
                C_shared[tn] = 0
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for k in T.vectorized(TILE_K):
                    A_shared[tk * TILE_K + k] = A[bk * BLOCK_K + tk * TILE_K + k]
                    B_shared[tn, tk * TILE_K + k] = B[bn * BLOCK_N + tn,
                                                      bk * BLOCK_K + tk * TILE_K + k]
                C_reg = T.alloc_local((1,), accum_dtype)
                T.clear(C_reg)
                for k in T.serial(TILE_K):
                    C_reg[0] += A_shared[tk * TILE_K +
                                         k].astype(accum_dtype) * B_shared[tn, tk * TILE_K +
                                                                           k].astype(accum_dtype)
                T.atomic_add(C_shared[tn], C_reg[0])
            C[bn * BLOCK_N + tn] = C_shared[tn]

    return main


def check_correctness_and_bench(kernel, N, K, bench_ref=False):
    kernel = tl.compile(kernel, out_idx=-1)
    a = torch.randn(K).cuda().half()
    b = torch.randn(N, K).cuda().half()
    out_c = kernel(a, b)
    ref_c = a @ b.T
    profiler = kernel.get_profiler()
    profiler.assert_allclose(lambda x, y: x @ y.T, atol=1e-2, rtol=1e-2)
    if bench_ref:
        latency = profiler.do_bench(lambda x, y: x @ y.T, warmup=500)
        print(f"Torch Latency: {latency:.2f} ms")
    latency = profiler.do_bench(kernel, warmup=500)
    print(f"TileLang Latency: {latency:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEMV Example")
    parser.add_argument("--n", type=int, default=1024, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=1024, help="Matrix dimension K")
    args = parser.parse_args()
    N, K = args.n, args.k
    check_correctness_and_bench(naive_gemv(N, K, 128, 128), N, K, bench_ref=True)
    check_correctness_and_bench(naive_splitk_gemv(N, K, 32, 32), N, K)
    check_correctness_and_bench(splitk_gemv(N, K, 32, 32, 32), N, K)
    check_correctness_and_bench(splitk_gemv_vectorized(N, K, 32, 16), N, K)
    print("Test passed!")

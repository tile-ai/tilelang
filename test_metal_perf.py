import tilelang
from tilelang import tvm as tvm
from time import sleep
import tilelang.testing
import tilelang.language as T
import json
import torch
import os


def get_configs():
    bs = [8, 16, 32]
    r = [{}]
    for k in "MNK":
        r = [{f"block_{k}": b, **i} for i in r for b in bs]
    return r


@tilelang.autotune(configs=get_configs())
@tilelang.jit(execution_backend="torch", out_idx=-1)
def matmul(M, N, K, block_M, block_N, block_K, dtype="float32", accum_dtype="float"):

    @T.prim_func
    def gemm(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
                    bx,
                    by,
                ):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared, coalesced_width=2)
                T.copy(B[ko * block_K, bx * block_N], B_shared, coalesced_width=2)

                for i, j, k in T.Parallel(block_M, block_N, block_K):
                    C_local[i, j] += A_shared[i, k] * B_shared[k, j]

            T.copy(C_local, C[by * block_M, bx * block_N], coalesced_width=2)

    return gemm


SS = [1024, 2048, 4096]
TESTS = [(a, b, c) for a in SS for b in SS for c in SS]


def benchmark(f, n, *args, **kwargs):
    # trigger jit
    f(*args, **kwargs)

    torch.mps.synchronize()
    with torch.mps.profiler.profile(mode="interval,event", wait_until_completed=True):
        start = torch.mps.Event(enable_timing=True)
        end = torch.mps.Event(enable_timing=True)
        start.record()

        for _ in range(n):
            f(*args, **kwargs)

        end.record()

        start.synchronize()
        end.synchronize()

        return start.elapsed_time(end) / 1000


def tune():
    results = {}
    for test in TESTS:
        m, n, k = test

        size_key = f"{m}_{n}_{k}"
        for dtype in ("float32", "float16"):
            conf_key = f"{size_key},{dtype}"
            r = []

            torch_dtype = getattr(torch, dtype)

            a = torch.randn(m, k, device="mps", dtype=torch_dtype)
            b = torch.randn(k, n, device="mps", dtype=torch_dtype)
            c = torch.zeros(m, n, device="mps", dtype=torch_dtype)

            torch_add = lambda: torch.matmul(a, b, out=c)
            torch_add()
            r.append(benchmark(torch_add, n=100))

            jit_kernel = matmul(m, n, k, dtype=dtype, accum_dtype="float")

            # warmup
            tl_add = lambda: jit_kernel(a, b, c)
            tl_add()

            torch_add = lambda: torch.matmul(a, b, out=c)
            torch_add()

            r.append(benchmark(tl_add, n=100))

            results[conf_key] = r[0], r[1]

    try:
        # from tabulate import tabulate

        # print(
        #     tabulate(
        #         [(key, tl_1, tl_2, torch, torch_native)
        #          for (key, (tl_1, tl_2, torch, torch_native)) in results.items()],
        #         headers=["config", "tl_run1", "tl_run2", "torch", "torch_native"],
        #     ))
        print(json.dumps(results, indent=2))
    except ImportError:
        print(results)


if __name__ == "__main__":
    tune()

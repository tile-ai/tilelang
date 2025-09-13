import tilelang
from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T
import torch

from tilelang.profiler.bench import do_bench


def get_configs():
    # return [
    #     {'block_M': 16, 'block_N': 16, 'block_K': 16,}
    # ]

    bs = [16, 32, 64, 128, 256]
    bs = [16, 32]
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
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
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


TESTS = [(1024, 1024, 1024)]


def benchmark(f, n, *args, **kwargs):
    # trigger jit
    f(*args, **kwargs)

    torch.mps.synchronize()
    start = torch.mps.Event(enable_timing=True)
    end = torch.mps.Event(enable_timing=True)
    start.record()

    for _ in range(n):
        f(*args, **kwargs)

    end.record()
    # end.synchronize()

    return start.elapsed_time(end) / 1000


if __name__ == "__main__":
    for test in TESTS:
        m, n, k = test

        for dtype in ("float32", "float16"):
            torch_dtype = getattr(torch, dtype)

            a = torch.randn(m, k, device="mps", dtype=torch_dtype)
            b = torch.randn(k, n, device="mps", dtype=torch_dtype)
            c = torch.zeros(m, n, device="mps", dtype=torch_dtype)

            torch_add = lambda: torch.matmul(a, b, out=c)
            torch_add()
            print(benchmark(torch_add, n=100))

            jit_kernel = matmul(m, n, k, dtype=dtype, accum_dtype="float")

            # warmup
            tl_add = lambda: jit_kernel(a, b, c)
            tl_add()

            torch_add = lambda: torch.matmul(a, b, out=c)
            torch_add()

            print(benchmark(tl_add, n=100))

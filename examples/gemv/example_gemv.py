import argparse
import itertools
import tilelang as tl
import tilelang.language as T
from tvm import DataType
from tilelang.autotuner import autotune
from tilelang import jit


def ref_program(A, B):
    return A @ B.T


@tl.jit(out_idx=[-1])
def naive_gemv(
    N: int,
    K: int,
    BLOCK_N: int,
    BLOCK_K: int,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float,
):
    @T.prim_func
    def main(
        A: T.Tensor((K,), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((N,), dtype),
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
                    C_reg[0] += A_shared[tk].astype(accum_dtype) * B_shared[tn, tk].astype(accum_dtype)
            C[bn * BLOCK_N + tn] = C_reg[0]

    return main


@tl.jit(out_idx=[-1])
def naive_splitk_gemv(
    N: int,
    K: int,
    BLOCK_N: int,
    BLOCK_K: int,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float,
):
    @T.prim_func
    def main(
        A: T.Tensor((K,), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, BLOCK_K)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local = T.alloc_local((1,), dtype)
            B_local = T.alloc_local((1,), dtype)
            C_accum = T.alloc_local((1,), accum_dtype)
            C_shared = T.alloc_shared((BLOCK_N,), accum_dtype)
            if tk == 0:
                C_shared[tn] = 0
            T.clear(C_accum)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                A_local[0] = A[bk * BLOCK_K + tk]
                B_local[0] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk]
                C_accum[0] += A_local[0].astype(accum_dtype) * B_local[0].astype(accum_dtype)
            T.atomic_add(C_shared[tn], C_accum[0])
            C[bn * BLOCK_N + tn] = C_shared[tn]

    return main


@tl.jit(out_idx=[-1])
def splitk_gemv(
    N: int,
    K: int,
    BLOCK_N: int,
    BLOCK_K: int,
    reduce_threads: int,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float,
):
    TILE_K = T.ceildiv(BLOCK_K, reduce_threads)

    @T.prim_func
    def main(
        A: T.Tensor((K,), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local = T.alloc_local((TILE_K,), dtype)
            B_local = T.alloc_local((TILE_K,), dtype)
            C_shared = T.alloc_shared((BLOCK_N,), accum_dtype)
            C_accum = T.alloc_local((1,), accum_dtype)
            if tk == 0:
                C_shared[tn] = 0
            T.clear(C_accum)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for k in T.serial(TILE_K):
                    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                    B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                for k in T.serial(TILE_K):
                    C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
            T.atomic_add(C_shared[tn], C_accum[0])
            C[bn * BLOCK_N + tn] = C_shared[tn]

    return main


@tl.jit(out_idx=[-1])
def splitk_gemv_vectorized(
    N: int,
    K: int,
    BLOCK_N: int,
    reduce_threads: int,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float,
):
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
    BLOCK_K = reduce_threads * TILE_K

    @T.prim_func
    def main(
        A: T.Tensor((K,), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local = T.alloc_local((TILE_K,), dtype)
            B_local = T.alloc_local((TILE_K,), dtype)
            C_shared = T.alloc_shared((BLOCK_N,), accum_dtype)
            C_accum = T.alloc_local((1,), accum_dtype)
            if tk == 0:
                C_shared[tn] = 0
            T.clear(C_accum)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for k in T.vectorized(TILE_K):
                    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                    B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                for k in T.serial(TILE_K):
                    C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
            T.atomic_add(C_shared[tn], C_accum[0])
            C[bn * BLOCK_N + tn] = C_shared[tn]

    return main


@tl.jit(out_idx=[-1])
def splitk_gemv_vectorized_tvm(
    N: int,
    K: int,
    BLOCK_N: int,
    reduce_threads: int,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float,
):
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
    BLOCK_K = reduce_threads * TILE_K

    @T.prim_func
    def main(
        A: T.Tensor((K,), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local = T.alloc_local((TILE_K,), dtype)
            B_local = T.alloc_local((TILE_K,), dtype)
            C_accum = T.alloc_local((1,), accum_dtype)

            T.clear(C_accum)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for k in T.vectorized(TILE_K):
                    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                    B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                for k in T.serial(TILE_K):
                    C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
            C_reduced = T.alloc_local((1,), accum_dtype)
            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        C_accum[0],
                        True,
                        C_reduced[0],
                        tk,
                        dtype="handle",
                    )
                )

            C[bn * BLOCK_N + tn] = C_reduced[0]

    return main


def get_block_template_configs():
    iter_params = dict(
        block_M=[2, 4, 8, 32, 64, 128], block_N=[2, 4, 8, 32, 64, 128], num_stages=[0, 1, 2, 3, 4], threads=[32, 64, 128, 256]
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


@tl.autotune(
    configs=get_block_template_configs(),
    warmup=3,
    rep=20,
)
@tl.jit(
    pass_configs={
        tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
    out_idx=[2],
)
def gemv_alloc_reducer(
    M, N, block_M=128, block_N=128, num_stages=2, threads=256, dtype: T.dtype = T.float16, accum_dtype: T.dtype = T.float
):
    @T.prim_func
    def main(a: T.Tensor((M, N), dtype), x: T.Tensor(N, dtype), o: T.Tensor(M, dtype)):  # type: ignore
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as i0_m:
            o_reducer = T.alloc_reducer(block_M, accum_dtype, replication="all")
            T.clear(o_reducer)
            for i0_n in T.Pipelined(T.ceildiv(N, block_N), num_stages=num_stages):
                a_smem = T.alloc_shared((block_M, block_N), dtype)
                T.copy(a[i0_m * block_M, i0_n * block_N], a_smem)
                a_frag = T.alloc_fragment((block_M, block_N), dtype)
                T.copy(a_smem, a_frag)
                x_frag = T.alloc_fragment(block_N, dtype)
                T.copy(x[i0_n * block_N], x_frag)
                for i1_m, i1_n in T.Parallel(block_M, block_N):
                    o_reducer[i1_m] += a_frag[i1_m, i1_n] * x_frag[i1_n]
            T.finalize_reducer(o_reducer)
            T.copy(o_reducer, o[i0_m * block_M])

    return main


def get_thread_template_configs():
    iter_params = dict(BLOCK_N=[2, 4, 8, 32, 64, 128], reduce_threads=[4, 8, 32])
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


@autotune(
    configs=get_thread_template_configs(),
    warmup=3,
    rep=20,
)
@jit(
    out_idx=[-1],
    target="auto",
)
def get_autotuned_kernel(
    N,
    K,
    BLOCK_N=None,
    reduce_threads=None,
):
    dtype = T.float16
    accum_dtype = T.float32
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
    BLOCK_K = reduce_threads * TILE_K

    @T.prim_func
    def main(
        A: T.Tensor((K,), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local = T.alloc_local((TILE_K,), dtype)
            B_local = T.alloc_local((TILE_K,), dtype)
            C_accum = T.alloc_local((1,), accum_dtype)

            T.clear(C_accum)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for k in T.vectorized(TILE_K):
                    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                    B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                for k in T.serial(TILE_K):
                    C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
            C_reduced = T.alloc_local((1,), accum_dtype)
            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        C_accum[0],
                        True,
                        C_reduced[0],
                        tk,
                        dtype="handle",
                    )
                )

            C[bn * BLOCK_N + tn] = C_reduced[0]

    return main


def check_correctness_and_bench(kernel, N, K, do_bench=True):
    profiler = kernel.get_profiler()
    profiler.assert_allclose(lambda x, y: x @ y.T, atol=1e-2, rtol=1e-2)
    if do_bench:
        latency = profiler.do_bench(lambda x, y: x @ y.T, warmup=50)
        print(f"Torch Latency: {latency} ms")
        latency = profiler.do_bench(kernel, warmup=50)
        print(f"TileLang Latency: {latency} ms\n")


def main(do_bench: bool = True):
    parser = argparse.ArgumentParser(description="GEMV Example")
    parser.add_argument("--n", type=int, default=1024, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=1024, help="Matrix dimension K")
    args, _ = parser.parse_known_args()
    N, K = args.n, args.k
    check_correctness_and_bench(naive_gemv(N, K, 128, 128), N, K, do_bench=do_bench)
    check_correctness_and_bench(naive_splitk_gemv(N, K, 32, 32), N, K, do_bench=do_bench)
    check_correctness_and_bench(splitk_gemv(N, K, 32, 32, 32), N, K, do_bench=do_bench)
    check_correctness_and_bench(splitk_gemv_vectorized(N, K, 2, 32), N, K, do_bench=do_bench)
    check_correctness_and_bench(splitk_gemv_vectorized_tvm(N, K, 2, 32), N, K, do_bench=do_bench)
    check_correctness_and_bench(gemv_alloc_reducer(N, K, block_M=128, block_N=128), N, K, do_bench=do_bench)

    print("Test passed!")

    if do_bench:
        best_result = get_autotuned_kernel(N, K)
        best_config = best_result.config
        kernel = splitk_gemv_vectorized_tvm(N, K, **best_config)
        profiler = kernel.get_profiler()
        latency = profiler.do_bench(lambda x, y: x @ y.T, warmup=500)
        print(f"Torch Latency: {latency} ms")
        tilelang_thread_latency = profiler.do_bench(kernel, warmup=500)
        print(f"TileLang SIMT Latency: {tilelang_thread_latency} ms\n")
        kernel = gemv_alloc_reducer(N, K)
        profiler = kernel.get_profiler()
        tilelang_tile_latency = profiler.do_bench(kernel, warmup=500)
        print(f"TileLang BlockReduce Latency: {tilelang_tile_latency} ms\n")


def run_regression_perf():
    N, K = 1024, 1024
    latency = 0.0
    kernel_list = [
        naive_gemv(N, K, 128, 128),
        naive_splitk_gemv(N, K, 32, 32),
        splitk_gemv(N, K, 32, 32, 32),
        splitk_gemv_vectorized(N, K, 2, 32),
        splitk_gemv_vectorized_tvm(N, K, 2, 32),
        gemv_alloc_reducer(N, K, block_M=128, block_N=128),
    ]
    for kernel in kernel_list:
        profiler = kernel.get_profiler()
        # Benchmark the TileLang kernel itself, not the PyTorch reference.
        latency += profiler.do_bench(warmup=50, backend="cupti")
    return latency / len(kernel_list)


if __name__ == "__main__":
    main()

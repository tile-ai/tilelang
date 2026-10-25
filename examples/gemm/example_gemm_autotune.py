import argparse
import itertools
import tilelang as tl
import tilelang.language as T
from tilelang.autotuner import AutoTuner
import torch


def ref_program(A, B):
    """
    Compute the matrix product of A and the transpose of B.

    A and B are expected to be 2-D tensors where A has shape (M, K) and B has shape (N, K). The result is a tensor with shape (M, N) equal to A @ B.T, using the inputs' dtypes.
    """
    return A @ B.T


def get_configs(M, N, K, with_roller=False, topk=20):
    """Build explicit native TileLang candidates without Schedule or TE."""
    if with_roller:
        raise ValueError("Schedule-based Carver is retired; use explicit TileLang autotuner configurations")
    candidates = itertools.product([64, 128, 256], [64, 128, 256], [32, 64], [0, 1, 2, 3], [128, 256], [True, False])
    return [
        {
            "block_M": block_rows,
            "block_N": block_columns,
            "block_K": reduction_block,
            "num_stages": stages,
            "thread_num": threads,
            "enable_rasteration": rasterization,
        }
        for block_rows, block_columns, reduction_block, stages, threads, rasterization in candidates
    ]


def get_best_config(
    M,
    N,
    K,
    with_roller: bool = False,
    profile_backend: str = "event",
):
    def kernel(
        block_M=None,
        block_N=None,
        block_K=None,
        num_stages=None,
        thread_num=None,
        enable_rasteration=None,
    ):
        dtype = T.bfloat16
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
                C_shared = T.alloc_shared((block_M, block_N), dtype)
                T.use_swizzle(panel_size=10, enable=enable_rasteration)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.gemm(
                        A_shared,
                        B_shared,
                        C_local,
                        transpose_B=True,
                    )
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

        return main

    autotuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=get_configs(M, N, K, with_roller))
        .set_compile_args(
            out_idx=[-1],
            target="auto",
        )
        .set_profile_args(
            supply_type=tl.TensorSupplyType.Auto,
            ref_prog=ref_program,
            skip_check=False,
            backend=profile_backend,
        )
    )
    return autotuner.run(warmup=3, rep=20)


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


@tl.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, num_stages, thread_num, enable_rasteration, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm_autotune(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)
            T.use_swizzle(panel_size=10, enable=enable_rasteration)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(
                    A_shared,
                    B_shared,
                    C_local,
                    transpose_B=True,
                )
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return gemm_autotune


def main(
    M: int = 4096,
    N: int = 4096,
    K: int = 4096,
    use_autotune: bool = False,
    with_roller: bool = False,
    profile_backend: str = "event",
):
    if use_autotune:
        result = get_best_config(
            M,
            N,
            K,
            with_roller=with_roller,
            profile_backend=profile_backend,
        )
        print(result.config)
        kernel = result.kernel
    else:
        config = get_heuristic_config()
        kernel = matmul(M, N, K, **config)

    # benchmark
    profiler = kernel.get_profiler(tensor_supply_type=tl.TensorSupplyType.Auto)
    tilelang_latency = profiler.do_bench(
        backend=profile_backend,
    )
    ref_latency = profiler.do_bench(
        ref_program,
        backend=profile_backend,
    )
    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)
    print(f"TileLang latency: {tilelang_latency}")
    print(f"Ref latency: {ref_latency}")
    print(f"TileLang TFlops: {2 * M * N * K / tilelang_latency * 1e-9}")
    print(f"Ref TFlops: {2 * M * N * K / ref_latency * 1e-9}")


def run_regression_perf(M: int = 4096, N: int = 4096, K: int = 4096):
    config = get_heuristic_config()
    kernel = matmul(M, N, K, **config)
    profiler = kernel.get_profiler(tensor_supply_type=tl.TensorSupplyType.Auto)
    return profiler.do_bench(backend="cupti")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autotuned MatMul Benchmark")
    parser.add_argument("--m", type=int, default=4096, help="Matrix dimension M")
    parser.add_argument("--n", type=int, default=4096, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=4096, help="Matrix dimension K")
    parser.add_argument("--use_autotune", action="store_true", default=False, help="Whether to use autotune for matmul configs")
    parser.add_argument(
        "--with_roller", action="store_true", default=False, help="Retired option; use explicit TileLang configurations instead"
    )
    parser.add_argument("--profile_backend", type=str, default="event", help="Profiler backend")
    args = parser.parse_args()
    main(
        args.m,
        args.n,
        args.k,
        args.use_autotune,
        args.with_roller,
        args.profile_backend,
    )

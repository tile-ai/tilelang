"""
Scripts to test a series of advanced autotuning techniques.

To maximze the autotuning speed in a 8xGPUs node, please use the following command:

python3 examples/gemm/example_gemm_advanced_autotune.py --use_autotune --enable_grouped_compile --use_pipeline --benchmark_multi_gpu --benchmark_devices 0 1 2 3 4 5 6 7
"""

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
    execution_backend: str = "auto",
    warmup: int = 3,
    rep: int = 20,
    timeout: int = 30,
    skip_check: bool = False,
    cache_input_tensors: bool = False,
    topk: int = 20,
    use_pipeline: bool = False,
    enable_grouped_compile: bool = False,
    group_compile_size: int = 2,
    benchmark_multi_gpu: bool = False,
    benchmark_devices: list[int] | None = None,
):
    autotuner, _, _ = _build_autotuner(
        M=M,
        N=N,
        K=K,
        with_roller=with_roller,
        profile_backend=profile_backend,
        execution_backend=execution_backend,
        skip_check=skip_check,
        cache_input_tensors=cache_input_tensors,
        topk=topk,
    )
    autotuner_result = autotuner.run(
        warmup=warmup,
        rep=rep,
        timeout=timeout,
        use_pipeline=use_pipeline,
        enable_grouped_compile=enable_grouped_compile,
        group_compile_size=group_compile_size,
        benchmark_multi_gpu=benchmark_multi_gpu,
        benchmark_devices=benchmark_devices,
    )
    return autotuner_result


def _build_autotuner(
    M: int,
    N: int,
    K: int,
    with_roller: bool,
    profile_backend: str,
    execution_backend: str,
    skip_check: bool,
    cache_input_tensors: bool,
    topk: int,
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

    configs = get_configs(M, N, K, with_roller, topk=topk)
    autotuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=configs)
        .set_compile_args(
            out_idx=[-1],
            target="auto",
            execution_backend=execution_backend,
        )
        .set_profile_args(
            supply_type=tl.TensorSupplyType.Auto,
            ref_prog=None if skip_check else ref_program,
            skip_check=skip_check,
            cache_input_tensors=cache_input_tensors,
            backend=profile_backend,
        )
    )
    return autotuner, configs, kernel


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
    use_pipeline: bool = False,
    enable_grouped_compile: bool = False,
    group_compile_size: int = 2,
    benchmark_multi_gpu: bool = False,
    benchmark_devices: list[int] | None = None,
):
    benchmark_devices = benchmark_devices or []

    if use_autotune:
        result = get_best_config(
            M,
            N,
            K,
            with_roller=with_roller,
            profile_backend=profile_backend,
            use_pipeline=use_pipeline,
            enable_grouped_compile=enable_grouped_compile,
            group_compile_size=group_compile_size,
            benchmark_multi_gpu=benchmark_multi_gpu,
            benchmark_devices=benchmark_devices,
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
    parser.add_argument("--use_pipeline", action="store_true", default=False, help="Enable compile/benchmark pipeline in autotune")
    parser.add_argument("--enable_grouped_compile", action="store_true", default=False, help="Enable grouped compilation in autotune")
    parser.add_argument("--group_compile_size", type=int, default=2, help="Number of configs per grouped compile unit")
    parser.add_argument("--benchmark_multi_gpu", action="store_true", default=False, help="Benchmark autotune configs across multiple GPUs")

    parser.add_argument(
        "--benchmark_devices",
        nargs="+",
        type=int,
        default=[],
        help="Benchmark devices number (e.g. --benchmark-devices 0 1 2)",
    )

    args = parser.parse_args()
    main(
        M=args.m,
        N=args.n,
        K=args.k,
        use_autotune=args.use_autotune,
        with_roller=args.with_roller,
        profile_backend=args.profile_backend,
        use_pipeline=args.use_pipeline,
        enable_grouped_compile=args.enable_grouped_compile,
        group_compile_size=args.group_compile_size,
        benchmark_multi_gpu=args.benchmark_multi_gpu,
        benchmark_devices=args.benchmark_devices,
    )

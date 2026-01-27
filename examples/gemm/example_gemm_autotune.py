import argparse
import itertools
import tilelang as tl
import tilelang.language as T
from tilelang.autotuner import AutoTuner
from tilelang.carver.template import MatmulTemplate
from tilelang.carver.arch import CUDA
from tilelang.carver.arch import CDNA
from tilelang.carver.roller.rasterization import NoRasterization
import torch


def ref_program(A, B):
    """
    Compute the matrix product of A and the transpose of B.

    A and B are expected to be 2-D tensors where A has shape (M, K) and B has shape (N, K). The result is a tensor with shape (M, N) equal to A @ B.T, using the inputs' dtypes.
    """
    return A @ B.T


def get_configs(M, N, K, with_roller=False, topk=20):
    """
    Generate a list of kernel tuning configuration dictionaries for a tiled matrix-multiply.

    When with_roller is True this queries the MatmulTemplate roller to produce up to `topk` recommended
    configurations (device-specific TensorCore-friendly tilings). Each returned dict contains:
      - block_M, block_N, block_K: tile sizes
      - num_stages: pipeline staging (0 means no explicit staging)
      - thread_num: total threads used for the block
      - enable_rasteration: whether a rasterization/swizzle layout was recommended (note spelling)

    When with_roller is False this returns the Cartesian product of a fixed set of candidate
    parameters; the returned dicts use the backward-compatible key name "enable_rasteration" for that flag.

    Parameters:
        M, N, K (int): GEMM dimensions used to generate valid tile sizes.
        with_roller (bool): If True, use MatmulTemplate's roller to generate device-aware hints;
            otherwise use a predefined candidate grid.
        topk (int): Maximum number of roller hints to request when with_roller is True.

    Returns:
        List[dict]: A list of configuration dictionaries as described above.

    Raises:
        ValueError: if with_roller is True but the roller returns no hints.
    """
    if with_roller:
        arch = CUDA("cuda") if torch.version.hip is None else CDNA("hip")
        carve_template = MatmulTemplate(
            M=M,
            N=N,
            K=K,
            in_dtype=T.float16,
            out_dtype=T.float16,
            accum_dtype=T.float32,
        ).with_arch(arch)

        func = carve_template.equivalent_function()
        assert func is not None, "Function is None"
        roller_hints = carve_template.recommend_hints(topk=topk)
        if roller_hints is None:
            raise ValueError("No Roller Hints Found for TensorCore Scheduling")
        configs = []
        for hint in roller_hints:
            config = {}
            block_m, block_n = hint.block
            warp_m, warp_n = hint.warp
            # block_rows, block_cols represents warp partitioning
            block_rows, block_cols = block_m // warp_m, block_n // warp_n
            config["block_M"] = block_m
            config["block_N"] = block_n
            config["block_K"] = hint.rstep[0]
            config["num_stages"] = hint.pipeline_stage if hint.pipeline_stage > 1 else 0
            config["thread_num"] = block_rows * block_cols * 32
            config["enable_rasteration"] = hint.rasterization_plan is not NoRasterization
            configs.append(config)
    else:
        block_M = [64, 128, 256]
        block_N = [64, 128, 256]
        block_K = [32, 64]
        num_stages = [0, 1, 2, 3]
        thread_num = [128, 256]
        enable_rasterization = [True, False]
        _configs = list(
            itertools.product(
                block_M,
                block_N,
                block_K,
                num_stages,
                thread_num,
                enable_rasterization,
            )
        )

        configs = [
            {
                "block_M": c[0],
                "block_N": c[1],
                "block_K": c[2],
                "num_stages": c[3],
                "thread_num": c[4],
                "enable_rasteration": c[5],  # keep param name for backward-compat
            }
            for c in _configs
        ]
    return configs


def get_best_config(M, N, K, with_roller=False):
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
            supply_type=tl.TensorSupplyType.Integer,
            ref_prog=ref_program,
            skip_check=False,
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


def main(M: int = 4096, N: int = 4096, K: int = 4096, use_autotune: bool = False, with_roller: bool = False):
    use_autotune = True
    if use_autotune:
        result = get_best_config(M, N, K, with_roller)
        print(result.config)
        kernel = result.kernel
    else:
        config = get_heuristic_config()
        kernel = matmul(M, N, K, **config)

    # benchmark
    profiler = kernel.get_profiler(tensor_supply_type=tl.TensorSupplyType.Auto)
    tilelang_latency = profiler.do_bench()
    ref_latency = profiler.do_bench(ref_program)
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
    parser.add_argument("--with_roller", action="store_true", default=False, help="Whether to enable BitBLAS roller for search space")
    args = parser.parse_args()
    main(args.m, args.n, args.k, args.use_autotune, args.with_roller)

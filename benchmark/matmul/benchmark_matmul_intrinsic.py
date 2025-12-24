import argparse
import logging
from tilelang import tvm as tvm
from tvm import DataType
import tilelang as tl
import tilelang.language as T
from tilelang.intrinsics import get_swizzle_layout
from tilelang.intrinsics.mma_macro_generator import (
    TensorCoreIntrinEmitter,
)
from tilelang.transform import simplify_prim_func
from tilelang.autotuner import autotune
import itertools

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def make_swizzle_layout(shared_buf):
    dtype = shared_buf.dtype
    shape = shared_buf.shape

    can_swizzle = shape[-1] * DataType(dtype).bits == 512
    if not can_swizzle:
        return T.Layout(shape, lambda *args: args)

    def transform_func(i, j):
        new_warp_i, new_warp_j = get_swizzle_layout(i, j, shape[-1], dtype)
        return [new_warp_i, new_warp_j]

    return T.Layout(shape, transform_func)


@simplify_prim_func
def tl_matmul(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    block_row_warps=1,
    block_col_warps=1,
    warp_row_tiles=16,
    warp_col_tiles=16,
    chunk=32,
    stage=2,
    enable_rasteration=False,
):
    assert in_dtype in [
        T.float16,
        T.int8,
    ], "Currently only float16 and int8 are supported"
    assert out_dtype in [
        T.float16,
        T.float32,
        T.int32,
    ], "Currently only float16, float32 and int32 are supported"

    micro_size_x = micro_size_y = micro_size_k = 16

    if out_dtype == T.int32:
        micro_size_k = 32

    # This is a debug config
    # chunk = 32 if in_dtype == T.float16 else 64
    shared_scope = "shared.dyn"

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    A_shape = (M, K)
    B_shape = (N, K)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)
    C_shared_shape = (
        block_M,
        block_N,
    )

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size_a = (micro_size_x * micro_size_k) // warp_size
    local_size_b = (micro_size_y * micro_size_k) // warp_size
    local_size_c = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    # MMA Wrapper to Auto Generate Code for MMA
    mma_emitter = TensorCoreIntrinEmitter(
        a_dtype=in_dtype,
        b_dtype=in_dtype,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
    )

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope=shared_scope)
            C_shared = T.alloc_shared(C_shared_shape, out_dtype, scope=shared_scope)
            A_local = T.alloc_local((warp_rows * local_size_a), in_dtype)
            B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size_c), accum_dtype)

            T.annotate_layout(
                {
                    A_shared: make_swizzle_layout(A_shared),
                    B_shared: make_swizzle_layout(B_shared),
                }
            )

            # Improve L2 Cache
            T.use_swizzle(panel_size=10, enable=enable_rasteration)

            T.clear(C_local)

            for ko in T.Pipelined((K // block_K), num_stages=stage):
                # Load A into shared memory
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]

                # Load B into shared memory
                for j, k in T.Parallel(block_N, block_K):
                    B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

                for ki in T.serial(0, (block_K // micro_size_k)):
                    # Load A into fragment
                    mma_emitter.ldmatrix_a(A_local, A_shared, ki)

                    # Load B into fragment
                    mma_emitter.ldmatrix_b(B_local, B_shared, ki)

                    # Perform Matrix Multiplication
                    mma_emitter.mma(A_local, B_local, C_local)

            # Perform STMatrix
            mma_emitter.stmatrix(C_local, C_shared)

            # Store shared into global
            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = C_shared[i, j]

    return main


def ref_program(A, B):
    """Reference matrix multiplication program."""
    return A @ B.T


def get_configs(args, kwargs):
    """
    Generate a list of configuration dictionaries that will be used for tuning.

    Parameters
    ----------
    with_roller : bool
        Whether to enable bitblas roller to deduce search spaces

    Returns
    -------
    list of dict
        Each configuration dict includes various block sizes, pipeline stages,
        thread numbers, and other parameters to explore during autotuning.
    """
    M, N, K = args[:3]
    with_roller = args[6]

    if with_roller:
        from tilelang.carver.template import MatmulTemplate
        from tilelang.carver.arch import CUDA
        from tilelang.carver.arch import CDNA
        from tilelang.carver.roller.rasterization import NoRasterization
        import torch

        arch = CUDA("cuda") if torch.version.hip is None else CDNA("hip")
        topk = 10

        carve_template = MatmulTemplate(
            M=M,
            N=N,
            K=K,
            in_dtype=T.float16,
            out_dtype=T.float16,
            accum_dtype=T.float16,
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
            config["block_row_warps"] = block_m // warp_m
            config["block_col_warps"] = block_n // warp_n
            config["warp_row_tiles"] = warp_m
            config["warp_col_tiles"] = warp_n
            config["chunk"] = hint.rstep[0]
            config["stage"] = hint.pipeline_stage
            config["enable_rasteration"] = hint.rasterization_plan is not NoRasterization
            configs.append(config)
        for config in configs:
            print(config)
    else:
        iter_params = dict(
            block_row_warps=[1, 2, 4],
            block_col_warps=[1, 2, 4],
            warp_row_tiles=[16, 32, 64, 128],
            warp_col_tiles=[16, 32, 64, 128],
            chunk=[32, 64, 128, 256],
            stage=[0, 2],
            enable_rasteration=[True, False],
        )
        return [{k: v for k, v in zip(iter_params, values)} for values in itertools.product(*iter_params.values())]

    return configs


@autotune(
    configs=get_configs,
    warmup=3,
    rep=5,
    ref_prog=ref_program,
    skip_check=True,
)
@tl.jit(
    out_idx=[2],
)
def matmul(
    M,
    N,
    K,
    in_dtype=T.float16,
    out_dtype=T.float16,
    accum_dtype=T.float16,
    with_roller=False,
    block_row_warps=None,
    block_col_warps=None,
    warp_row_tiles=None,
    warp_col_tiles=None,
    chunk=None,
    stage=None,
    enable_rasteration=None,
):
    """Create an autotuned tensor core matrix multiplication kernel."""

    def kernel():
        return tl_matmul(
            M,
            N,
            K,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            accum_dtype=accum_dtype,
            block_row_warps=block_row_warps,
            block_col_warps=block_col_warps,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=chunk,
            stage=stage,
            enable_rasteration=enable_rasteration,
        )

    return kernel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autotuned TensorCore MatMul Benchmark")
    parser.add_argument("--m", type=int, default=16384, help="Matrix dimension M")
    parser.add_argument("--n", type=int, default=16384, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=16384, help="Matrix dimension K")
    parser.add_argument("--with_roller", type=bool, default=False, help="Whether to use roller to deduce search spaces")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "int8"], help="Input data type")
    args = parser.parse_args()

    M, N, K = args.m, args.n, args.k
    in_dtype = T.dtype(args.dtype)
    out_dtype = T.float32 if in_dtype == T.int8 else T.float16
    accum_dtype = T.float32 if in_dtype == T.int8 else T.float16
    with_roller = args.with_roller
    with_roller = True
    # Compute total floating-point operations
    total_flops = 2 * M * N * K

    # Run autotuning
    best_result = matmul(M, N, K, in_dtype, out_dtype, accum_dtype, with_roller)
    best_latency = best_result.latency
    best_config = best_result.config
    ref_latency = best_result.ref_latency

    # Print benchmark results
    print(f"Best latency (s): {best_latency}")
    print(f"Best TFlops: {total_flops / best_latency * 1e-9:.3f}")
    print(f"Best config: {best_config}")
    print(f"Reference TFlops: {total_flops / ref_latency * 1e-9:.3f}")

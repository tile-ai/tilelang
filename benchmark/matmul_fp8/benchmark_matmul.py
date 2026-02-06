import argparse
import itertools
import torch
import logging
import tilelang.language as T
from tilelang.autotuner import autotune
from tilelang import jit

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def ref_program(A, B):
    """
    A reference matrix multiplication program, used to compare performance.

    Parameters
    ----------
    A : numpy.ndarray
        The matrix with shape (M, K).
    B : numpy.ndarray
        The matrix with shape (N, K).

    Returns
    -------
    np.ndarray
        The result of A @ B.T, shape (M, N).
    """
    return A.float() @ B.T.float()


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
    M, N, K, with_roller = args[:4]

    if with_roller:
        from tilelang.carver.template import MatmulTemplate
        from tilelang.carver.arch import CUDA
        from tilelang.carver.arch import CDNA
        from tilelang.carver.roller.rasterization import NoRasterization
        import torch

        arch = CDNA("hip") if torch.version.hip is not None else CUDA("cuda")

        topk = 10

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
            config["num_stages"] = hint.pipeline_stage
            config["thread_num"] = block_rows * block_cols * 32
            config["policy"] = T.GemmWarpPolicy.from_warp_partition(block_rows, block_cols)
            config["enable_rasteration"] = hint.rasterization_plan is not NoRasterization
            configs.append(config)
        for config in configs:
            print(config)
    else:
        iter_params = dict(
            block_M=[64, 128, 256],
            block_N=[64, 128, 256],
            block_K=[64, 128],
            num_stages=[0, 1, 2, 3],
            thread_num=[128, 256],
            k_pack=[1, 2],
            policy=[T.GemmWarpPolicy.Square],
            enable_rasteration=[True, False],
        )
        return [{k: v for k, v in zip(iter_params, values)} for values in itertools.product(*iter_params.values())]

    return configs


@autotune(
    configs=get_configs,
    warmup=3,
    rep=20,
)
@jit(
    out_idx=[2],
)
def matmul(
    M,
    N,
    K,
    with_roller,
    block_M=None,
    block_N=None,
    block_K=None,
    num_stages=None,
    thread_num=None,
    k_pack=None,
    policy=None,
    enable_rasteration=None,
):
    """
    Create an autotuned matrix multiplication kernel for matrices of shape:
      - A: (M, K)
      - B: (N, K)
      - C: (M, N)

    Parameters
    ----------
    M : int
        The dimension M of the matrix multiplication.
    N : int
        The dimension N of the matrix multiplication.
    K : int
        The dimension K of the matrix multiplication.

    Returns
    -------
    (best_latency, best_config, ref_latency)
        best_latency : float
            The best latency found among the tuned configurations.
        best_config : dict
            The parameter configuration that yielded best_latency.
        ref_latency : float
            The baseline latency of the reference program (for computing speedup).
    """

    # Use half-precision for input data to reduce memory bandwidth,
    # accumulate in float for better numerical accuracy
    dtype = T.float8_e4m3fnuz if torch.version.hip is not None else T.float8_e4m3fn
    accum_dtype = T.float32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        """
        The compiled TVM function for block-level matrix multiplication.

        - We divide the entire (M, N) domain into blocks of shape
            (block_M, block_N).
        - Each block has its own allocated shared memory for sub-blocks
            of A and B.
        - The partial results go into C_local, and then we copy them back
            to global memory C.
        """
        # Bind x-dimension to block index in N,
        #     y-dimension to block index in M.
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            # Allocate shared memory for A sub-block of shape (block_M, block_K)
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            # Allocate shared memory for B sub-block of shape (block_N, block_K)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            # Allocate a local fragment for intermediate accumulation
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            # Allocate a shared memory for C sub-block of shape (block_M, block_N)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            # Enable (or disable) swizzling optimization
            T.use_swizzle(panel_size=10, enable=enable_rasteration)

            # Clear out the accumulation buffer
            T.clear(C_local)

            # Loop over sub-blocks in K dimension, pipelined by num_stages
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                # Load a sub-block of A from global memory into A_shared
                T.copy(A[by * block_M, k * block_K], A_shared)
                # Load a sub-block of B from global memory into B_shared
                T.copy(B[bx * block_N, k * block_K], B_shared)
                # Perform a partial matrix multiplication:
                #   C_local += A_shared @ B_shared^T
                T.gemm(
                    A_shared,
                    B_shared,
                    C_local,
                    transpose_B=True,
                    policy=policy,
                    k_pack=k_pack,
                )
            # Write back the results from C_local to the global memory C
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


if __name__ == "__main__":
    # Parse command-line arguments for matrix dimensions
    parser = argparse.ArgumentParser(description="Autotuned MatMul Benchmark")
    parser.add_argument("--m", type=int, default=16384, help="Matrix dimension M")
    parser.add_argument("--n", type=int, default=16384, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=16384, help="Matrix dimension K")
    parser.add_argument(
        "--with_roller",
        action="store_true",
        help="Whether to enable BitBLAS roller for search space",
    )
    args = parser.parse_args()

    M, N, K = args.m, args.n, args.k
    with_roller = args.with_roller

    # Compute total floating-point operations to measure throughput
    total_flops = 2 * M * N * K

    # matmul(...) returns (best_latency, best_config, ref_latency)
    best_result = matmul(M, N, K, with_roller)
    best_latency = best_result.latency
    best_config = best_result.config

    # Print out the benchmark results
    print(f"Best latency (s): {best_latency}")
    print(f"Best TFlops: {total_flops / best_latency * 1e-9:.3f}")
    print(f"Best config: {best_config}")

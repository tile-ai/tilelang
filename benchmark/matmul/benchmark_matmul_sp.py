import argparse
import itertools
import logging
import torch
from triton.testing import do_bench

import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
from tilelang import jit
from tilelang.contrib import nvcc
from tilelang.layout import make_cutlass_metadata_layout

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

arch = nvcc.get_target_compute_version()

ARCH_INFO = {"8.0": (16, "int16"), "8.9": (16, "int16"), "9.0": (8, "uint8")}


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
    return A @ B.T


def get_configs(M, N, K):
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
    block_M = [64, 128, 256]
    block_N = [64, 128, 256]
    block_K = [64, 128]
    num_stages = [0, 1, 2, 3]
    thread_num = [128, 256]
    enable_rasterization = [True, False]
    policy = [T.GemmWarpPolicy.Square]
    _configs = list(
        itertools.product(
            block_M,
            block_N,
            block_K,
            num_stages,
            thread_num,
            policy,
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
            "policy": c[5],
            "enable_rasterization": c[6],  # keep param name for backward-compat
        }
        for c in _configs
    ]
    return configs


def matmul_sp(M, N, K, in_dtype, accum_dtype):
    """
    Create an autotuned matrix multiplication kernel for matrices of shape:
      - A: (M, K)
      - B: (K, N)
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

    # Decorate the kernel with autotune & jit, specifying:
    #  - Tuning config list
    #  - Profiling keys
    #  - Warmup and repetition counts for better measurement
    #  - A reference program for correctness verification
    #  - The "tvm" profiler backend
    #  - HIP as the compilation target (modify as needed for your hardware)

    @autotune(
        configs=get_configs(M, N, K),
        warmup=3,
        rep=20,
    )
    @jit(
        out_idx=[2],
    )
    def kernel(
        block_M=None,
        block_N=None,
        block_K=None,
        num_stages=None,
        thread_num=None,
        policy=None,
        enable_rasterization=None,
    ):
        """
        The actual kernel to compute C = A @ B^T.

        Parameters
        ----------
        block_M : int
            Block size in M dimension.
        block_N : int
            Block size in N dimension.
        block_K : int
            Block size in K dimension.
        num_stages : int
            Number of pipelined stages (for asynchronous load).
        thread_num : int
            Number of threads to use per block.
        k_pack : int
            K dimension packing factor to improve memory coalescing.

        Returns
        -------
        Function
            A TVM Tensor Language function (T.prim_func) that computes matmul.
        """
        # Use half-precision for input data to reduce memory bandwidth,
        # accumulate in float for better numerical accuracy
        e_factor, e_dtype = ARCH_INFO[arch]

        @T.prim_func
        def main(
            A_sparse: T.Tensor((M, K // 2), in_dtype),
            E: T.Tensor((M, K // e_factor), e_dtype),
            B: T.Tensor((K, N), in_dtype),
            C: T.Tensor((M, N), accum_dtype),
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
                A_shared = T.alloc_shared((block_M, block_K // 2), in_dtype)
                # Allocate shared memory for B sub-block of shape (block_N, block_K)
                B_shared = T.alloc_shared((block_K, block_N), in_dtype)
                # Allocate shared memory for E sub-block of shape (block_M, block_K // E_factor)
                E_shared = T.alloc_shared((block_M, block_K // e_factor), e_dtype)
                # Allocate a local fragment for intermediate accumulation
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                # Allocate a shared memory for C sub-block of shape (block_M, block_N)
                C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

                # Clear out the accumulation buffer
                T.clear(C_local)
                T.disable_warp_group_reg_alloc()

                T.use_swizzle(panel_size=10, enable=enable_rasterization)
                T.annotate_layout(
                    {
                        E: make_cutlass_metadata_layout(E, mma_dtype=in_dtype, block_k=block_K),
                        E_shared: make_cutlass_metadata_layout(E_shared, mma_dtype=in_dtype, block_k=block_K),
                    }
                )
                # Loop over sub-blocks in K dimension, pipelined by num_stages
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    # Load a sub-block of A from global memory into A_shared
                    T.copy(A_sparse[by * block_M, k * block_K // 2], A_shared)
                    # Load a sub-block of E from global memory into E_shared
                    T.copy(E[by * block_M, k * block_K // e_factor], E_shared)
                    # Load a sub-block of B from global memory into B_shared
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    # Perform a partial matrix multiplication:
                    #   C_local += A_shared @ B_shared
                    T.gemm_sp_v2(
                        A_shared,
                        E_shared,
                        B_shared,
                        C_local,
                        transpose_B=False,
                        policy=policy,
                    )
                # Write back the results from C_local to the global memory C
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

        return main

    return kernel()


if __name__ == "__main__":
    # Parse command-line arguments for matrix dimensions
    parser = argparse.ArgumentParser(description="Autotuned MatMul Benchmark")
    parser.add_argument("--m", type=int, default=16384, help="Matrix dimension M")
    parser.add_argument("--n", type=int, default=16384, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=16384, help="Matrix dimension K")
    parser.add_argument("--disable_cache", action="store_true")
    parser.add_argument("--accum_dtype", type=str, default="float", choices=["float", "float16"], help="Accumulation datatype")
    parser.add_argument(
        "--bench_torch_sparse",
        type=str,
        choices=["cutlass", "cusparselt"],
        default=None,
        help="Whether to benchmark against torch sparse implementation, note that at current time only sm80 is supported",
    )
    args = parser.parse_args()

    if args.disable_cache:
        tilelang.disable_cache()

    M, N, K = args.m, args.n, args.k

    # Compute total floating-point operations to measure throughput
    total_flops = 2 * M * N * K

    # matmul(...) returns (best_latency, best_config, ref_latency)
    best_result = matmul_sp(M, N, K, T.float16, args.accum_dtype)
    best_latency = best_result.latency
    best_config = best_result.config
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    ref_latency = do_bench(lambda: A @ B)

    if args.bench_torch_sparse is not None:
        from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

        if args.bench_torch_sparse == "cutlass":
            SparseSemiStructuredTensor._FORCE_CUTLASS = True
        A_sp = to_sparse_semi_structured(A, transposed=False)
        torch_sparse_latency = do_bench(lambda: A_sp @ B)

    # Print out the benchmark results
    print(f"Best latency (s): {best_latency}")
    print(f"Best TFlops: {total_flops / best_latency * 1e-9:.3f}")
    print(f"Best config: {best_config}")

    if args.bench_torch_sparse is not None:
        print(f"Torch sparse ({args.bench_torch_sparse}) TFlops: {total_flops / torch_sparse_latency * 1e-9:.3f}")

    print(f"Reference Dense TFlops: {total_flops / ref_latency * 1e-9:.3f}")

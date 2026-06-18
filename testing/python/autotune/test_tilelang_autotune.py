import itertools
import inspect

import tilelang.testing
import tilelang.language as T
from tilelang.autotuner import AutoTuner
from tilelang.autotuner.tuner import _first_autotune_config, _provided_jit_parameter_names


def ref_program(A, B):
    """
    A reference matrix multiplication program, used to compare performance.

    Parameters
    ----------
    A : torch.Tensor
        The matrix with shape (M, K).
    B : torch.Tensor
        The matrix with shape (N, K).

    Returns
    -------
    torch.Tensor
        The result of A @ B.T, shape (M, N).
    """
    return A @ B.T


def get_configs():
    block_M = [64]
    block_N = [64]
    block_K = [32]
    num_stages = [0, 1]
    thread_num = [128]
    enable_rasterization = [False]

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


def matmul(M, N, K):
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

    # Decorate the kernel with autotune & jit, specifying:
    #  - Tuning config list
    #  - Profiling keys
    #  - Warmup and repetition counts for better measurement
    #  - A reference program for correctness verification
    #  - The "tvm" profiler backend
    #  - HIP as the compilation target (modify as needed for your hardware)

    def kernel(
        block_M=None,
        block_N=None,
        block_K=None,
        num_stages=None,
        thread_num=None,
        enable_rasteration=None,
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
        enable_rasteration : bool
            Whether to enable rasterization (swizzling) optimization.
        k_pack : int
            K dimension packing factor to improve memory coalescing.

        Returns
        -------
        Function
            A TVM Tensor Language function (T.prim_func) that computes matmul.
        """
        # Use half-precision for input data to reduce memory bandwidth,
        # accumulate in float for better numerical accuracy
        dtype = T.float16
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

                # Enable (or disable) swizzling optimization
                T.use_swizzle(panel_size=10, enable=enable_rasteration)

                # Clear out the accumulation buffer
                T.clear(C_local)

                # Loop over sub-blocks in K dimension, pipelined by num_stages
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    # Load a sub-block of A from global memory into A_shared
                    T.copy(
                        A[by * block_M, k * block_K],
                        A_shared,
                    )
                    # Load a sub-block of B from global memory into B_shared
                    T.copy(
                        B[bx * block_N, k * block_K],
                        B_shared,
                    )
                    # Perform a partial matrix multiplication:
                    #   C_local += A_shared @ B_shared^T
                    T.gemm(
                        A_shared,
                        B_shared,
                        C_local,
                        transpose_B=True,
                    )
                # Write back the results from C_local to the global memory C
                T.copy(C_local, C[by * block_M, bx * block_N])

        return main

    autotuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=get_configs())
        .set_compile_args(
            out_idx=[-1],
        )
        .set_profile_args(
            ref_prog=ref_program,
        )
    )
    return autotuner.run(warmup=3, rep=20)


@tilelang.testing.requires_cuda
def test_autotune_get_configs():
    get_configs()


def test_autotune_validation_config_skips_positional_args():
    def kernel(
        B,
        S,
        H,
        DK,
        DV,
        input_dtype,
        output_dtype,
        accum_dtype,
        gate_dtype,
        state_dtype,
        chunk_size,
        use_g,
        use_initial_state,
        store_final_state,
        save_new_value,
        block_DK,
        block_DV,
        threads,
        num_stages,
    ):
        return None

    signature = inspect.signature(kernel)
    args = (
        1,
        1024,
        32,
        128,
        128,
        "bfloat16",
        "bfloat16",
        "float32",
        "float32",
        "bfloat16",
        64,
        True,
        True,
        True,
        False,
        32,
        32,
        128,
        1,
    )

    provided = _provided_jit_parameter_names(signature, args, {})
    first_config = _first_autotune_config([{"block_DK": 64, "block_DV": 64, "threads": 256, "num_stages": 2}])
    validation_kwargs = {}
    for name in signature.parameters:
        if name in first_config and name not in provided:
            validation_kwargs[name] = first_config[name]

    assert validation_kwargs == {}


def test_first_autotune_config_accepts_mapping_and_sequence_shapes():
    assert _first_autotune_config({"threads": 128}) == {"threads": 128}
    assert _first_autotune_config([{"threads": 128}]) == {"threads": 128}
    assert _first_autotune_config(({"threads": 128},)) == {"threads": 128}
    assert _first_autotune_config(lambda: [{"threads": 128}]) is None
    assert _first_autotune_config([]) is None


@tilelang.testing.requires_cuda
def test_autotune_matmul():
    matmul(1024, 1024, 1024)


if __name__ == "__main__":
    tilelang.testing.main()

import itertools
import logging
import tilelang
import tilelang.testing
from tilelang.autotuner import set_autotune_inputs
import tilelang.language as T

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
    return A @ B.T


def get_configs():
    iter_params = dict(block_M=[64], block_N=[64], block_K=[32], num_stages=[0, 1], thread_num=[128], enable_rasterization=[False])
    return [{k: v for k, v in zip(iter_params, values)} for values in itertools.product(*iter_params.values())]


@tilelang.autotune(
    configs=get_configs(),
)
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M=128, block_N=128, block_K=32, num_stages=0, thread_num=128, enable_rasterization=False):
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
            T.use_swizzle(panel_size=10, enable=enable_rasterization)

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


def run_autotune(M, N, K, M_value=None, N_value=None, K_value=None):
    import torch

    def _resolve(dim, provided, name):
        if isinstance(dim, T.Var):
            if provided is None:
                raise ValueError(f"Dynamic dimension {name} requires a concrete value.")
            return provided
        return dim

    actual_M = _resolve(M, M_value, "M")
    actual_N = _resolve(N, N_value, "N")
    actual_K = _resolve(K, K_value, "K")

    a = torch.randn(actual_M, actual_K, dtype=torch.float16).cuda()
    b = torch.randn(actual_N, actual_K, dtype=torch.float16).cuda()

    with set_autotune_inputs([a, b]):
        kernel = matmul(M, N, K)

    c = kernel(a, b)

    ref_c = ref_program(a, b)
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


def test_autotune_matmul():
    """
    Run the autotuning validation for the matmul kernel on a 1024x1024x1024 problem.

    This test constructs random CUDA tensors, autotunes the JIT-compiled block-level matrix-multiplication kernel,
    executes it, and asserts the result matches a reference CPU implementation within tolerances.
    """
    run_autotune(1024, 1024, 1024)


def test_autotune_matmul_symbolic_m():
    run_autotune(T.symbolic("m"), 1024, 1024, M_value=1024)


if __name__ == "__main__":
    tilelang.testing.main()

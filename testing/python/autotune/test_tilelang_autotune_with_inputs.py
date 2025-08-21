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
    """
    Generate the Cartesian product of autotuning configuration parameters.
    
    Returns a list of dictionaries, each representing one autotuning configuration mapping
    the parameter names to a chosen value. The explored parameters and their candidate
    values are:
    - block_M: [64]
    - block_N: [64]
    - block_K: [32]
    - num_stages: [0, 1]
    - thread_num: [128]
    - enable_rasterization: [False]
    
    Each dictionary is produced by iterating over the Cartesian product of these value lists.
    """
    iter_params = dict(
        block_M=[64],
        block_N=[64],
        block_K=[32],
        num_stages=[0, 1],
        thread_num=[128],
        enable_rasterization=[False])
    return [{
        k: v for k, v in zip(iter_params, values)
    } for values in itertools.product(*iter_params.values())]


@tilelang.autotune(configs=get_configs(),)
@tilelang.jit(out_idx=[-1])
def matmul(M,
           N,
           K,
           block_M=128,
           block_N=128,
           block_K=32,
           num_stages=0,
           thread_num=128,
           enable_rasterization=False):

    """
           Create a tiled, autotune-ready TVM kernel (prim_func) that computes block-wise matrix multiplication C = A @ B.T.
           
           The returned prim_func expects:
           - A with shape (M, K) and dtype float16
           - B with shape (N, K) and dtype float16
           - C with shape (M, N) and dtype float16
           
           Parameters that affect tiling and execution:
           - block_M, block_N, block_K: sizes of the M, N, and K sub-blocks used for shared-memory tiling.
           - num_stages: number of pipeline stages used when iterating over K sub-blocks (0 = no pipelining).
           - thread_num: number of threads provided to the kernel launch.
           - enable_rasterization: when True, enables swizzling (panel-based memory layout) to improve memory access patterns.
           
           Implementation details (high level):
           - Uses shared memory buffers for A and B sub-blocks, a local accumulator for partial results, and writes back the accumulated block to global C.
           - Accumulation uses float (accum_dtype) while inputs use float16.
           
           Returns:
               A TVM prim_func implementing the described block-level matmul ready for JIT/autotuning.
           """
           dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        """
            Block-level tiled matrix multiplication kernel produced for JIT compilation.
            
            This function computes a tile of the MxN output C by:
            - Tiling the (M, N) domain into blocks of shape (block_M, block_N) and binding
              the kernel grid to (ceildiv(N, block_N), ceildiv(M, block_M)).
            - Allocating shared buffers A_shared (block_M x block_K) and B_shared
              (block_N x block_K) and a local accumulator C_local (block_M x block_N).
            - Optionally enabling swizzling via T.use_swizzle(panel_size=10, enable=enable_rasterization).
            - Iterating over K in sub-blocks (ceildiv(K, block_K)) with pipelining controlled
              by num_stages:
              - Loading sub-blocks of A and B into shared buffers.
              - Performing partial multiply–accumulate into C_local using T.gemm with B transposed.
            - Writing the accumulated C_local back into the global output C at
              (by * block_M, bx * block_N).
            
            Inputs A, B, C are the tiled global tensors corresponding to shapes (M, K), (N, K),
            and (M, N) respectively; the kernel writes the computed tile into C.
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


def run_autotune(M: int, N: int, K: int):
    import torch
    a = torch.randn(M, K, dtype=torch.float16).cuda()
    b = torch.randn(N, K, dtype=torch.float16).cuda()

    with set_autotune_inputs([a, b]):
        kernel = matmul(M, N, K)

    c = kernel(a, b)

    ref_c = ref_program(a, b)
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


def test_autotune_matmul():
    run_autotune(8192, 8192, 8192)


if __name__ == "__main__":
    tilelang.testing.main()

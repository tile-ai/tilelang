import tilelang
import tilelang.language as T
import torch
import tilelang.testing


# add decorator @tilelang.jit if you want to return a torch function
# @tilelang.jit
def tilelang_copy(M, N, block_M, block_N, dtype="float16"):

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                B[by * block_M + i, bx * block_N + j] = A[by * block_M + i, bx * block_N + j]

    return main


def run_tilelang_copy(M=1024, N=1024, block_M=128, block_N=128, dtype="float16"):
    """
    Builds, compiles, and runs the tiled copy kernel produced by tilelang_copy on random CUDA data and verifies the output matches the input.
    
    This helper:
    - Constructs the TileLang program for the given matrix shape and tile sizes.
    - Compiles it for CUDA with warp specialization and TMA lowering disabled.
    - Allocates a random input tensor `a` on CUDA with the specified dtype, runs the kernel to produce `b`, and asserts `b` is close to `a`.
    
    Parameters:
        M, N (int): Dimensions of the 2D tensor to copy.
        block_M, block_N (int): Tile sizes used by the kernel.
        dtype (str): Name of a torch dtype (e.g., "float16", "float") used to construct the input tensor.
    """
    program = tilelang_copy(M, N, block_M, block_N, dtype)
    kernel = tilelang.compile(
        program,
        out_idx=[1],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True
        })
    a = torch.randn(M, N, device="cuda", dtype=getattr(torch, dtype))
    b = kernel(a)
    torch.testing.assert_close(b, a, rtol=1e-2, atol=1e-2)


def test_tilelang_copy():
    run_tilelang_copy(M=1024, N=1024, block_M=128, block_N=128)
    run_tilelang_copy(M=1024, N=576, block_M=32, block_N=576)
    run_tilelang_copy(M=1024, N=576, block_M=32, block_N=576, dtype="float")


def tilelang_copy_with_stride(M, N, NN, block_M, block_N, dtype="float16"):

    """
    Create a TileLang primitive that copies a 2D strided source tensor into a dense destination using a tiled 2D parallel kernel.
    
    The returned primitive `main` expects:
    - A: a T.StridedTensor of logical shape (M, N) with memory strides (NN, 1) — i.e., the physical leading dimension is NN (typically NN >= N).
    - B: a dense T.Tensor of shape (M, N).
    
    The kernel launches a grid sized ceil(N / block_N) x ceil(M / block_M) with 128 threads and, in each tile, copies elements
    B[by*block_M + i, bx*block_N + j] = A[by*block_M + i, bx*block_N + j] for i in [0, block_M) and j in [0, block_N).
    
    Parameters:
    - M, N (int or symbolic): logical tensor dimensions.
    - NN (int or symbolic): physical leading dimension / stride for the first axis of A (used in A's strides).
    - block_M, block_N (int): tile sizes in the two dimensions.
    - dtype (str): element dtype for A and B (default "float16").
    
    Returns:
    - A TileLang primitive function `main` implementing the tiled copy.
    """
    @T.prim_func
    def main(
            A: T.StridedTensor((M, N), (NN, 1), dtype),
            B: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                B[by * block_M + i, bx * block_N + j] = A[by * block_M + i, bx * block_N + j]

    return main


def run_tilelang_copy_with_stride(M=1024,
                                  N=1024,
                                  NN=2048,
                                  block_M=128,
                                  block_N=128,
                                  dtype="float16"):
    """
                                  Builds, compiles, and runs a tiled CUDA kernel that copies from a strided input tensor to a dense output and verifies correctness.
                                  
                                  This helper constructs a TileLang program via tilelang_copy_with_stride(M, N, NN, block_M, block_N, dtype),
                                  compiles it for CUDA with warp-specialization and TMA lowering disabled, runs the kernel on randomly
                                  generated input, and asserts the kernel output equals the input slice a[:, :N] within tolerances.
                                  
                                  Parameters:
                                      M (int): Number of rows.
                                      N (int): Number of columns to copy (width of the dense view).
                                      NN (int or tilelang.language.T.Var): Leading dimension (stride) of the source buffer. If an int,
                                          must be greater than N; if symbolic (T.Var), the function will allocate a buffer with
                                          leading dimension N*2 for testing.
                                      block_M (int): Tile size in the M dimension.
                                      block_N (int): Tile size in the N dimension.
                                      dtype (str): Torch dtype name used for input/output (e.g., "float16" or "float").
                                      
                                  Raises:
                                      AssertionError: If NN is an int and NN <= N.
                                  """
                                  if isinstance(NN, int):
        assert NN > N, "NN must be greater than N"
    program = tilelang_copy_with_stride(M, N, NN, block_M, block_N, dtype)
    kernel = tilelang.compile(
        program,
        out_idx=[1],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        })
    if isinstance(NN, T.Var):
        NN = N * 2
    a = torch.randn(M, NN, device="cuda", dtype=getattr(torch, dtype))
    b = kernel(a[:, :N])
    torch.testing.assert_close(b, a[:, :N], rtol=1e-2, atol=1e-2)


def test_tilelang_copy_with_stride():
    """
    Run stride-copy tests for TileLang.
    
    Executes run_tilelang_copy_with_stride twice: once with a concrete NN (2048) and once with a symbolic NN (T.symbolic("NN")) to validate copying from a StridedTensor (stride NN) into a dense tensor for both concrete and symbolic stride sizes.
    """
    run_tilelang_copy_with_stride(M=1024, N=1024, NN=2048, block_M=128, block_N=128)
    run_tilelang_copy_with_stride(M=1024, N=1024, NN=T.symbolic("NN"), block_M=128, block_N=128)


if __name__ == "__main__":
    tilelang.testing.main()

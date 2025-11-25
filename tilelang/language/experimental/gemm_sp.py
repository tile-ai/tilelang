"""The language interface for tl programs."""
from __future__ import annotations
from tilelang.primitives.gemm.base import GemmWarpPolicy
import tilelang.language as T
from tvm import tir
from tilelang.language.utils import buffer_to_tile_region


def gemm_sp(
    A_sparse: tir.Buffer | tir.Var,
    E: tir.Buffer | tir.Var,
    B: tir.Buffer | tir.Var,
    C: tir.Buffer | tir.Var,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
):
    """Perform a Sparse General Matrix Multiplication (GEMM-sp) operation.

    This function computes C = A @ B where A and B can optionally be transposed.
    The operation supports various warp policies and accumulation modes.

    Args:
        A_sparse (Union[tir.Buffer, tir.Var]): First input matrix dense values
        E (Union[tir.Buffer, tir.Var]): First input matrix sparse metadata
        B (Union[tir.Buffer, tir.Var]): Second input matrix
        C (Union[tir.Buffer, tir.Var]): Output matrix for results
        transpose_A (bool, optional): Whether to transpose matrix A. Defaults to False.
        transpose_B (bool, optional): Whether to transpose matrix B. Defaults to False.
        policy (GemmWarpPolicy, optional): Warp execution policy. Defaults to GemmWarpPolicy.Square.
        clear_accum (bool, optional): Whether to clear accumulator before computation. Defaults to False.
        k_pack (int, optional): Number of k dimensions packed into a single warp. Defaults to 1.
        wg_wait (int, optional): Warp group wait count. Defaults to 0.

    Returns:
        tir.Call: A handle to the GEMM operation

    Raises:
        AssertionError: If the K dimensions of matrices A and B don't match
    """

    def legalize_arguments(arg: tir.Buffer | tir.Var):
        """Convert let-bound variables to their corresponding buffers.

        Args:
            arg (Union[tir.Buffer, tir.Var]): Input argument to legalize

        Returns:
            Union[tir.Buffer, tir.Var]: The legalized argument
        """
        if isinstance(arg, tir.Var) and T.has_let_value(arg):
            return T.get_let_value(arg).buffer
        return arg

    A_sparse = legalize_arguments(A_sparse)
    B = legalize_arguments(B)
    C = legalize_arguments(C)
    M = C.shape[0]
    N = C.shape[1]
    K_A = A_sparse.shape[0] if transpose_A else A_sparse.shape[1]
    K_B = B.shape[1] if transpose_B else B.shape[0]
    assert K_A * 2 == K_B, f"T.gemm_sp K shape check failed: K_A = {K_A}, K_B = {K_B}"
    # Build tl.region descriptors for operands
    A_arg = buffer_to_tile_region(A_sparse, "r")
    E_arg = buffer_to_tile_region(E, "r")
    B_arg = buffer_to_tile_region(B, "r")
    C_arg = buffer_to_tile_region(C, "rw")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.gemm_sp"),
        A_arg,
        E_arg,
        B_arg,
        C_arg,
        transpose_A,
        transpose_B,
        M,
        N,
        K_B,
        policy,
        clear_accum,
        k_pack,
        wg_wait,
    )

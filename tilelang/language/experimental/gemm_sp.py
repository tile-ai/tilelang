"""The language interface for tl programs."""

from __future__ import annotations
from tilelang.tileop.base import GemmWarpPolicy
import tilelang.language as T
from tvm import tir
from tilelang.utils.language import (
    to_buffer_region,
    retrieve_shape,
    retrieve_stride,
    retrieve_offset,
    prim_expr_equal,
)
from tilelang.language.utils import (
    buffer_region_to_tile_region,
)


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
    A_arg = to_buffer_region(A_sparse, access_type="r")
    E_arg = to_buffer_region(E, access_type="r")
    B_arg = to_buffer_region(B, access_type="r")
    C_arg = to_buffer_region(C, access_type="rw")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.gemm_sp"),
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


# experimental currently, for fast compilation
def gemm_sp_v2(
    A_sparse: tir.Buffer | tir.Var,
    E: tir.Buffer | tir.Var,
    B: tir.Buffer | tir.Var,
    C: tir.Buffer | tir.Var,
    transpose_A: bool = False,
    transpose_B: bool = False,
    transpose_E: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
):
    """Perform a General Matrix Multiplication (GEMM) operation.

    This function computes C = A @ B where A and B can optionally be transposed.
    The operation supports various warp policies and accumulation modes.

    Args:
        A_sparse (Union[tir.Buffer, tir.Var]): First input matrix, contains only non-zero elements
        E (Union[tir.Buffer, tir.Var]): The metadata of A_sparse, noted as E
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
    E = legalize_arguments(E)
    B = legalize_arguments(B)
    C = legalize_arguments(C)

    A_region = to_buffer_region(A_sparse)
    E_region = to_buffer_region(E)
    B_region = to_buffer_region(B)
    C_region = to_buffer_region(C)

    A_shape = retrieve_shape(A_sparse)
    E_shape = retrieve_shape(E)  # nolint: F841
    B_shape = retrieve_shape(B)
    C_shape = retrieve_shape(C)

    A_stride = retrieve_stride(A_sparse)
    B_stride = retrieve_stride(B)

    assert len(C_shape) == 2, "current only support C as a 2D tensor"
    assert len(A_shape) >= 2, "current only support A as a 2D or higher-order tensor"
    assert len(B_shape) >= 2, "current only support B as a 2D or higher-order tensor"
    if len(A_shape) > 2:
        for i in range(len(A_shape) - 2):
            assert A_shape[i] == 1, (
                "current only support A as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"
            )
    if len(B_shape) > 2:
        for i in range(len(B_shape) - 2):
            assert B_shape[i] == 1, (
                "current only support B as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"
            )

    M, N = C_shape
    K = 2 * (A_shape[-2] if transpose_A else A_shape[-1])
    K_B = B_shape[-1] if transpose_B else B_shape[-2]
    assert prim_expr_equal(K, K_B), f"T.gemm_sp K shape check failed: K_A (wo sparse) = {K}, K_B = {K_B}"

    stride_a = A_stride[-2]
    stride_b = B_stride[-2]

    A_offset = retrieve_offset(A_sparse)
    B_offset = retrieve_offset(B)
    assert A_offset[-2] == 0, "The offset of the first dimension of A must be 0"
    assert B_offset[-2] == 0, "The offset of the first dimension of B must be 0"
    offset_a = A_offset[-1]
    offset_b = B_offset[-1]

    A_arg = buffer_region_to_tile_region(A_region, "r", [r for r in A_shape])
    E_arg = buffer_region_to_tile_region(E_region, "r", [r for r in E_shape])
    B_arg = buffer_region_to_tile_region(B_region, "r", [r for r in B_shape])
    C_arg = buffer_region_to_tile_region(C_region, "rw", [r for r in C_shape])
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.gemm_sp_py"),
        A_arg,
        E_arg,
        B_arg,
        C_arg,
        transpose_A,
        transpose_B,
        transpose_E,
        M,
        N,
        K,
        policy,
        clear_accum,
        stride_a,
        stride_b,
        offset_a,
        offset_b,
        k_pack,
        wg_wait,
    )

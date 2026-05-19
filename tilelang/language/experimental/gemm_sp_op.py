"""Sparse GEMM operators exposed on the TileLang language surface."""

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
from tilelang._typing import BufferLikeType


def _gemm_sp_impl(
    op_key: str,
    A_sparse: BufferLikeType | tir.Var,
    E: BufferLikeType | tir.Var,
    B: BufferLikeType | tir.Var,
    C: BufferLikeType | tir.Var,
    transpose_A: bool = False,
    transpose_E: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
) -> tir.Call:
    """Shared sparse GEMM implementation.

    Returns a call_intrin handle for the given op key.
    """

    def legalize_arguments(arg: BufferLikeType | tir.Var) -> BufferLikeType:
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
    E_shape = retrieve_shape(E)
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
        tir.op.Op.get(op_key),
        A_arg,
        E_arg,
        B_arg,
        C_arg,
        transpose_A,
        transpose_E,
        transpose_B,
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


def gemm_sp(
    A_sparse: BufferLikeType | tir.Var,
    E: BufferLikeType | tir.Var,
    B: BufferLikeType | tir.Var,
    C: BufferLikeType | tir.Var,
    transpose_A: bool = False,
    transpose_E: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
) -> tir.Call:
    """TileLang sparse GEMM operator.

    This is the default synchronous sparse GEMM interface. On Hopper, if the
    compiler selects WGMMA SP lowering, TileLang inserts the corresponding wait
    implicitly.

    For manual asynchronous scheduling, use ``T.wgmma_gemm_sp(...)`` with
    ``T.wait_wgmma(...)`` on Hopper, or ``T.tcgen05_gemm_sp(...)`` on Blackwell.

    Args:
        A_sparse: Compressed sparse matrix containing only non-zero elements.
        E: Metadata tensor encoding the sparsity pattern of A.
        B: Dense input matrix.
        C: Output accumulator matrix.
        transpose_A: Whether to transpose A. Defaults to False.
        transpose_E: Whether to transpose E. Defaults to False.
        transpose_B: Whether to transpose B. Defaults to False.
        policy: Warp partition policy. Defaults to GemmSPWarpPolicy.Square.
        clear_accum: Whether to zero the accumulator before computation. Defaults to False.
        k_pack: Number of K dimensions packed per warp. Defaults to 1.
        wg_wait: Warp group wait count. Defaults to 0.

    Returns:
        tir.Call: A handle to the sparse GEMM operation.
    """
    return _gemm_sp_impl(
        "tl.tileop.gemm_sp",
        A_sparse,
        E,
        B,
        C,
        transpose_A,
        transpose_E,
        transpose_B,
        policy,
        clear_accum,
        k_pack,
        wg_wait,
    )


def wgmma_gemm_sp(
    A_sparse: BufferLikeType | tir.Var,
    E: BufferLikeType | tir.Var,
    B: BufferLikeType | tir.Var,
    C: BufferLikeType | tir.Var,
    transpose_A: bool = False,
    transpose_E: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
) -> tir.Call:
    """Explicit Hopper WGMMA sparse GEMM without an implicit wait.

    This is the explicit asynchronous Hopper WGMMA counterpart to the default
    synchronous ``T.gemm_sp(...)`` interface, with two stricter guarantees:
    - it always requests the WGMMA SP lowering path
    - it never auto-emits an inlined ``warpgroup_wait``

    If the current target or operand pattern cannot use Hopper WGMMA SP,
    compilation fails instead of silently falling back to MMA SP.

    Args:
        A_sparse: Compressed sparse matrix containing only non-zero elements.
        E: Metadata tensor encoding the sparsity pattern of A.
        B: Dense input matrix.
        C: Output accumulator matrix.
        transpose_A: Whether to transpose A. Defaults to False.
        transpose_E: Whether to transpose E. Defaults to False.
        transpose_B: Whether to transpose B. Defaults to False.
        policy: Warp partition policy. Defaults to GemmSPWarpPolicy.Square.
        clear_accum: Whether to zero the accumulator before computation. Defaults to False.

    Returns:
        tir.Call: A handle to the sparse GEMM operation.
    """
    return _gemm_sp_impl(
        "tl.tileop.wgmma_gemm_sp",
        A_sparse,
        E,
        B,
        C,
        transpose_A,
        transpose_E,
        transpose_B,
        policy,
        clear_accum,
        1,
        -1,
    )


def tcgen05_gemm_sp(
    A_sparse: BufferLikeType | tir.Var,
    E: BufferLikeType | tir.Var,
    B: BufferLikeType | tir.Var,
    C: BufferLikeType | tir.Var,
    transpose_A: bool = False,
    transpose_E: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
) -> tir.Call:
    """Explicit Blackwell TCGEN05 sparse GEMM without an implicit wait.

    This is the explicit asynchronous Blackwell TCGEN05 counterpart to the
    default synchronous ``T.gemm_sp(...)`` interface, with two stricter
    guarantees:
    - it always requests the TCGEN05 SP lowering path
    - it never auto-emits an inlined ``mbarrier_wait_parity``

    If the current target or operand pattern cannot use Blackwell TCGEN05 SP,
    compilation fails instead of silently falling back to another sparse GEMM
    path.

    Args:
        A_sparse: Compressed sparse matrix containing only non-zero elements.
        E: Metadata tensor encoding the sparsity pattern of A.
        B: Dense input matrix.
        C: Output accumulator matrix.
        transpose_A: Whether to transpose A. Defaults to False.
        transpose_E: Whether to transpose E. Defaults to False.
        transpose_B: Whether to transpose B. Defaults to False.
        policy: Warp partition policy. Defaults to GemmSPWarpPolicy.Square.
        clear_accum: Whether to zero the accumulator before computation. Defaults to False.

    Returns:
        tir.Call: A handle to the sparse GEMM operation.
    """
    return _gemm_sp_impl(
        "tl.tileop.tcgen05_gemm_sp",
        A_sparse,
        E,
        B,
        C,
        transpose_A,
        transpose_E,
        transpose_B,
        policy,
        clear_accum,
        1,
        0,
    )

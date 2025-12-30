"""GEMM (General Matrix Multiplication) operators exposed on the TileLang language surface."""

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
from tilelang.env import env as _env


def _gemm_impl(
    op_key: str,
    A: tir.Buffer | tir.Var,
    B: tir.Buffer | tir.Var,
    C: tir.Buffer | tir.Var,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
    mbar: tir.Buffer | None = None,
):
    """Shared GEMM implementation.

    Returns a call_intrin handle for the given op key.
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

    A = legalize_arguments(A)
    B = legalize_arguments(B)
    C = legalize_arguments(C)
    mbar = legalize_arguments(mbar) if mbar is not None else None

    # Normalize A/B/C to BufferRegion for shape/stride/offset analysis
    A_region = to_buffer_region(A)
    B_region = to_buffer_region(B)
    C_region = to_buffer_region(C)

    A_shape = retrieve_shape(A_region)
    B_shape = retrieve_shape(B_region)
    C_shape = retrieve_shape(C_region)

    A_stride = retrieve_stride(A_region)
    B_stride = retrieve_stride(B_region)

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
    K = A_shape[-2] if transpose_A else A_shape[-1]
    K_B = B_shape[-1] if transpose_B else B_shape[-2]
    assert prim_expr_equal(K, K_B), f"T.gemm K shape check failed: K_A = {K}, K_B = {K_B}"

    stride_a = A_stride[-2]
    stride_b = B_stride[-2]

    A_offset = retrieve_offset(A_region)
    B_offset = retrieve_offset(B_region)
    assert A_offset[-2] == 0, "The offset of the first dimension of A must be 0"
    assert B_offset[-2] == 0, "The offset of the first dimension of B must be 0"
    offset_a = A_offset[-1]
    offset_b = B_offset[-1]

    mbar = to_buffer_region(mbar, access_type="rw") if mbar is not None else tir.const(0, T.uint32)
    C_coords = [r.min for r in C_region.region]
    # Convert BufferRegion to tl.region calls for arguments
    A_arg = buffer_region_to_tile_region(A_region, "r", [r for r in A_shape])
    B_arg = buffer_region_to_tile_region(B_region, "r", [r for r in B_shape])
    C_arg = buffer_region_to_tile_region(C_region, "rw", [r for r in C_shape])
    return tir.call_intrin(
        "handle",
        tir.op.Op.get(op_key),
        A_arg,
        B_arg,
        C_arg,
        transpose_A,
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
        mbar,
        C_coords[0],
        C_coords[1],
    )


# Public wrappers
def gemm_v1(
    A: tir.Buffer | tir.Var,
    B: tir.Buffer | tir.Var,
    C: tir.Buffer | tir.Var,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
    mbar: tir.Buffer | None = None,
):
    """GEMM v1: use op tl.gemm."""
    return _gemm_impl(
        "tl.tileop.gemm",
        A,
        B,
        C,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        k_pack,
        wg_wait,
        mbar,
    )


# experimental currently, for fast compilation
def gemm_v2(
    A: tir.Buffer | tir.Var,
    B: tir.Buffer | tir.Var,
    C: tir.Buffer | tir.Var,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
    mbar: tir.Buffer | None = None,
):
    """GEMM v2: use op tl.gemm_py."""
    return _gemm_impl(
        "tl.tileop.gemm_py",
        A,
        B,
        C,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        k_pack,
        wg_wait,
        mbar,
    )


# Default to v2; allow forcing v1 via environment variable
# gemm = gemm_v1 if _env.use_gemm_v1() else gemm_v2


def gemm(
    A: tir.Buffer | tir.Var,
    B: tir.Buffer | tir.Var,
    C: tir.Buffer | tir.Var,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
    mbar: tir.Buffer | None = None,
):
    """TileLang GEMM operator.

    Args:
        A (tir.Buffer | tir.Var): Input buffer A.
        B (tir.Buffer | tir.Var): Input buffer B.
        C (tir.Buffer | tir.Var): Output buffer C.
        transpose_A (bool): Whether to transpose A. Defaults to False.
        transpose_B (bool): Whether to transpose B. Defaults to False.
        policy (GemmWarpPolicy): GEMM warp partition policy.
        clear_accum (bool): Whether to clear the accumulator.
        k_pack (int): Numbers of packed matrix cores, for ROCm only. Defaults to 1.
        wg_wait (int): Int identifier of the warpgroup MMA batch to wait on.. Defaults to 0.
        mbar (tir.Buffer | None, optional): Mbarrier in Blackwell. Defaults to None.

    Returns:
        tir.Call: A handle to the GEMM operation.
    """

    impl = gemm_v1 if _env.use_gemm_v1() else gemm_v2
    return impl(A, B, C, transpose_A, transpose_B, policy, clear_accum, k_pack, wg_wait, mbar)

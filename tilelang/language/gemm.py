"""The language interface for tl programs."""
from __future__ import annotations
from tilelang.primitives.gemm.base import GemmWarpPolicy
import tilelang.language as T
from tvm import tir
from tilelang.utils.language import (
    to_buffer_region,
    retrieve_shape,
    retrieve_stride,
    retrieve_ptr,
    retrieve_offset,
    prim_expr_equal,
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

    # Normalize A/B/C to BufferRegion to pass into tl.gemm
    A = to_buffer_region(A)
    B = to_buffer_region(B)
    C = to_buffer_region(C)

    A_shape = retrieve_shape(A)
    B_shape = retrieve_shape(B)
    C_shape = retrieve_shape(C)

    A_stride = retrieve_stride(A)
    B_stride = retrieve_stride(B)

    assert len(C_shape) == 2, "current only support C as a 2D tensor"
    assert len(A_shape) >= 2, "current only support A as a 2D or higher-order tensor"
    assert len(B_shape) >= 2, "current only support B as a 2D or higher-order tensor"
    if len(A_shape) > 2:
        for i in range(len(A_shape) - 2):
            assert A_shape[i] == 1, \
                "current only support A as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"
    if len(B_shape) > 2:
        for i in range(len(B_shape) - 2):
            assert B_shape[i] == 1, \
                "current only support B as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"

    M, N = C_shape
    K = A_shape[-2] if transpose_A else A_shape[-1]
    K_B = B_shape[-1] if transpose_B else B_shape[-2]
    assert prim_expr_equal(K, K_B), f"T.gemm K shape check failed: K_A = {K}, K_B = {K_B}"

    stride_a = A_stride[-2]
    stride_b = B_stride[-2]

    A_offset = retrieve_offset(A)
    B_offset = retrieve_offset(B)
    assert A_offset[-2] == 0, "The offset of the first dimension of A must be 0"
    assert B_offset[-2] == 0, "The offset of the first dimension of B must be 0"
    offset_a = A_offset[-1]
    offset_b = B_offset[-1]

    mbarptr = retrieve_ptr(mbar, "rw") if mbar is not None else tir.const(0, "uint32")
    C_coords = [r.min for r in C.region]
    return tir.call_intrin("handle", tir.op.Op.get(op_key), A, B, C, transpose_A, transpose_B, M, N,
                           K, policy, clear_accum, stride_a, stride_b, offset_a, offset_b, k_pack,
                           wg_wait, mbarptr, C_coords[0], C_coords[1])


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
        "tl.gemm",
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
        "tl.gemm_py",
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
gemm = gemm_v1 if _env.use_gemm_v1() else gemm_v2

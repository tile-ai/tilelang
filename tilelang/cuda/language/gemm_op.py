"""CUDA dialect of ``T.gemm``: the common GEMM plus CUDA-specific knobs."""

from __future__ import annotations

from tvm import tirx

from tilelang._typing import BufferLikeType
from tilelang.language.experimental.gemm_sp_op import _gemm_sp_impl
from tilelang.language.gemm_op import BarrierType, GemmWarpPolicy, _gemm_impl

__all__ = ["gemm", "gemm_sp"]


def gemm(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    mbar: BarrierType | None = None,
    annotations: dict | None = None,
) -> tirx.PrimExpr:
    """TileLang GEMM operator for CUDA.

    Same semantics as the common :func:`tilelang.language.gemm_op.gemm`: the
    default synchronous GEMM. On Hopper, if the compiler selects WGMMA
    lowering, TileLang inserts the corresponding wait implicitly. On Blackwell
    TCGEN5MMA, TileLang inserts the corresponding
    ``mbarrier_wait_parity(...)`` implicitly after issue.

    For manual asynchronous scheduling, use ``T.wgmma_gemm(...)`` with
    ``T.wait_wgmma(...)`` on Hopper, or ``T.tcgen05_gemm(...)`` with
    ``T.mbarrier_wait_parity(...)`` on Blackwell.

    Args:
        A (BufferLikeType, i.e. Buffer | BufferLoad | BufferRegion, or Var): Input buffer A.
        B (BufferLikeType): Input buffer B.
        C (BufferLikeType): Output buffer C.
        transpose_A (bool): Whether to transpose A. Defaults to False.
        transpose_B (bool): Whether to transpose B. Defaults to False.
        policy (GemmWarpPolicy): GEMM warp partition policy.
        clear_accum (bool): Whether to clear the accumulator.
        mbar (BarrierType, i.e. Buffer | BufferLoad, or Var, optional): Mbarrier in Blackwell.
            Required when this GEMM lowers to TCGEN5MMA. Defaults to None.
        annotations (Optional[dict]): Additional annotations.

    Returns:
        tirx.Call: A handle to the GEMM operation.
    """
    return _gemm_impl(
        "tl.tileop.gemm",
        A,
        B,
        C,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        mbar,
        annotations=annotations,
    )


def gemm_sp(
    A_sparse: BufferLikeType | tirx.Var,
    E: BufferLikeType | tirx.Var,
    B: BufferLikeType | tirx.Var,
    C: BufferLikeType | tirx.Var,
    transpose_A: bool = False,
    transpose_E: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    wg_wait: int = 0,
    annotations: dict | None = None,
) -> tirx.Call:
    """Sparse GEMM (2:4 structured sparsity) for CUDA.

    Same semantics as the common :func:`tilelang.language.experimental.gemm_sp_op.gemm_sp`.
    ``wg_wait`` is the Hopper warpgroup wait count consumed when the WGMMA SP
    lowering is selected (``-1`` defers the wait to an explicit
    ``T.wait_wgmma``); it rides in the tile-op annotations.

    Args:
        A_sparse: Compressed sparse matrix containing only non-zero elements.
        E: Metadata tensor encoding the sparsity pattern of A.
        B: Dense input matrix.
        C: Output accumulator matrix.
        transpose_A: Whether to transpose A. Defaults to False.
        transpose_E: Whether to transpose E. Defaults to False.
        transpose_B: Whether to transpose B. Defaults to False.
        policy: Warp partition policy. Defaults to GemmWarpPolicy.Square.
        clear_accum: Whether to zero the accumulator before computation. Defaults to False.
        wg_wait: Warp group wait count. Defaults to 0.
        annotations: Additional annotations; values in it take precedence.

    Returns:
        tirx.Call: A handle to the sparse GEMM operation.
    """
    ann = dict(annotations) if annotations is not None else {}
    if wg_wait != 0:
        ann.setdefault("wg_wait", wg_wait)
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
        annotations=ann or None,
    )

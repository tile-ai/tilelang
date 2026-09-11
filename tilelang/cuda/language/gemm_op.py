"""CUDA dialect of ``T.gemm``: the common GEMM plus CUDA-specific knobs."""

from __future__ import annotations

from tvm import tirx

from tilelang._typing import BufferLikeType
from tilelang.language.gemm_op import BarrierType, GemmWarpPolicy, _gemm_impl

__all__ = ["gemm"]


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
        0,
        mbar,
        annotations=annotations,
    )

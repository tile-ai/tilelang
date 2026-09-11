"""ROCm dialect of ``T.gemm``: the common GEMM plus ROCm-specific knobs."""

from __future__ import annotations

from tvm import tirx

from tilelang._typing import BufferLikeType
from tilelang.language.gemm_op import GemmWarpPolicy, _gemm_impl
from tilelang.language.utils import _normalize_annotations

__all__ = ["gemm"]


def gemm(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    annotations: dict | None = None,
) -> tirx.PrimExpr:
    """TileLang GEMM operator for ROCm.

    Same semantics as the common :func:`tilelang.language.gemm_op.gemm`.
    ``k_pack`` packs multiple matrix-core operations along K in the MFMA/WMMA
    lowering (CDNA/RDNA); it is a performance knob with no counterpart on
    other targets.

    Args:
        A (BufferLikeType, i.e. Buffer | BufferLoad | BufferRegion, or Var): Input buffer A.
        B (BufferLikeType): Input buffer B.
        C (BufferLikeType): Output buffer C.
        transpose_A (bool): Whether to transpose A. Defaults to False.
        transpose_B (bool): Whether to transpose B. Defaults to False.
        policy (GemmWarpPolicy): GEMM warp partition policy.
        clear_accum (bool): Whether to clear the accumulator.
        k_pack (int): Number of packed matrix cores along K. Must be 1 or 2.
            Defaults to 1.
        annotations (Optional[dict]): Additional annotations.

    Returns:
        tirx.Call: A handle to the GEMM operation.
    """
    if not (isinstance(k_pack, int) and not isinstance(k_pack, bool) and k_pack in (1, 2)):
        raise ValueError(f"T.gemm k_pack must be an int equal to 1 or 2, got {k_pack!r}")
    ann = _normalize_annotations(annotations)
    if k_pack != 1:
        ann.setdefault("k_pack", k_pack)
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
        None,
        annotations=ann,
    )

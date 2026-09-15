"""Ascend dialect of the GEMM operators: the common ops plus Ascend hints."""

from __future__ import annotations

from tilelang._typing import BufferLikeType
from tilelang.language.gemm_op import _gemm_impl
from tilelang.language.utils import buffer_region_to_tile_region
from tilelang.tileop.base import GemmWarpPolicy
from tilelang.utils.language import retrieve_shape, to_buffer_region
from tvm import tirx

__all__ = ["gemm", "blockscaled_gemm"]


def blockscaled_gemm(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    sfa: BufferLikeType | None = None,
    sfb: BufferLikeType | None = None,
    transpose_A: bool = False,
    transpose_B: bool = False,
    clear_accum: bool = False,
    unit_flag_ctrl: int | tirx.PrimExpr | None = None,
) -> tirx.PrimExpr:
    """Ascend block-scaled MXFP8 GEMM.

    Scale buffers are required for L1 A/B inputs. For L0 A/B inputs the scales
    are expected to have been loaded by the preceding ``T.copy(..., scale=...)``.
    """

    ann = {"blockscaled": 1}
    if unit_flag_ctrl is not None:
        ann["unit_flag_ctrl"] = unit_flag_ctrl
    call = _gemm_impl(
        "tl.tileop.gemm",
        A,
        B,
        C,
        transpose_A=transpose_A,
        transpose_B=transpose_B,
        policy=GemmWarpPolicy.Square,
        clear_accum=clear_accum,
        mbar=None,
        annotations=ann,
    )
    if sfa is None and sfb is None:
        return call
    assert sfa is not None and sfb is not None, "block-scaled GEMM requires both sfa and sfb"
    # Mirror T.tcgen05_gemm_blockscaled's wire format: the scale-factor
    # regions ride as trailing tl.tileop.gemm args (SFA, SFB, sf_k_start),
    # parsed into GemmNode's sfaRegion/sfbRegion. Appending to the common
    # builder's call keeps the positional contract in one place.
    sfa_region = to_buffer_region(sfa, access_type="r")
    sfb_region = to_buffer_region(sfb, access_type="r")
    sfa_arg = buffer_region_to_tile_region(sfa_region, "r", list(retrieve_shape(sfa_region)))
    sfb_arg = buffer_region_to_tile_region(sfb_region, "r", list(retrieve_shape(sfb_region)))
    return tirx.call_intrin(
        "handle",
        call.op,
        *call.args,
        sfa_arg,
        sfb_arg,
        tirx.const(0, dtype="int32"),
        annotations=call.annotations,
    )


def gemm(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    unit_flag_ctrl: int | tirx.PrimExpr | None = None,
    annotations: dict | None = None,
) -> tirx.PrimExpr:
    """TileLang GEMM operator, with the Ascend unit-flag hint.

    Same semantics as the common :func:`tilelang.language.gemm_op.gemm`.
    ``unit_flag_ctrl`` records the Cube unit-flag control on the tile op, which
    the Ascend lowering pairs with the following accumulator drain; ``None``
    omits the annotation and lowers as 0.

    On Ascend, L0 operand regions specify the effective MAD M/N/K. Their
    trailing matrix dimensions must start at zero and describe a compact tile.
    L0 allocations and producer copies may be padded for hardware alignment;
    for example, a transposed FP32 load can copy K32 while GEMM consumes
    ``A[:, :24]`` and ``B[:, :24]``. Copy regions must cover the physical
    transfer. Allocation padding remains part of the storage budget.

    Args:
        A, B, C (BufferLikeType): Input A, input B and output C.
        transpose_A (bool): Whether to transpose A. Defaults to False.
        transpose_B (bool): Whether to transpose B. Defaults to False.
        policy (GemmWarpPolicy): GEMM warp partition policy.
        clear_accum (bool): Whether to clear the accumulator.
        unit_flag_ctrl (int | tirx.PrimExpr, optional): Unit flag control for the
            instruction. ``None`` omits the annotation and lowers as 0.
        annotations (Optional[dict]): Additional annotations; values in it take
            precedence over the individual keywords.

    Returns:
        tirx.PrimExpr: A handle to the GEMM operation.
    """
    ann: dict = dict(annotations) if annotations else {}
    if unit_flag_ctrl is not None:
        ann.setdefault("unit_flag_ctrl", unit_flag_ctrl)
    return _gemm_impl(
        "tl.tileop.gemm",
        A,
        B,
        C,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        None,
        annotations=ann or None,
    )

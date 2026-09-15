"""Ascend dialect of the GEMM operators: the common ops plus Ascend hints."""

from __future__ import annotations

import tilelang.language as T
from tilelang._typing import BufferLikeType
from tilelang.language.gemm_op import _gemm_impl
from tilelang.language.utils import _normalize_annotations, buffer_region_to_tile_region
from tilelang.tileop.base import GemmWarpPolicy
from tilelang.utils.language import prim_expr_equal, retrieve_shape, to_buffer_region
from tvm import arith, tirx

__all__ = ["gemm", "blockscaled_gemm"]


def _legalize_argument(arg):
    """Convert a let-bound variable to its corresponding buffer."""
    if isinstance(arg, tirx.Var) and T.has_let_value(arg):
        return T.get_let_value(arg).buffer
    return arg


def _prove_equal(expr1, expr2) -> bool:
    return prim_expr_equal(expr1, expr2) or arith.Analyzer().can_prove_equal(expr1, expr2)


def _l0_gemm_call(A_region, B_region, C_region, transpose_A, transpose_B, policy, clear_accum, annotations):
    """Assemble tl.tileop.gemm for an L0-input MAD tile.

    Mirrors the common ``_gemm_impl`` assembly (same positional contract),
    with the two Ascend L0 differences: the serialized M/N/K stay static by
    reading the allocation shape — the region extents give the effective,
    possibly symbolic MAD geometry, which the L0 lowering reads from the
    BufferRegions — and a higher-order C region is allowed for multi-version
    L0C accumulators.
    """
    annotations = _normalize_annotations(annotations)

    A_shape = retrieve_shape(A_region)
    B_shape = retrieve_shape(B_region)
    C_shape = retrieve_shape(C_region)
    for shape, name in ((A_shape, "A"), (B_shape, "B"), (C_shape, "C")):
        assert len(shape) >= 2, f"current only support {name} as a 2D or higher-order tensor"
        for i in range(len(shape) - 2):
            assert shape[i] == 1, (
                f"current only support {name} as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"
            )

    M, N = C_shape[-2], C_shape[-1]
    M_A = A_shape[-1] if transpose_A else A_shape[-2]
    K = A_shape[-2] if transpose_A else A_shape[-1]
    N_B = B_shape[-2] if transpose_B else B_shape[-1]
    K_B = B_shape[-1] if transpose_B else B_shape[-2]
    assert _prove_equal(M_A, M), f"T.gemm M shape check failed: M_A = {M_A}, M_C = {M}"
    assert _prove_equal(K, K_B), f"T.gemm K shape check failed: K_A = {K}, K_B = {K_B}"
    assert _prove_equal(N_B, N), f"T.gemm N shape check failed: N_B = {N_B}, N_C = {N}"

    node_M, node_N = C_region.buffer.shape[-2:]
    node_K = A_region.buffer.shape[-2] if transpose_A else A_region.buffer.shape[-1]
    for name, dim in (("M", node_M), ("N", node_N), ("K", node_K)):
        if not isinstance(dim, tirx.IntImm):
            raise ValueError(f"T.gemm requires static tile dimensions, but {name} is symbolic: {dim}")

    C_coords = [r.min for r in C_region.region[-2:]]
    A_arg = buffer_region_to_tile_region(A_region, "r", list(A_shape))
    B_arg = buffer_region_to_tile_region(B_region, "r", list(B_shape))
    C_arg = buffer_region_to_tile_region(C_region, "w" if isinstance(clear_accum, bool) and clear_accum else "rw", list(C_shape))

    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.gemm"),
        A_arg,
        B_arg,
        C_arg,
        transpose_A,
        transpose_B,
        node_M,
        node_N,
        node_K,
        policy,
        clear_accum,
        tirx.const(0, dtype="int32"),  # mbar placeholder, ignored unless BufferLoad
        C_coords[-2],
        C_coords[-1],
        annotations=annotations,
    )


def _dialect_gemm_call(A, B, C, transpose_A, transpose_B, policy, clear_accum, annotations):
    """Route an Ascend gemm: L0 MAD tiles get their own assembly."""
    A_region = to_buffer_region(_legalize_argument(A))
    B_region = to_buffer_region(_legalize_argument(B))
    C_region = to_buffer_region(_legalize_argument(C))
    if A_region.buffer.scope() == "shared.l0a" and B_region.buffer.scope() == "shared.l0b" and C_region.buffer.scope() == "shared.l0c":
        return _l0_gemm_call(A_region, B_region, C_region, transpose_A, transpose_B, policy, clear_accum, annotations)
    return _gemm_impl(
        "tl.tileop.gemm",
        A,
        B,
        C,
        transpose_A=transpose_A,
        transpose_B=transpose_B,
        policy=policy,
        clear_accum=clear_accum,
        mbar=None,
        annotations=annotations,
    )


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
    call = _dialect_gemm_call(A, B, C, transpose_A, transpose_B, GemmWarpPolicy.Square, clear_accum, ann)
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
    return _dialect_gemm_call(A, B, C, transpose_A, transpose_B, policy, clear_accum, ann or None)

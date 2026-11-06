"""Ascend dialect of the GEMM operators: the common ops plus Ascend hints."""

from __future__ import annotations

from tilelang.language import _intrinsics
from tilelang.language import _tensor_spec
from tvm_tirx import ir as _ir

import tilelang.language as T
import tvm_ffi
from tilelang._typing import BufferLikeType
from tilelang.language.gemm_op import _gemm_impl
from tilelang.language.utils import _normalize_annotations, buffer_region_to_tile_region
from tilelang.tileop.base import GemmWarpPolicy
from tilelang.utils.language import prim_expr_equal, retrieve_shape, to_buffer_region
from tvm_tirx import sym


__all__ = ["gemm", "gemm_blockscaled"]


def _legalize_argument(arg):
    """Recover a macro operand while retaining its selected tensor region."""
    while isinstance(arg, _ir.Var) and not _tensor_spec.is_buffer_like(arg) and T.has_let_value(arg):
        arg = T.get_let_value(arg)
    return arg


def _prove_equal(expr1, expr2) -> bool:
    return prim_expr_equal(expr1, expr2) or sym.Analyzer().can_prove_equal(expr1, expr2)


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

    node_M, node_N = C_region.source.ty.shape[-2:]
    node_K = A_region.source.ty.shape[-2] if transpose_A else A_region.source.ty.shape[-1]
    for name, dim in (("M", node_M), ("N", node_N), ("K", node_K)):
        if not isinstance(dim, _ir.IntImm):
            raise ValueError(f"T.gemm requires static tile dimensions, but {name} is symbolic: {dim}")

    C_coords = [r.begin for r in C_region.region[-2:]]
    A_arg = buffer_region_to_tile_region(A_region, "r", list(A_shape))
    B_arg = buffer_region_to_tile_region(B_region, "r", list(B_shape))
    C_arg = buffer_region_to_tile_region(C_region, "w" if isinstance(clear_accum, bool) and clear_accum else "rw", list(C_shape))

    return _intrinsics.call_intrin(
        "handle",
        _ir.Op.get("tl.tileop.gemm"),
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
        _ir.const(0, dtype="int32"),  # mbar placeholder, ignored unless BufferLoad
        C_coords[-2],
        C_coords[-1],
        annotations=annotations,
    )


def _dialect_gemm_call(A, B, C, transpose_A, transpose_B, policy, clear_accum, annotations):
    """Route an Ascend gemm: L0 MAD tiles get their own assembly."""
    A_region = to_buffer_region(_legalize_argument(A))
    B_region = to_buffer_region(_legalize_argument(B))
    C_region = to_buffer_region(_legalize_argument(C))
    if (
        str(A_region.source.ty.scope) == "shared.l0a"
        and str(B_region.source.ty.scope) == "shared.l0b"
        and str(C_region.source.ty.scope) == "shared.l0c"
    ):
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


def gemm_blockscaled(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    SFA: BufferLikeType,
    SFB: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    clear_accum: bool = False,
    unit_flag_ctrl: int | _ir.Expr | None = None,
) -> _ir.Expr:
    """Ascend block-scaled MXFP8 GEMM, shadowing the common ``T.gemm_blockscaled``.

    ``SFA``/``SFB`` are required. For L1 A/B inputs they are the L1
    scale-factor buffers, consumed directly by the fused L1 lowering. For L0
    A/B inputs they are the MX scale-factor handles of the operand tiles
    (:func:`tilelang.ascend.language.alloc_l0a_sf` /
    :func:`~tilelang.ascend.language.alloc_l0b_sf`), loaded by a preceding
    ``T.copy(sf_l1, handle)``. Each handle must be allocated for the
    corresponding data buffer, select the same leading indices, and describe
    its compact trailing tile (zero origin and matching K). The MAD reads
    the slots implied by its A/B data
    addresses, so the operands here carry the read-region truth and select
    the block-scaled lowering.

    Unlike the common surface, ``k_start`` and the ``sf_*_granularity_k``
    knobs are implicit: the Ascend lowering derives the scale K offset from
    the SFA region slice, and MX scales cover 32 K elements per factor.
    """

    ann: dict = {}
    if unit_flag_ctrl is not None:
        ann["unit_flag_ctrl"] = unit_flag_ctrl
    assert SFA is not None and SFB is not None, "block-scaled GEMM requires both SFA and SFB"
    call = _dialect_gemm_call(A, B, C, transpose_A, transpose_B, GemmWarpPolicy.Square, clear_accum, ann or None)
    # Re-emit the dense slots under the dedicated tl.tileop.gemm_blockscaled
    # op with the SFA/SFB regions and k_start appended (the 16-slot protocol
    # parsed by GemmBlockScaled). k_start stays 0: the Ascend lowering derives
    # the scale K offset from the SFA region slice instead. MX scales cover 32
    # K elements per factor.
    sfa_region = to_buffer_region(SFA, access_type="r")
    sfb_region = to_buffer_region(SFB, access_type="r")
    sfa_arg = buffer_region_to_tile_region(sfa_region, "r", list(retrieve_shape(sfa_region)))
    sfb_arg = buffer_region_to_tile_region(sfb_region, "r", list(retrieve_shape(sfb_region)))
    operand_count = int(call.attrs["operand_count"])
    annotations = dict(tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[operand_count], call.attrs["annotation_schema"]))
    annotations["sf_a_granularity_k"] = 32
    annotations["sf_b_granularity_k"] = 32
    return _intrinsics.call_intrin(
        "handle",
        _ir.Op.get("tl.tileop.gemm_blockscaled"),
        *call.args[:operand_count],
        sfa_arg,
        sfb_arg,
        _ir.const(0, dtype="int32"),
        annotations=annotations,
        span=call.loc,
    )


def gemm(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    unit_flag_ctrl: int | _ir.Expr | None = None,
    annotations: dict | None = None,
) -> _ir.Expr:
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

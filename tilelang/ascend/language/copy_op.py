"""Ascend dialect of the copy operators: the common ops plus Ascend copy hints."""

from __future__ import annotations

from typing import Any

from tilelang._typing import BufferLikeType
from tilelang.language.copy_op import (
    copy as _common_copy,
)
from tilelang.language.utils import _normalize_annotations
from tilelang.utils.language import to_buffer_region
from tvm import tirx

__all__ = ["copy", "dual_copy"]


def _get_shape(buf: BufferLikeType) -> list:
    """Extract shape list from Buffer, BufferLoad, or BufferRegion."""
    if isinstance(buf, tirx.Buffer):
        return list(buf.shape)
    if isinstance(buf, tirx.BufferLoad):
        return list(buf.buffer.shape)
    if isinstance(buf, tirx.BufferRegion):
        return [r.extent for r in buf.region]
    raise TypeError(f"Cannot get shape from {type(buf)}")


def _get_allocation_shape(buf: BufferLikeType) -> list:
    """Extract the underlying allocation shape rather than region extents."""
    if isinstance(buf, tirx.Buffer):
        return list(buf.shape)
    if isinstance(buf, (tirx.BufferLoad, tirx.BufferRegion)):
        return list(buf.buffer.shape)
    raise TypeError(f"Cannot get allocation shape from {type(buf)}")


def _is_half(full_dim, half_dim) -> bool:
    """Check whether full_dim is exactly twice half_dim."""
    if isinstance(full_dim, (int, tirx.IntImm)) and isinstance(half_dim, (int, tirx.IntImm)):
        full_val = full_dim if isinstance(full_dim, int) else full_dim.value
        half_val = half_dim if isinstance(half_dim, int) else half_dim.value
        return half_val * 2 == full_val
    try:
        from tvm.ir import structural_equal

        return structural_equal(half_dim * 2, full_dim)
    except Exception:
        return False


def dual_copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    unit_flag_ctrl: int | tirx.PrimExpr | None = None,
    l2_cache_ctrl: int | str = 0,
) -> tirx.PrimExpr | tirx.Stmt:
    """Copy a region using an M- or N-split across the two AIVs.

    ``dual_copy`` is a mixed-kernel operation: the user must also provide the
    Cube-side work in the enclosing kernel. It is not supported as a way to
    turn an otherwise pure Vector kernel into a two-AIV launch; the compiler
    assumes this contract and does not synthesize Cube-side work.

    With auto-scheduling disabled, write a manual mixed kernel as ``T.Kernel``
    containing explicit ``T.Cube()`` and ``T.Vector()`` blocks, and place each
    software dual copy inside the ``T.Vector()`` block. The rewrite pass reuses
    that block's ``cthread`` sid and rejects unscoped software dual copies.

    Software dual copies also accept one-dimensional regions whose sole extent
    has a 2:1 ratio. For regions with rank at least two, exactly one trailing
    dimension must have a 2:1 extent ratio between source and destination. The
    supported memory paths are:

    - L0C→UB: lowered as a hardware dual-destination copy.
    - GM→UB, UB→GM, and UB→L1: lowered as an ordinary per-AIV copy
      indexed by the ``cthread`` sid.

    For UB→L1, the InsertNd2Nz pass also handles ND→NZ format conversion.

    - ``[M, N]`` ↔ ``[M/2, N]``: M-split (``dual_dst_ctl=0b01``)
    - ``[M, N]`` ↔ ``[M, N/2]``: N-split (``dual_dst_ctl=0b10``)

    Hardware L0C→UB requires rank at least two. Its N-split additionally
    requires the full N extent to be a multiple of 32.

    The larger region is the full logical tile. Each AIV processes the
    corresponding half of that tile along the inferred split dimension.
    A statically known full extent must therefore be even; odd extents are
    rejected rather than rounded into unequal partitions.

    Args:
        src: Source region in GM, L0C, or UB, according to the supported paths.
        dst: Destination region in UB, GM, or L1, according to the supported paths.
        unit_flag_ctrl: Unit flag control (0=manual sync, 3=pipelined with mad).
            ``None`` omits the annotation and lowers as 0.
        l2_cache_ctrl (int | str): Ascend L2 cache control policy for the UB→GM
            store path. Accepts an integer or case-insensitive string name
            (e.g. ``"notalloc_clean"``). Defaults to 0 (``"normal_fv"``).
            Ascend only; ignored on other backends.

    Raises:
        ValueError: If the split direction cannot be inferred from shapes.
    """
    src_shape = _get_shape(src)
    dst_shape = _get_shape(dst)

    alloc_src_shape = _get_allocation_shape(src)
    alloc_dst_shape = _get_allocation_shape(dst)

    if len(src_shape) == 1 and len(dst_shape) == 1:
        split_candidates = [(-1, 2)]
        ratio_requirement = "The sole dimension must have an exact 2:1 extent ratio."
    elif len(src_shape) >= 2 and len(dst_shape) >= 2:
        split_candidates = [(-2, 1), (-1, 2)]
        ratio_requirement = "Exactly one trailing dimension must be halved."
    else:
        raise ValueError("dual_copy requires source and destination regions that are both one-dimensional or both have rank at least two")

    def infer_halved_axes(lhs_shape, rhs_shape):
        return [_is_half(lhs_shape[axis], rhs_shape[axis]) or _is_half(rhs_shape[axis], lhs_shape[axis]) for axis, _ in split_candidates]

    halved_axes = infer_halved_axes(src_shape, dst_shape)
    # Runtime tail regions may not retain a structural 2:1 relationship with
    # the fixed allocation. Fall back to allocation shapes for split inference
    # while preserving the actual region extents for the copy.
    if sum(halved_axes) != 1:
        halved_axes = infer_halved_axes(alloc_src_shape, alloc_dst_shape)
    if sum(halved_axes) != 1:
        raise ValueError(f"Cannot infer dual_copy split direction: src shape {src_shape} vs dst shape {dst_shape}. " + ratio_requirement)

    split_axis, dual_dst_ctl = split_candidates[halved_axes.index(True)]

    # Convert both to buffer regions to bypass the shape equality check in copy()
    # This allows src[M,N] to be copied to dst[M/2,N] or dst[M,N/2]
    src_region = to_buffer_region(src)
    dst_region = to_buffer_region(dst)

    # WARNING: dav-950 does NOT support quant_pre (type conversion) with
    # fixpipe (cc->ub) dual-destination copies. If src and dst dtypes differ, the hardware may
    # silently produce incorrect results.
    src_dtype = src_region.buffer.dtype
    dst_dtype = dst_region.buffer.dtype
    if src_dtype != dst_dtype and src_region.buffer.scope() == "shared.l0c":
        raise ValueError(
            f"dual_copy: src dtype ({src_dtype}) != dst dtype ({dst_dtype}). "
            "Type conversion during cc->ub dual-destination copy is NOT supported on dav-950. "
            "Use a separate copy or cast inside a VF block instead."
        )

    annotations = {"dual_dst_ctl": dual_dst_ctl}
    if _is_half(dst_shape[split_axis], src_shape[split_axis]) or _is_half(alloc_dst_shape[split_axis], alloc_src_shape[split_axis]):
        annotations["double"] = True

    return copy(
        src_region,
        dst_region,
        annotations=annotations,
        unit_flag_ctrl=unit_flag_ctrl,
        l2_cache_ctrl=l2_cache_ctrl,
    )


# Short-name → int mapping for L2 cache control.
_L2_CACHE_CTRL_MAP = {
    # ── LD_L2CacheType (Load / GM→L1) ──
    "NORMAL_FV": 0,
    "NORMAL_LV": 1,
    "NORMAL_PERS": 2,
    "NORMAL_PREF": 3,
    "NOTALLOC_KEEP": 4,
    "NOTALLOC_CLEAN": 5,
    "NOTALLOC_DROP": 6,
    "IDS_FV": 8,
    "IDS_LV": 9,
    "IDS_PERS": 10,
    "IDS_PREF": 11,
    "EXCLUSIV_FV": 12,
    "EXCLUSIV_LV": 13,
    "EXCLUSIV_PERS": 14,
    "EXCLUSIV_PREF": 15,
    "INVALID": 16,
    # ── ST_L2CacheType (Store / UB→GM) — names that differ from LD ──
    "NORMAL_RED": 3,
    "NOTALLOC_CI": 4,
    "NOTALLOC_PW": 5,
    "NOTALLOC_PI": 6,
    "NOTALLOC_RED": 7,
    "WBH_FV": 8,
    "WBH_LV": 9,
    "WBH_PERS": 10,
    "WBH_RED": 11,
    "WTS_FV": 12,
    "WTS_LV": 13,
    "WTS_PERS": 14,
    "WTS_RED": 15,
}


def _normalize_l2_cache_ctrl(value: int | str | tirx.IntImm | tirx.StringImm | None) -> int | None:
    """Convert a string l2_cache_ctrl name to its integer value.

    Matching is case-insensitive and suffix-based:
    ``"NORMAL_FV"``, ``"normal_fv"``, and ``"L2_CACHE_HINT_NORMAL_FV"``
    all map to ``0``.  Integers pass through unchanged.
    """
    if value is None:
        return None
    if isinstance(value, tirx.IntImm):
        value = int(value)
    elif isinstance(value, tirx.StringImm):
        value = value.value
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        key = value.upper()
        # Exact match
        if key in _L2_CACHE_CTRL_MAP:
            return _L2_CACHE_CTRL_MAP[key]
        # Suffix match (e.g. "L2_CACHE_HINT_NORMAL_FV" or "_NORMAL_FV")
        for map_key, map_val in _L2_CACHE_CTRL_MAP.items():
            if key.endswith(map_key):
                return map_val
        raise ValueError(f"Unknown l2_cache_ctrl string {value!r}. Valid suffixes: {sorted(_L2_CACHE_CTRL_MAP.keys())}")
    raise TypeError(f"l2_cache_ctrl must be int, str, or None, got {type(value)}")


def copy(  # noqa: A001
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    coalesced_width: int | None = None,
    transpose: bool = False,
    l2_cache_ctrl: int | str | None = None,
    unit_flag_ctrl: int | tirx.PrimExpr | None = None,
    sub_blockid: int | tirx.PrimExpr | None = None,
    pad_value: int | float | tirx.PrimExpr | None = None,
    data_select: bool = False,
    annotations: dict | None = None,
    loop_layout: Any | None = None,
) -> tirx.PrimExpr | tirx.Stmt:
    """Copy data between memory regions, with Ascend DMA lowering hints.

    Same semantics as the common :func:`tilelang.language.copy_op.copy`; the
    extra keywords steer how the Ascend backend lowers the copy (GM↔L1 L2
    cache control, ND/NZ transpose, MTE pad handling, Cube unit-flag control
    for the following MAD, sub-block routing). They are performance hints
    recorded on the tile op: compiling the same kernel for a target that has
    no use for them leaves the result unchanged.

    A copy whose destination is an MX scale-factor handle
    (:func:`tilelang.ascend.language.alloc_l0a_sf` /
    :func:`~tilelang.ascend.language.alloc_l0b_sf`) loads the per-block
    scales into the L0 tile's MX slot shadow instead of moving data.

    Args:
        src: Source memory region (Buffer, BufferLoad or BufferRegion).
        dst: Destination memory region.
        coalesced_width (Optional[int], keyword-only): Width for coalesced
            memory access. Defaults to None.
        transpose (bool, keyword-only): Ascend GM→L1 only. Emit dn2nz (which
            transposes the N/D mapping) instead of nd2nz. Defaults to False.
        l2_cache_ctrl (Optional[Union[int, str]], keyword-only): L2 cache
            control for the GM↔L1 and UB↔GM DMA paths. Accepts the raw integer
            or a cache-policy name (case-insensitive, suffix-matched), e.g.
            ``"NOTALLOC_KEEP"``.
        unit_flag_ctrl (Optional[Union[int, PrimExpr]], keyword-only): Unit flag
            control for the Cube instruction; ``None`` omits the annotation.
        sub_blockid (Optional[Union[int, PrimExpr]], keyword-only): Sub-block id
            that routes the copy to one of the AIV sub-blocks.
        pad_value (Optional[Union[int, float, PrimExpr]], keyword-only): Ascend
            GM→UB only. Round the row width up to the next 32B boundary and fill
            the pad lanes with this value. Emits a leading
            ``T.ascend_set_copy_pad_value(value)`` so AutoSchedule syncs the
            pad-register write before the copy. Mutually exclusive with
            ``data_select``.
        data_select (bool, keyword-only): Ascend GM→UB only. Same right-pad
            behavior as ``pad_value``, but reuses whatever the hardware pad
            register currently holds, which the caller must have set via
            ``T.ascend_set_copy_pad_value(...)`` beforehand.
        annotations (Optional[dict], keyword-only): Additional annotations
            dict; values in it take precedence over the individual keywords.
        loop_layout (Optional[Fragment], keyword-only): Parallel loop layout
            hint for the SIMT copy path.

    Returns:
        tirx.PrimExpr | tirx.Stmt: A handle to the copy operation.
    """
    ann = _normalize_annotations(annotations)

    # Ascend GM→L1: use dn2nz (transpose N/D mapping) instead of nd2nz
    if transpose and "transpose" not in ann:
        ann["transpose"] = tirx.IntImm("int32", 1)

    # Ascend DMA: L2 cache control for GM↔L1 and UB↔GM paths.
    if l2_cache_ctrl is not None and "l2_cache_ctrl" not in ann:
        ann["l2_cache_ctrl"] = l2_cache_ctrl
    if "l2_cache_ctrl" in ann:
        ann["l2_cache_ctrl"] = _normalize_l2_cache_ctrl(ann["l2_cache_ctrl"])
    if unit_flag_ctrl is not None and "unit_flag_ctrl" not in ann:
        ann["unit_flag_ctrl"] = unit_flag_ctrl
    if sub_blockid is not None and "sub_blockid" not in ann:
        ann["sub_blockid"] = sub_blockid

    if pad_value is not None and data_select:
        raise ValueError(
            "T.copy: pad_value and data_select are mutually exclusive. Pass "
            "pad_value to set the fill value on this copy, or data_select to "
            "reuse a pad value already set via T.ascend_set_copy_pad_value()."
        )
    if pad_value is not None:
        import tvm.script.ir_builder.tir as tb_tir
        from tilelang.ascend.language.dma import ascend_set_copy_pad_value

        # Emit SetPadValue as a leading TIR statement (not folded into the copy
        # lowering) so it exists in the IR before AutoSchedule, which then
        # inserts the PIPE_S -> PIPE_MTE2 sync between the scalar pad-register
        # write and the MTE2 copy that reads it. Folding it in at LowerTileOp
        # (which runs after AutoSchedule) would leave the two unsynchronized and
        # the copy would read a stale pad register. The copy itself then just
        # reuses the register via data_select. dst is a
        # Buffer/BufferRegion/BufferLoad; all but a raw Buffer expose .buffer.
        if not isinstance(dst, (tirx.Buffer, tirx.BufferRegion, tirx.BufferLoad)):
            raise TypeError(
                "T.copy: pad_value requires dst to be a Buffer, BufferRegion, or "
                f"BufferLoad so the pad fill dtype can be derived, got {type(dst)}."
            )
        dst_buf = dst.buffer if isinstance(dst, (tirx.BufferRegion, tirx.BufferLoad)) else dst
        tb_tir.evaluate(ascend_set_copy_pad_value(pad_value, dtype=str(dst_buf.dtype)))
        ann["data_select"] = tirx.IntImm("int32", 1)
    if data_select and "data_select" not in ann:
        ann["data_select"] = tirx.IntImm("int32", 1)

    ret = _common_copy(
        src,
        dst,
        coalesced_width=coalesced_width,
        annotations=ann or None,
        loop_layout=loop_layout,
    )
    # Respell the common tile op as the Ascend dialect op so the C++ side
    # parses it into the typed AscendCopyNode. Scalar copies come back as a
    # plain BufferStore and stay untouched.
    if isinstance(ret, tirx.Call) and ret.op.same_as(tirx.op.Op.get("tl.tileop.copy")):
        return tirx.call_intrin(
            "handle",
            tirx.op.Op.get("tl.tileop.ascend_copy"),
            *ret.args,
            annotations=ret.annotations,
        )
    return ret

"""Copy operations exposed on the TileLang language surface."""

from __future__ import annotations
from typing import Literal, Any

from tilelang._typing import BufferLikeType
from tilelang.utils.language import (
    to_buffer_region,
    legalize_pairwise_extents,
)
from tilelang.utils.deprecated import deprecated
from tilelang.language.utils import get_extent, buffer_region_to_tile_region, _normalize_annotations
import tvm
from tvm import ir, tirx


def _normalize_copy_regions_with_extents(
    src: BufferLikeType, dst: BufferLikeType
) -> tuple[
    tirx.BufferRegion | tirx.BufferLoad | tirx.Buffer,
    tirx.BufferRegion | tirx.BufferLoad | tirx.Buffer,
    list[tirx.PrimExpr] | None,
    list[tirx.PrimExpr] | None,
]:
    # If both side are buffers, check total element counts match.  Shape
    # equality is NOT required: Ascend fractal copies between differently-
    # major'd buffers (e.g. [K,M] -> [M,K]) have transposed shapes but equal
    # element counts.
    if isinstance(src, tirx.Buffer) and isinstance(dst, tirx.Buffer):
        from tvm import arith

        src_elems = 1
        for s in src.shape:
            src_elems = src_elems * s
        dst_elems = 1
        for s in dst.shape:
            dst_elems = dst_elems * s
        analyzer = arith.Analyzer()
        assert analyzer.can_prove_equal(src_elems, dst_elems), (
            f"T.copy src/dst element count mismatch: src={src.shape} ({src_elems}) vs dst={dst.shape} ({dst_elems})"
        )

    src_extent = get_extent(src)
    dst_extent = get_extent(dst)

    src_is_scalar_load = src_extent is None and isinstance(src, tirx.BufferLoad)
    dst_is_scalar_load = dst_extent is None and isinstance(dst, tirx.BufferLoad)

    # copy(buffer_a[i], buffer_b[i]) where both are BufferLoad nodes
    # In this case, lower it to a simple BufferStore: buffer_b[i] = buffer_a[i]
    if src_is_scalar_load and dst_is_scalar_load:
        return src, dst, None, None

    assert src_extent or dst_extent, "Can't deduce copy extents from args. Both src and dst miss extents info."
    # Treat missing extent as length-matched ones for convenience. This provides limited
    # broadcasting-like syntactic sugar, but does not implement general broadcasting support.
    src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
    dst_extent = list(dst_extent) if dst_extent else [1] * len(src_extent)

    # Align and broadcast extents from the right (tail) side.
    # This is majorly for supporting some syntactic sugar, not the whole broadcasting ability of copy op.
    src_extent, dst_extent = legalize_pairwise_extents(src_extent, dst_extent)

    # Use legalized extents for src and dst respectively.
    src = to_buffer_region(src, access_type="r", extents=src_extent)
    dst = to_buffer_region(dst, access_type="w", extents=dst_extent)
    return src, dst, src_extent, dst_extent


def _normalize_copy_regions(
    src: BufferLikeType, dst: BufferLikeType
) -> tuple[
    tirx.BufferRegion | tirx.BufferLoad | tirx.Buffer,
    tirx.BufferRegion | tirx.BufferLoad | tirx.Buffer,
]:
    src, dst, _, _ = _normalize_copy_regions_with_extents(src, dst)
    return src, dst


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


def copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    coalesced_width: int | None = None,
    disable_tma: bool = False,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    prefer_instruction: str | None = None,
    annotations: dict | None = None,
    loop_layout: Any | None = None,
    transpose: bool = False,
    l2_cache_ctrl: int | str | None = None,
    unit_flag_ctrl: int | tirx.PrimExpr | None = None,
    sub_blockid: int | tirx.PrimExpr | None = None,
    scale: BufferLikeType | None = None,
    pad_value: int | float | tirx.PrimExpr | None = None,
    data_select: bool = False,
) -> tirx.PrimExpr | tirx.Stmt:
    """Copy data between memory regions.

    Args:
        src (Union[tirx.Buffer, tirx.BufferLoad, tirx.BufferRegion]): Source memory region
        dst (Union[tirx.Buffer, tirx.BufferLoad, tirx.BufferRegion]): Destination memory region
        coalesced_width (Optional[int], keyword-only): Width for coalesced memory access. Defaults to None.
        disable_tma (bool, keyword-only): Whether to disable TMA acceleration. Defaults to False.
        eviction_policy (Optional[str], keyword-only): Cache eviction policy. Defaults to None.
        prefer_instruction (Optional[str], keyword-only): Backend-specific preferred lowering
            instruction category. For CUDA, recognized values include "tma", "cp_async", and
            "sync". For "tma", T.copy keeps synchronous copy semantics; global -> shared copies
            lower through TMA with an automatically allocated barrier and wait when constraints
            are satisfied.
        annotations (Optional[dict], keyword-only): Additional annotations dict. If provided,
            coalesced_width, disable_tma, eviction_policy, and prefer_instruction can also
            be specified here.
            Values in annotations take precedence over individual arguments.
        loop_layout (Optional[Fragment], keyword-only): A parallel loop layout hint for the SIMT copy
            (only valid for normal SIMT copy; incompatible with TMA/LDSM/STSM/TMem). When provided,
            it is attached to the outermost parallel loop generated by this copy.
        transpose (bool, keyword-only): Ascend GM-to-L1 transpose hint for dn2nz layout.
            Defaults to False.
        l2_cache_ctrl (Optional[int | str], keyword-only): Ascend L2 cache control
            policy for GM↔UB, GM↔L1, and UB↔GM DMA paths. Accepts an integer (hardware
            L2Ctrl value) or a case-insensitive string name.
            Common values: ``0`` / ``"normal_fv"``, ``4`` / ``"notalloc_keep"``
            (bypass L2 to stream past), ``5`` / ``"notalloc_clean"`` (bypass L2 to
            avoid pollution). Defaults to None; Ascend UB→GM store paths use
            default 4, while load paths use default 0.
            Ascend only; ignored on other backends.
        unit_flag_ctrl (Optional[int | PrimExpr], keyword-only): Ascend unit-flag
            control. ``None`` omits the annotation and lowers as 0.
        sub_blockid (Optional[int | PrimExpr], keyword-only): Ascend AIV sub-block
            selector. ``None`` omits the annotation and lowers as 0.
        scale (Optional[BufferLikeType], keyword-only): Ascend MX scale-factor source
            buffer (in L1/cbuf) for an L1→L0A/L0B copy. When provided, the copy
            additionally loads the per-block scale factors into the L0 MX scale
            registers via ``asc_copy_l12l0a_mx`` / ``asc_copy_l12l0b_mx`` so that a
            subsequent ``asc_mmad_mx`` applies the scaling. K-offset / K-step are
            auto-derived from the data slice (1 SF pair = 64 K-elements); the
            NZ stride is taken from the scale buffer's second-to-last dimension.
            Ascend L1→L0 only; ignored on other paths/backends. Defaults to None.
        pad_value (Optional[int | float | PrimExpr], keyword-only): Ascend GM→UB
            padding fill value. When set, an unaligned copy row is right-padded
            up to the next 32B boundary and the pad lanes are filled with this
            value. Emits a leading ``T.ascend_set_copy_pad_value(value)`` (so
            AutoSchedule syncs the pad-register write before the copy) and pads
            the copy via ``data_select``. The fill dtype is the destination
            element dtype. Opt-in: without it, an unaligned multi-row copy still
            errors as before. Ascend GM→UB (global→shared) only; ignored on
            other paths/backends. Defaults to None.
        data_select (bool, keyword-only): Ascend GM→UB. Same right-pad behavior
            as ``pad_value`` (dataSelect=1 + rightPadding to the 32B boundary),
            but the fill value is NOT set by this copy — it uses whatever the
            hardware pad register currently holds, which the caller must have
            set via ``T.ascend_set_copy_pad_value(...)`` beforehand. Use this to
            reuse one pad value across several copies without re-setting it.
            Mutually exclusive with ``pad_value`` (passing both raises
            ValueError). Ascend GM→UB only. Defaults to False.

    Raises:
        TypeError: If copy extents cannot be deduced from arguments

    Returns:
        tirx.Call: A handle to the copy operation

    Range handling notes:
    - Accepts `Buffer`/`BufferRegion`/`BufferLoad` on either side. Extents are
      derived as follows: `Buffer -> shape`, `BufferRegion -> [r.extent]`,
      `BufferLoad -> extents from its inferred/encoded region`.
    - Normally, we require the extents of both sides to be the same. If they
      differ, the copy instruction follows an internal rule to select one side
      as the base range and create iteration space. This may generate unexpected
      code. And if some dimensions are 1, unexpected errors may happen.
    - Small Optimization: If both `src` and `dst` are scalar `BufferLoad` without
      region extents, lowers to a direct store: `dst[...] = src[...]`.
    - Syntactic Sugar: TileLang supports passing the head address of a buffer to represent
      the whole buffer if there are no ambiguity. For example, T.copy(A, A_shared[i, j]).
      To support this, we need some special shape checking. But remember currently we don't
      support something like "broadcast".
    - The finalized extents are encoded with `tl.region` via `to_buffer_region`
      and passed through to the backend; low-level loop construction and any
      scope-specific decisions happen during lowering.
    - On Ascend, a UB-to-UB copy from a dense source into a destination annotated
      with ``make_ascend_compact_nz_layout`` lowers to the ND-to-NZ scatter. The
      destination allocation must reserve one padding row, and the copied region
      must exclude that row (for example, ``T.copy(src, dst[:rows, :])``).
    """
    dst_orig = dst
    src, dst, src_extent, dst_extent = _normalize_copy_regions_with_extents(src, dst)

    # Build annotations dict before selecting the scalar fast path: a scalar
    # copy with metadata must remain a tile op so the metadata is preserved.
    ann = _normalize_annotations(annotations)

    # Individual arguments take lower precedence than annotations
    if "coalesced_width" not in ann and coalesced_width is not None:
        ann["coalesced_width"] = coalesced_width
    if "disable_tma" not in ann and disable_tma:
        ann["disable_tma"] = disable_tma
    if "eviction_policy" not in ann and eviction_policy is not None:
        eviction_policy_map = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}
        ann["eviction_policy"] = eviction_policy_map[eviction_policy]
    if "prefer_instruction" not in ann and prefer_instruction is not None:
        ann["prefer_instruction"] = tirx.StringImm(prefer_instruction)

    # Parallel loop layout hint (Fragment). Mirrors T.Parallel(loop_layout=...)
    if loop_layout is not None and "parallel_loop_layout" not in ann:
        ann["parallel_loop_layout"] = loop_layout

    # Ascend GM→L1: use dn2nz (transpose N/D mapping) instead of nd2nz
    if transpose and "transpose" not in ann:
        ann["transpose"] = tirx.IntImm("int32", 1)

    # Ascend DMA: L2 cache control for GM↔L1 and UB↔GM paths.
    if l2_cache_ctrl is not None and "l2_cache_ctrl" not in ann:
        ann["l2_cache_ctrl"] = l2_cache_ctrl
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
        # reuses the register via data_select. dst_orig is a
        # Buffer/BufferRegion/BufferLoad; all but a raw Buffer expose .buffer.
        if not isinstance(dst_orig, (tirx.Buffer, tirx.BufferRegion, tirx.BufferLoad)):
            raise TypeError(
                "T.copy: pad_value requires dst to be a Buffer, BufferRegion, or "
                f"BufferLoad so the pad fill dtype can be derived, got {type(dst_orig)}."
            )
        dst_buf = dst_orig.buffer if isinstance(dst_orig, (tirx.BufferRegion, tirx.BufferLoad)) else dst_orig
        tb_tir.evaluate(ascend_set_copy_pad_value(pad_value, dtype=str(dst_buf.dtype)))
        ann["data_select"] = tirx.IntImm("int32", 1)
    if data_select and "data_select" not in ann:
        ann["data_select"] = tirx.IntImm("int32", 1)
    if "l2_cache_ctrl" in ann:
        ann["l2_cache_ctrl"] = _normalize_l2_cache_ctrl(ann["l2_cache_ctrl"])

    # Ascend MX scale-factor companion load (L1→L0A/L0B). Pass the scale source
    # as a third positional region so the backend can derive the scale L1 pointer
    # and emit asc_copy_l12l0a_mx / asc_copy_l12l0b_mx alongside the data load.
    if scale is not None:
        scale_extent = get_extent(scale)
        scale_region = to_buffer_region(scale, access_type="r", extents=scale_extent)
        return tirx.call_intrin(
            "handle",
            tirx.op.Op.get("tl.tileop.copy"),
            src,
            dst,
            scale_region,
            annotations=ann if ann else None,
        )

    if isinstance(src, tirx.BufferLoad) and isinstance(dst, tirx.BufferLoad) and not ann:
        # Scalar fast path. Mirror the dtype conversion the region path applies
        # in copy.cc; cast to the load dtype, which is what BufferStore checks
        # once index lanes are folded in.
        value = src
        if src.dtype != dst.dtype:
            value = tirx.Cast(dst.dtype, src)
        return tirx.BufferStore(dst.buffer, value, dst.indices)

    return tirx.call_intrin("handle", tirx.op.Op.get("tl.tileop.copy"), src, dst, annotations=ann if ann else None)


def copy_cluster(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    dst_block: int | tirx.PrimExpr | None = None,
    cluster_mask: int | None = None,
    remote_barrier: tirx.BufferLoad | None = None,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    coalesced_width: int | None = None,
    annotations: dict | None = None,
    loop_layout: Any | None = None,
) -> tirx.PrimExpr | tirx.Stmt:
    """Cluster-aware copy for TMA multicast or SM-to-SM shared-memory copy.

    Args:
        src: Source memory region.
        dst: Destination memory region.
        dst_block: Destination CTA rank in the cluster for SM-to-SM copy.
        cluster_mask: Bitmask of CTAs that participate in TMA multicast.
        remote_barrier: Shared-memory mbarrier for asynchronous SM-to-SM copy
            completion signalling.  The destination CTA should wait on its
            local copy of this barrier.
        eviction_policy: Cache eviction hint passed to the TMA instruction.
            Only relevant for the TMA multicast path (``cluster_mask`` set).
        coalesced_width: Vectorization width (in elements) for the SIMT loop
            used on the SM-to-SM fallback path (``dst_block`` set, no fast
            bulk-async route available).
        annotations: Additional annotations dict. Values in annotations take
            precedence over individual arguments.
        loop_layout: Parallel loop layout hint (Fragment) for the SIMT loop on
            the SM-to-SM fallback path. Incompatible with the TMA multicast
            path (``cluster_mask`` set).

    Returns:
        tirx.Call: A handle to the copy operation.
    """
    src, dst = _normalize_copy_regions(src, dst)

    ann = _normalize_annotations(annotations)
    if "dst_block" not in ann and dst_block is not None:
        ann["dst_block"] = dst_block
    if "cluster_mask" not in ann and cluster_mask is not None:
        ann["cluster_mask"] = cluster_mask
    if "barrier" not in ann and remote_barrier is not None:
        ann["barrier"] = remote_barrier
    if "eviction_policy" not in ann and eviction_policy is not None:
        eviction_policy_map = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}
        ann["eviction_policy"] = eviction_policy_map[eviction_policy]
    if "coalesced_width" not in ann and coalesced_width is not None:
        ann["coalesced_width"] = coalesced_width
    if loop_layout is not None and "parallel_loop_layout" not in ann:
        ann["parallel_loop_layout"] = loop_layout

    return tirx.call_intrin("handle", tirx.op.Op.get("tl.tileop.copy"), src, dst, annotations=ann)


def async_copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    coalesced_width: int | None = None,
    annotations: dict | None = None,
    loop_layout: Any | None = None,
) -> tirx.PrimExpr | tirx.Stmt:
    """Asynchronous copy primitive lowered through cp.async.

    This operator is intended for explicitly asynchronous global->shared copy.
    The backend enforces cp.async constraints and emits:
      `ptx_cp_async(...)` + `ptx_commit_group()`.
    No wait is auto-inserted for `T.async_copy`; synchronization is explicit.

    Args:
        src (Union[tirx.Buffer, tirx.BufferLoad, tirx.BufferRegion]): Source memory region
        dst (Union[tirx.Buffer, tirx.BufferLoad, tirx.BufferRegion]): Destination memory region
        coalesced_width (Optional[int], keyword-only): Width for coalesced memory access. Defaults to None.
        annotations (Optional[dict], keyword-only): Additional annotations dict.
        loop_layout (Optional[Fragment], keyword-only): A parallel loop layout hint for the SIMT copy loop.

    Returns:
        tirx.Call: A handle to the async copy operation
    """
    src, dst = _normalize_copy_regions(src, dst)
    ann = _normalize_annotations(annotations)
    if "coalesced_width" not in ann and coalesced_width is not None:
        ann["coalesced_width"] = coalesced_width
    if loop_layout is not None and "parallel_loop_layout" not in ann:
        ann["parallel_loop_layout"] = loop_layout

    if isinstance(src, tirx.BufferLoad) and isinstance(dst, tirx.BufferLoad) and not ann:
        return tirx.BufferStore(dst.buffer, src, dst.indices)

    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.async_copy"),
        src,
        dst,
        annotations=ann,
    )


def tma_copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    barrier=None,
    cluster_mask: int | None = None,
    leader_scope_threads: int | None = None,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    annotations: dict | None = None,
) -> tirx.PrimExpr | tirx.Stmt:
    """TMA copy with user-managed synchronization.

    For **loads** (global -> shared): issues expect_tx + tma_load (no wait).
    Unlike T.copy() which emits a full synchronous TMA sequence (arrive + load + wait),
    T.tma_copy() emits only the producer part (expect_tx + tma_load).
    The user manages synchronization explicitly via T.barrier_arrive() and
    T.mbarrier_wait_parity(). ``barrier`` is required for loads.

    ``cluster_mask`` turns a load into a TMA **multicast** within the thread-block
    cluster while keeping that same split-phase contract, unlike T.copy_cluster()
    whose multicast path also emits the wait. The lowest-ranked CTA in the mask
    issues the multicast and the other in-mask CTAs issue nothing; a CTA *outside*
    the mask falls back to its own unicast load. Every CTA runs its own expect_tx
    against its local copy of ``barrier``, so each one waits on its own arrival.

    For **stores** (shared -> global): issues tma_store + tma_store_arrive (no wait).
    Unlike T.copy() which emits tma_store + tma_store_arrive + tma_store_wait,
    T.tma_copy() omits the wait so the user can batch multiple stores before
    calling T.tma_store_wait() explicitly. ``barrier`` is not needed for stores.
    FP4 unpacked shared-memory storage is load-only for TMA: packed global
    ``float4_e2m1fn`` may be loaded into unpacked shared
    ``float4_e2m1_unpacked``, but the reverse TMA store is not supported.

    Args:
        src: Source memory region (global or shared)
        dst: Destination memory region (shared or global)
        barrier: Mbarrier (from T.alloc_barrier()) for TMA load synchronization.
            Required for loads (global -> shared). Not needed for stores.
            The TMA load will arrive at this barrier with expected byte count.
            The user must wait on the same barrier via T.mbarrier_wait_parity().
        cluster_mask: Bitmask of the CTAs in the thread-block cluster that receive
            a multicast load, e.g. ``0b11`` for the first two ranks. Loads only;
            ``None`` issues an ordinary unicast load.
        leader_scope_threads: Number of threads in each TMA leader-election scope
            (e.g., 32 for per-warp). Defaults to the thread extend in the current context if not specified.
        eviction_policy: Cache eviction policy. Defaults to None.
        annotations: Additional annotations dict. Values in annotations take
            precedence over individual arguments.

    Returns:
        tirx.Call: A handle to the tma_copy operation
    """
    # If both side are buffers, we should make sure their shapes are equal
    if isinstance(src, tirx.Buffer) and isinstance(dst, tirx.Buffer):
        ir.assert_structural_equal(src.shape, dst.shape)

    src_extent = get_extent(src)
    dst_extent = get_extent(dst)

    assert src_extent or dst_extent, "Can't deduce copy extents from args. Both src and dst miss extents info."
    src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
    dst_extent = list(dst_extent) if dst_extent else [1] * len(src_extent)

    src_extent, dst_extent = legalize_pairwise_extents(src_extent, dst_extent)

    src = to_buffer_region(src, access_type="r", extents=src_extent)
    dst = to_buffer_region(dst, access_type="w", extents=dst_extent)

    ann = _normalize_annotations(annotations)

    if barrier is not None:
        from .builtin import _mbar_to_buffer_load

        ann["barrier"] = _mbar_to_buffer_load(barrier)

    if cluster_mask is not None:
        if not isinstance(cluster_mask, int) or cluster_mask <= 0:
            raise ValueError(f"cluster_mask must be a positive int bitmask, got {cluster_mask}")
        if "cluster_mask" not in ann:
            ann["cluster_mask"] = cluster_mask

    if leader_scope_threads is not None:
        if not isinstance(leader_scope_threads, int) or leader_scope_threads <= 0:
            raise ValueError(f"leader_scope_threads must be a positive int, got {leader_scope_threads}")
        if leader_scope_threads % 32 != 0:
            raise ValueError(f"leader_scope_threads must be a multiple of warp size (32), got {leader_scope_threads}")
        if "leader_scope_threads" not in ann:
            ann["leader_scope_threads"] = leader_scope_threads

    if "eviction_policy" not in ann and eviction_policy is not None:
        eviction_policy_map = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}
        ann["eviction_policy"] = eviction_policy_map[eviction_policy]

    return tirx.call_intrin("handle", tirx.op.Op.get("tl.tileop.tma_copy"), src, dst, annotations=ann)


_TMA_SUPPORTED_DTYPES = frozenset(
    {
        "uint8",
        "uint16",
        "uint32",
        "int32",
        "uint64",
        "int64",
        "float16",
        "float32",
        "float64",
        "bfloat16",
    }
)


def tma_gather4(
    src: tirx.Buffer,
    dst: tirx.Buffer,
    col: tirx.PrimExpr,
    rows,
    *,
    barrier,
    swizzle=None,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    annotations: dict | None = None,
):
    """Issue a TMA tile::gather4 load (sm_100a, Blackwell).

    Loads four arbitrary rows of a 2D global tensor ``src`` into a 2D shared
    tile ``dst`` of shape ``(4, K_box)``. The CUtensorMap descriptor (dtype +
    swizzle) is built by the compiler from buffer + layout info.

    Caller must wrap this with ``T.shuffle_elect`` and pair it with
    ``T.mbarrier_expect_tx`` (use :func:`tma_gather4_bytes`) before, and
    ``barrier_arrive`` / ``mbarrier_wait_parity`` after.

    The ``swizzle`` kwarg is deprecated; mark the shared tile via
    ``T.annotate_layout`` for non-default swizzle.

    ``annotations`` is an additional annotations dict, merged before the
    internal gather4 encoding keys.
    """
    if not isinstance(src, tirx.Buffer):
        raise TypeError("tma_gather4 src must be a tirx.Buffer (global)")
    if not isinstance(dst, tirx.Buffer):
        raise TypeError("tma_gather4 dst must be a tirx.Buffer (shared)")
    if src.scope() != "global":
        raise ValueError(f"tma_gather4 src must be a global buffer, got scope={src.scope()}")
    if dst.scope() not in ("shared", "shared.dyn"):
        raise ValueError(f"tma_gather4 dst must be a shared buffer, got scope={dst.scope()}")
    if len(src.shape) != 2:
        raise ValueError(f"tma_gather4 expects rank-2 global buffer, got {len(src.shape)}")
    if len(dst.shape) != 2:
        raise ValueError(f"tma_gather4 expects rank-2 shared buffer (4 x K_box), got {len(dst.shape)}")
    if src.dtype != dst.dtype:
        raise ValueError(f"tma_gather4 dtype mismatch: src={src.dtype}, dst={dst.dtype}")
    if not (isinstance(dst.shape[0], int) and dst.shape[0] == 4) and not (hasattr(dst.shape[0], "value") and int(dst.shape[0].value) == 4):
        raise ValueError(f"tma_gather4 shared tile leading dim must be 4, got {dst.shape[0]}")
    if src.strides:
        inner = src.strides[1]
        if not ((isinstance(inner, int) and inner == 1) or (hasattr(inner, "value") and int(inner.value) == 1)):
            raise ValueError(f"tma_gather4 requires unit innermost global stride, got {inner}")
    rows = list(rows)
    if len(rows) != 4:
        raise ValueError(f"tma_gather4 expects exactly 4 row indices, got {len(rows)}")
    if swizzle not in (None, "none", 0):
        import warnings

        warnings.warn(
            f"tma_gather4 swizzle={swizzle!r} is deprecated; use T.annotate_layout.",
            DeprecationWarning,
            stacklevel=2,
        )

    from .builtin import _mbar_to_buffer_load

    bar_load = _mbar_to_buffer_load(barrier)

    eviction_policy_map = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}
    ep = 0 if eviction_policy is None else eviction_policy_map[eviction_policy]

    # Matching (4, K_box) extents satisfy CopyNode's shape check; the actual
    # access pattern lives in the gather4_rows / gather4_col annotations.
    K_box = dst.shape[1]
    src_region = to_buffer_region(src, access_type="r", extents=[4, K_box])
    dst_region = to_buffer_region(dst, access_type="w", extents=[4, K_box])

    ann = _normalize_annotations(annotations)
    ann.update(
        {
            "is_gather4": True,
            "gather4_rows": rows,
            "gather4_col": col,
            "barrier": bar_load,
            "eviction_policy": ep,
        }
    )
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.copy"),
        src_region,
        dst_region,
        annotations=ann,
    )


def tma_gather4_bytes(K_box, dtype: str) -> int:
    """Transaction byte count for a 4-row gather4 of width ``K_box``. Pass
    to ``T.mbarrier_expect_tx`` immediately before ``T.tma_gather4``.
    """
    if dtype not in _TMA_SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    dt = tvm.DataType(dtype)
    if dt.is_float4_e2m1_unpacked():
        elem_bits = 4
    else:
        elem_bits = dt.bits
    return (4 * K_box * elem_bits + 7) // 8


def tma_scatter4(
    src: tirx.Buffer,
    dst: tirx.Buffer,
    col: tirx.PrimExpr,
    rows,
    *,
    swizzle=None,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    annotations: dict | None = None,
):
    """Issue a TMA tile::scatter4 store (sm_100a, Blackwell).

    Stores a 2D shared tile of shape ``(4, K_box)`` to four arbitrary rows of
    a 2D global tensor ``dst``. Caller is responsible for ``tma_store_arrive``
    / ``tma_store_wait`` and the ``T.shuffle_elect`` guard. See
    :func:`tma_gather4` for descriptor / swizzle inference details.

    ``annotations`` is an additional annotations dict, merged before the
    internal scatter4 encoding keys.
    """
    if not isinstance(src, tirx.Buffer):
        raise TypeError("tma_scatter4 src must be a tirx.Buffer (shared)")
    if not isinstance(dst, tirx.Buffer):
        raise TypeError("tma_scatter4 dst must be a tirx.Buffer (global)")
    if src.scope() not in ("shared", "shared.dyn"):
        raise ValueError(f"tma_scatter4 src must be a shared buffer, got scope={src.scope()}")
    if dst.scope() != "global":
        raise ValueError(f"tma_scatter4 dst must be a global buffer, got scope={dst.scope()}")
    if len(src.shape) != 2:
        raise ValueError(f"tma_scatter4 expects rank-2 shared buffer (4 x K_box), got {len(src.shape)}")
    if len(dst.shape) != 2:
        raise ValueError(f"tma_scatter4 expects rank-2 global buffer, got {len(dst.shape)}")
    if src.dtype != dst.dtype:
        raise ValueError(f"tma_scatter4 dtype mismatch: src={src.dtype}, dst={dst.dtype}")
    if not (isinstance(src.shape[0], int) and src.shape[0] == 4) and not (hasattr(src.shape[0], "value") and int(src.shape[0].value) == 4):
        raise ValueError(f"tma_scatter4 shared tile leading dim must be 4, got {src.shape[0]}")
    if dst.strides:
        inner = dst.strides[1]
        if not ((isinstance(inner, int) and inner == 1) or (hasattr(inner, "value") and int(inner.value) == 1)):
            raise ValueError(f"tma_scatter4 requires unit innermost global stride, got {inner}")
    rows = list(rows)
    if len(rows) != 4:
        raise ValueError(f"tma_scatter4 expects exactly 4 row indices, got {len(rows)}")
    if swizzle not in (None, "none", 0):
        import warnings

        warnings.warn(
            f"tma_scatter4 swizzle={swizzle!r} is deprecated; use T.annotate_layout.",
            DeprecationWarning,
            stacklevel=2,
        )

    eviction_policy_map = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}
    ep = 0 if eviction_policy is None else eviction_policy_map[eviction_policy]

    K_box = src.shape[1]
    src_region = to_buffer_region(src, access_type="r", extents=[4, K_box])
    dst_region = to_buffer_region(dst, access_type="w", extents=[4, K_box])

    ann = _normalize_annotations(annotations)
    ann.update(
        {
            "is_scatter4": True,
            "gather4_rows": rows,
            "gather4_col": col,
            "eviction_policy": ep,
        }
    )
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.copy"),
        src_region,
        dst_region,
        annotations=ann,
    )


def transpose(
    src: BufferLikeType,
    dst: BufferLikeType,
    annotations: dict | None = None,
) -> tirx.PrimExpr:
    """Transpose a 2D buffer in shared memory: dst[j, i] = src[i, j].

    Both src and dst should be shared memory buffers.
    If src has shape (M, N), dst should have shape (N, M).

    Args:
        src: Source buffer or region of shape (..., M, N).
        dst: Destination buffer or region of shape (..., N, M).
        annotations: Optional annotations to attach to the call.

    Returns:
        tirx.Call: A handle to the transpose operation.
    """
    src_extent = get_extent(src)
    dst_extent = get_extent(dst)

    assert src_extent is not None, "Cannot deduce extent for transpose src."
    assert dst_extent is not None, "Cannot deduce extent for transpose dst."
    assert len(src_extent) >= 2, "Transpose requires at least 2D buffers."
    assert len(dst_extent) >= 2, "Transpose requires at least 2D buffers."

    src_region = to_buffer_region(src)
    dst_region = to_buffer_region(dst)
    src = buffer_region_to_tile_region(src_region, "r", list(src_extent))
    dst = buffer_region_to_tile_region(dst_region, "w", list(dst_extent))

    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.transpose"),
        src,
        dst,
        annotations=_normalize_annotations(annotations),
    )


def im2col(
    img: BufferLikeType,
    col: BufferLikeType,
    nhw_step: tirx.PrimExpr,
    c_step: tirx.PrimExpr,
    kernel: int,
    stride: int,
    dilation: int,
    pad: int,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    annotations: dict | None = None,
) -> tirx.PrimExpr:
    """Perform im2col transformation for 2D convolution.

    Args:
        img (tirx.Buffer): Input image buffer
        col (tirx.Buffer): Output column buffer
        nhw_step (tirx.PrimExpr): Step size for batch and spatial dimensions
        c_step (tirx.PrimExpr): Step size for channel dimension
        kernel (int): Kernel size
        stride (int): Stride of the convolution
        dilation (int): Dilation rate
        pad (int): Padding size
        annotations: Optional annotations to attach to the call

    Returns:
        tirx.Call: A handle to the im2col operation
    """
    if eviction_policy is None:
        eviction_policy = 0
    else:
        eviction_policy = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}[eviction_policy]
    img_region = to_buffer_region(img)
    col_region = to_buffer_region(col)
    img_extents = [r.extent for r in img_region.region]
    col_extents = [r.extent for r in col_region.region]
    img_region = buffer_region_to_tile_region(img_region, "r", img_extents)
    col_region = buffer_region_to_tile_region(col_region, "w", col_extents)
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.im2col"),
        img_region,
        col_region,
        nhw_step,
        c_step,
        kernel,
        stride,
        dilation,
        pad,
        eviction_policy,
        annotations=_normalize_annotations(annotations),
    )


@deprecated("T.c2d_im2col", "T.im2col", "0.14.0")
def c2d_im2col(
    img: BufferLikeType,
    col: BufferLikeType,
    nhw_step: tirx.PrimExpr,
    c_step: tirx.PrimExpr,
    kernel: int,
    stride: int,
    dilation: int,
    pad: int,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    annotations: dict | None = None,
) -> tirx.PrimExpr:
    """Deprecated alias for :func:`im2col`.

    Deprecated:
        Use :func:`im2col` instead. This alias is scheduled for removal in
        TileLang 0.14.0.
    """
    return im2col(
        img,
        col,
        nhw_step,
        c_step,
        kernel,
        stride,
        dilation,
        pad,
        eviction_policy,
        annotations=annotations,
    )

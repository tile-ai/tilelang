"""SemanticIR -> TileIR lowering: tile-op handlers + dispatch.

Provides the ``@tile_op_impl`` handlers (copy, tma_copy, fill, gemm, reduce,
cumsum, barriers, storage_sync, allreduce, break/continue, device_assert,
debug_print), the ``tile_op`` dispatcher (``_lower_tile_op``), and the
region/static-int extraction helpers shared with the tile-level and atomic
modules.  Imports the shared foundation, scalar ``lower_expr``, and the parallel
region/partition helpers.
"""

from __future__ import annotations

import functools
import operator
from typing import Any

from tvm import tirx as _tirx

from tilelang.tileir.ir.builder import IRBuilder
from tilelang.tileir.errors import (
    TileIRLoweringError,
    TileIRLoweringNotImplementedError,
    _UnsupportedTileIRNode,
)
from tilelang.tileir.ir.types import MemSpace
from tilelang.tileir.ir.ops import (
    Barrier,
    Break,
    Cast,
    Continue,
    Copy,
    CopyGather,
    CopyScatter,
    Cumsum,
    DebugPrint,
    DeviceAssert,
    Fill,
    Gemm,
    GemmScaled,
    GridSync,
    Reduce,
    Tcgen05Gemm,
    ThreadAllreduce,
    TmaCopy,
    TransposeCopy,
)
from tilelang.tileir.semantic import SemanticStmt

from ._base import (
    TILE_OP_IMPL,
    LoweringScope,
    impl,
    tile_op_impl,
    _is_unsigned_dtype,
    _parse_bool,
    _tir_dtype_to_tile_type,
)
from .expr import lower_expr
from .parallel import (
    _compute_view_indices,
    _compute_tma_view_indices,
    _extract_tir_region_indices,
    _prove_region_in_bounds,
    _region_tile_shape,
)


def _require_region_buffers(
    stmt: SemanticStmt,
    scope: LoweringScope,
    *,
    op_name: str,
    roles: tuple[str, ...],
) -> tuple[Any, ...]:
    """Validate an op's exact region contract and resolve every buffer.

    A malformed ``SemanticStmt`` indicates a broken semantic-to-lowering
    boundary.  Dropping that statement would silently remove a computation,
    so all core tile-op handlers share this fail-loud path.
    """
    expected = len(roles)
    actual = len(stmt.regions)
    if actual != expected:
        raise TileIRLoweringError(f"sem_to_ir: {op_name} requires exactly {expected} regions; got {actual} regions.")

    values: list[Any] = []
    for index, (role, region) in enumerate(zip(roles, stmt.regions)):
        try:
            values.append(scope.lookup_buffer(region.buffer))
        except KeyError as exc:
            raise TileIRLoweringError(f"sem_to_ir: {op_name} region {index} ({role}) references unknown buffer {region.buffer!r}.") from exc
    return tuple(values)


def _require_full_buffer_region(region: Any, buf_val: Any, *, op_name: str, side: str, rank: int = 2) -> None:
    """Validate that a semantic region covers its buffer's FULL declared
    extent, statically, at offset 0.

    Shared authoritative guard for both ``tl.tileop.transpose`` (rank-2 src
    and dst) and the ``tl.tileop.copy`` gather4/scatter4 path (the rank-2
    SHARED-side ``(4, K_box)`` tile region) -- see the call sites
    (``_lower_transpose`` / ``_lower_gather4_scatter4``) for why each op only
    supports a whole-buffer region on that side. ``rank`` is parameterized so
    the validation loop is independent of region rank.

    Representation (see ``_semantic_region`` in ``tilelang/tileir/semantic.py``):
      - ``region.shape[i]`` is a Python ``int`` ONLY when the semantic
        layer's ``_shape()`` helper proved the dim came from a TIR
        ``IntImm`` (i.e. it is statically known); any other expression
        (e.g. ``seq_len - 0``) is stringified instead. So
        ``isinstance(region.shape[i], int)`` *is* the static/dynamic test --
        no separate static-int extraction helper is needed or possible here
        (there is no raw ``PrimExpr`` to extract from; ``_shape()`` already
        discarded it).
      - ``region.indices[i]`` is always a ``str`` -- ``_expr_text`` stringifies
        every index, including ``IntImm`` literals, via
        ``str(int(expr))``. A statically-zero offset therefore always
        stringifies to exactly ``"0"``; anything else (a symbolic offset, or
        one that merely *evaluates* to zero without being a literal) is
        rejected rather than guessed at, per the "parse conservatively"
        directive -- we never assume an offset is zero.

    Raises ``TileIRLoweringNotImplementedError`` naming the violation kind
    ("rank-N buffer", "partial region", or "dynamic-extent region") on any
    violation; never silently degrades.
    """
    buf_shape = tuple(int(d) for d in buf_val.type.shape)
    _suffix = f"{op_name} currently supports full-buffer rank-{rank} only."

    if len(buf_shape) != rank:
        raise TileIRLoweringNotImplementedError(
            f"{op_name}: {side} buffer '{region.buffer}' is rank-{len(buf_shape)} (shape {list(buf_shape)}); {_suffix}"
        )
    if len(region.shape) != rank or len(region.indices) != rank:
        raise TileIRLoweringNotImplementedError(
            f"{op_name}: {side} region on buffer '{region.buffer}' is rank-"
            f"{len(region.shape)} (shape {list(region.shape)}), not rank-{rank}; {_suffix}"
        )

    for dim, (offset, extent, buf_dim) in enumerate(zip(region.indices, region.shape, buf_shape)):
        if offset != "0":
            raise TileIRLoweringNotImplementedError(
                f"{op_name}: {side} buffer '{region.buffer}' dim {dim} has a "
                f"non-zero or unprovable-static offset ({offset!r}) — partial region is "
                f"not supported; {_suffix}"
            )
        if not isinstance(extent, int):
            raise TileIRLoweringNotImplementedError(
                f"{op_name}: {side} buffer '{region.buffer}' dim {dim} has a "
                f"dynamic-extent region ({extent!r}); cannot statically prove it covers "
                f"the full buffer dim ({buf_dim}); {_suffix}"
            )
        if extent != buf_dim:
            raise TileIRLoweringNotImplementedError(
                f"{op_name}: {side} buffer '{region.buffer}' dim {dim} region "
                f"extent ({extent}) does not match the declared buffer dim ({buf_dim}) — "
                f"partial region is not supported; {_suffix}"
            )


def _lower_gather4_index_scalar(expr: Any, scope: LoweringScope, builder: IRBuilder) -> Any:
    """Lower a ``T.tma_gather4``/``T.tma_scatter4`` row/col index ``PrimExpr``
    to an int32 scalar ``Value``, inserting a ``Cast`` if the expression's
    natural dtype isn't already int32 (e.g. a dynamic index computed in
    int64). ``gather4_rows``/``gather4_col`` annotation entries may surface
    as raw Python ``int``/``bool`` (not yet wrapped in a TIR node) depending
    on how the annotations ``Map`` round-tripped through the FFI, so those
    are coerced to an int32 ``IntImm`` first."""
    if isinstance(expr, bool):
        expr = _tirx.IntImm("int32", int(expr))
    elif isinstance(expr, int):
        expr = _tirx.IntImm("int32", expr)
    val = lower_expr(expr, scope, builder)
    if val.type.dtype.name == "int32":
        return val
    target_ty = _tir_dtype_to_tile_type("int32")
    cast_op = builder.create(
        Cast(src=val, dtype="int32", src_dtype=val.type.dtype.name),
        result_types=(target_ty,),
    )
    return cast_op.results[0]


def _lower_gather4_scatter4(stmt: SemanticStmt, attrs: dict, scope: LoweringScope, builder: IRBuilder, *, is_gather: bool) -> None:
    """Lower a 2-region ``tl.tileop.copy`` annotated ``is_gather4`` /
    ``is_scatter4`` (``T.tma_gather4`` / ``T.tma_scatter4``) to ``CopyGather``
    / ``CopyScatter``.

    Unlike a general index-buffer gather/scatter copy (which would carry row
    indices via a 3rd index-BUFFER region), gather4/scatter4 carry exactly
    4 row indices plus a column offset as
    scalar ``PrimExpr``s in the TIR call's ``annotations`` dict
    (``gather4_rows`` / ``gather4_col``). The semantic layer's generic
    ``_annotation_attrs`` stringifies these into ``attrs`` (used only for the
    routing check in ``_lower_copy`` below), so the actual ``PrimExpr``
    values are read directly from ``stmt.call_annotations`` here.

    ``barrier`` (gather-only) and ``eviction_policy`` are present in the
    annotations but intentionally accepted-and-IGNORED: TileIR's
    gather_scatter_view lowering is scheduled entirely by the compiler (the
    token-order pass sequences the load/store automatically), so the
    user-managed mbarrier / cache-eviction hints the real hardware
    ``tile::gather4`` / ``tile::scatter4`` TMA instructions need on the CUDA
    backend have no analog here.

    A missing buffer here is not warn-skipped (unlike the plain 2-region
    copy path below): silently dropping a gather/scatter copy would
    silently drop rows of data instead of failing loudly (the same
    "no silent degrade" contract as the blockscaled-gemm buffer lookups).

    Region shapes: the frontend (``copy_op.py``) builds BOTH regions with the
    same dummy ``(4, K_box)`` extents purely to satisfy ``CopyNode``'s arg-
    shape check -- the actual access pattern lives entirely in the
    ``gather4_rows``/``gather4_col`` annotations, so the GLOBAL-side region
    (``src`` for gather, ``dst`` for scatter) does not represent that
    buffer's real declared shape and is not checked against it. Only the
    SHARED-side region (``dst`` for gather, ``src`` for scatter) -- the
    literal ``(4, K_box)`` tile -- is required to be its buffer's whole
    extent, statically, via ``_require_full_buffer_region`` (the same guard
    ``_lower_transpose`` uses): a sliced SHARED tile would silently
    read/write the wrong offset.

    Memspace: a gather's ``src`` (the side viewed through the
    ``gather_scatter_view``) must be GLOBAL and its ``dst`` SHARED; a
    scatter's ``src`` must be SHARED and its ``dst`` GLOBAL --
    ``_make_gather_scatter_view`` requires a ``TensorView``, which only
    GLOBAL buffers have. The frontend (``copy_op.py``'s ``tma_gather4`` /
    ``tma_scatter4``) already enforces this at trace time; it is re-asserted
    here with a clear message since a hand-built ``SemanticStmt`` could
    bypass the frontend.

    Capability note: general index-BUFFER gather (an arbitrary-length
    runtime index buffer, as opposed to 4 literal/dynamic scalar rows) awaits
    a universal TileLang API -- this handler only recognizes the fixed-4-row
    ``T.tma_gather4`` / ``T.tma_scatter4`` surface.
    """
    op_name = "tma_gather4" if is_gather else "tma_scatter4"

    if len(stmt.regions) != 2:
        raise TileIRLoweringError(f"sem_to_ir: {op_name} requires 2 regions (src, dst); got {len(stmt.regions)}.")
    src_region, dst_region = stmt.regions[0], stmt.regions[1]
    try:
        src_val = scope.lookup_buffer(src_region.buffer)
        dst_val = scope.lookup_buffer(dst_region.buffer)
    except KeyError as exc:
        raise TileIRLoweringError(f"sem_to_ir: {op_name} buffer lookup failed: {exc}") from exc

    # Only the SHARED-side (4, K_box) tile region is authoritative -- see the
    # docstring above for why the GLOBAL-side region is a dummy shape.
    shared_region, shared_val, shared_side = (dst_region, dst_val, "dst") if is_gather else (src_region, src_val, "src")
    _require_full_buffer_region(shared_region, shared_val, op_name=op_name, side=shared_side, rank=2)

    global_val = src_val if is_gather else dst_val
    global_side = "src" if is_gather else "dst"
    if len(global_val.type.shape) != 2:
        raise TileIRLoweringError(
            f"{op_name}: {global_side} buffer is rank-{len(global_val.type.shape)} "
            f"(shape {list(global_val.type.shape)}); {op_name} requires a rank-2 GLOBAL buffer."
        )

    if is_gather:
        if src_val.type.space != MemSpace.GLOBAL:
            raise TileIRLoweringError(f"{op_name} src must be a GLOBAL buffer; got {src_val.type.space.value}.")
        if dst_val.type.space != MemSpace.SHARED:
            raise TileIRLoweringError(f"{op_name} dst must be a SHARED buffer; got {dst_val.type.space.value}.")
    else:
        if src_val.type.space != MemSpace.SHARED:
            raise TileIRLoweringError(f"{op_name} src must be a SHARED buffer; got {src_val.type.space.value}.")
        if dst_val.type.space != MemSpace.GLOBAL:
            raise TileIRLoweringError(f"{op_name} dst must be a GLOBAL buffer; got {dst_val.type.space.value}.")

    if src_val.type.dtype != dst_val.type.dtype:
        raise TileIRLoweringError(f"{op_name}: src dtype ({src_val.type.dtype.name}) must match dst dtype ({dst_val.type.dtype.name}).")

    # K_box (the row width) must be a power of two -- a CUDA Tile IR
    # gather_scatter_view dialect constraint. The leading dim is always 4
    # (already a power of two; enforced by the frontend), so only K_box needs
    # checking here.
    row_width = shared_region.shape[1]
    if not isinstance(row_width, int) or row_width <= 0 or (row_width & (row_width - 1)) != 0:
        raise TileIRLoweringError(
            f"{op_name}: CUDA Tile IR gather/scatter views require power-of-two tile dims "
            f"(row width = {row_width!r} is not a power of two)."
        )

    annotations = dict(stmt.call_annotations)
    if not annotations or "gather4_rows" not in annotations or "gather4_col" not in annotations:
        raise TileIRLoweringError(f"sem_to_ir: {op_name} requires 'gather4_rows'/'gather4_col' annotations on the TIR call.")

    rows = list(annotations["gather4_rows"])
    if len(rows) != 4:
        raise TileIRLoweringError(f"sem_to_ir: {op_name} expects exactly 4 row indices in 'gather4_rows', got {len(rows)}.")

    row_vals = tuple(_lower_gather4_index_scalar(row_expr, scope, builder) for row_expr in rows)
    col_val = _lower_gather4_index_scalar(annotations["gather4_col"], scope, builder)

    op_cls = CopyGather if is_gather else CopyScatter
    op = op_cls(src=src_val, dst=dst_val, row_indices=row_vals, col=col_val)
    builder.create(op)


@tile_op_impl("tl.tileop.copy")
def _lower_copy(stmt: SemanticStmt, attrs: dict, scope: LoweringScope, builder: IRBuilder) -> None:
    # T.tma_gather4 / T.tma_scatter4: a 2-region tl.tileop.copy annotated
    # is_gather4/is_scatter4 carries its 4 row indices + column offset in the
    # call's annotations dict (read directly in _lower_gather4_scatter4), not
    # in an extra index-BUFFER region.
    # Presence-with-truthy-value, matching the semantic layer's
    # `_annotation_is_truthy` (a falsy annotation value, e.g. "0", is treated
    # the same as "not set" on both sides of this boundary).
    is_gather4 = _parse_bool(attrs.get("annotation.is_gather4"))
    is_scatter4 = _parse_bool(attrs.get("annotation.is_scatter4"))
    if is_gather4 or is_scatter4:
        _lower_gather4_scatter4(stmt, attrs, scope, builder, is_gather=is_gather4)
        return

    src_val, dst_val = _require_region_buffers(stmt, scope, op_name="copy", roles=("src", "dst"))
    src_region = stmt.regions[0]
    dst_region = stmt.regions[1]
    # Use _region_tile_shape to preserve full rank when region has dynamic dims
    # (e.g. kv_end - kv_start).  A plain int-filter would drop such dims and
    # reduce rank, causing make_partition_view rank mismatches on 4D TensorViews
    # (e.g. KV buffer (1,64,1,64) with region shape (1,dynamic,1,64)).
    tile_shape = _region_tile_shape(src_region.shape, tuple(src_val.type.shape))
    # Capture dst tile shape separately so SHARED→GLOBAL copies
    # (e.g. O_shared[64,128] → Output[1,64,1,128]) can use the correct rank
    # when calling make_partition_view on the GLOBAL dst TensorView.
    dst_tile_shape_raw = _region_tile_shape(dst_region.shape, tuple(dst_val.type.shape))
    dst_tile_shape = dst_tile_shape_raw if dst_tile_shape_raw != tile_shape else ()

    # When the dst is a SHARED/REGISTER buffer (tile-map based) and the squeezed
    # src tile_shape has MORE elements than the dst declared type shape, the
    # dynamic-dim fallback in _region_tile_shape inflated a dimension to the full
    # buffer size (e.g. kv_ctx=128 instead of block_N=64).
    # Cap the inflated non-singleton dims in tile_shape to match the dst shape.
    if dst_val.type.space != MemSpace.GLOBAL:
        from tilelang.tileir.emission_utils import _squeeze_shape

        squeezed_src = _squeeze_shape(list(tile_shape))
        dst_declared = list(dst_val.type.shape)
        src_elems = functools.reduce(operator.mul, squeezed_src, 1) if squeezed_src else 1
        dst_elems = functools.reduce(operator.mul, dst_declared, 1) if dst_declared else 1
        if src_elems > dst_elems and len(squeezed_src) == len(dst_declared):
            # Build a corrected tile_shape: replace each non-singleton dim with
            # min(current_dim, corresponding_dst_dim).
            corrected: list[int] = []
            dst_idx = 0
            for d in tile_shape:
                if d == 1:
                    corrected.append(1)
                else:
                    corrected.append(min(d, dst_declared[dst_idx]) if dst_idx < len(dst_declared) else d)
                    dst_idx += 1
            tile_shape = tuple(corrected)

    # Resolve real view indices from TIR source (src and dst).
    # For GLOBAL buffers, the indices tell the TKO load/store which tile to
    # access.  For SHARED/REGISTER buffers the index is always 0.
    # A start offset that does not divide exactly by the tile extent (e.g.
    # varlen attention's ``cu_seqlens[b] + bx*block_M``) switches that side
    # to element-offset indexing over a unit-stride strided view.
    src_tir_indices = _extract_tir_region_indices(stmt.call_args, 0)
    dst_tir_indices = _extract_tir_region_indices(stmt.call_args, 1)
    if src_val.type.space != MemSpace.GLOBAL and dst_val.type.space != MemSpace.GLOBAL:
        src_logical = scope.buffer_logical_shapes.get(src_val.name)
        dst_logical = scope.buffer_logical_shapes.get(dst_val.name)
        if (
            src_tir_indices is not None
            and dst_tir_indices is not None
            and all(isinstance(idx, (int, _tirx.IntImm)) and int(idx) == 0 for idx in (*src_tir_indices, *dst_tir_indices))
            and src_logical == tile_shape == dst_tile_shape_raw == dst_logical
            and src_val.type.shape == dst_val.type.shape
        ):
            tile_shape = tuple(src_val.type.shape)
            dst_tile_shape = ()
    # Whole GLOBAL <-> tile copies use the allocation's physical tile shape.
    # Logical TensorView bounds mask padding; partial regions must retain their
    # original shape so this cannot overwrite valid elements outside the copy.
    padded_full_copy = False
    if (src_val.type.space == MemSpace.GLOBAL) != (dst_val.type.space == MemSpace.GLOBAL):
        from tilelang.tileir.emission_utils import _dtype_supports_tensor_view

        global_val, local_val = (src_val, dst_val) if src_val.type.space == MemSpace.GLOBAL else (dst_val, src_val)
        logical_shape = scope.buffer_logical_shapes.get(local_val.name)
        origin_zero = (
            src_tir_indices is not None
            and dst_tir_indices is not None
            and all(isinstance(idx, (int, _tirx.IntImm)) and int(idx) == 0 for idx in (*src_tir_indices, *dst_tir_indices))
        )
        if (
            origin_zero
            and logical_shape == tile_shape == dst_tile_shape_raw == tuple(global_val.type.shape)
            and tuple(local_val.type.shape) != logical_shape
            and _dtype_supports_tensor_view(global_val.type.dtype.name)
            and global_val not in scope.alloca_buffer_values
        ):
            tile_shape = tuple(local_val.type.shape)
            dst_tile_shape = ()
            padded_full_copy = True
    src_idx, src_elem = _compute_view_indices(src_tir_indices, tile_shape, scope, builder, what="copy src")
    # Use dst_tile_shape for computing dst partition indices when it differs.
    _dst_shape_for_idx = dst_tile_shape if dst_tile_shape else tile_shape
    dst_idx, dst_elem = _compute_view_indices(dst_tir_indices, _dst_shape_for_idx, scope, builder, what="copy dst")
    src_in_bounds = not padded_full_copy and _prove_region_in_bounds(src_tir_indices, tile_shape, tuple(src_val.type.shape), scope, builder)

    # Extract per-copy load/store hints from semantic attrs.
    # The TIR annotation keys are "annotation.tileir.latency" and "annotation.disable_tma".
    copy_latency: Any = None
    copy_allow_tma: Any = None
    raw_latency = attrs.get("annotation.tileir.latency")
    if raw_latency is not None:
        try:
            latency_int = int(raw_latency)
        except (ValueError, TypeError):
            latency_int = None
        if latency_int is not None:
            if latency_int < 1 or latency_int > 10:
                raise TileIRLoweringError(f"TileIR backend requires copy `latency` in [1, 10]; got {latency_int}.")
            copy_latency = latency_int
    # disable_tma annotation → allow_tma=False.
    if attrs.get("annotation.disable_tma") is not None:
        copy_allow_tma = False

    op = Copy(
        src=src_val,
        dst=dst_val,
        tile_shape=tile_shape,
        dst_tile_shape=dst_tile_shape,
        src_indices=src_idx,
        dst_indices=dst_idx,
        src_elem_view=src_elem,
        dst_elem_view=dst_elem,
        src_in_bounds=src_in_bounds,
        latency=copy_latency,
        allow_tma=copy_allow_tma,
    )
    builder.create(op)


@tile_op_impl("tl.tileop.tma_copy")
def _lower_tma_copy(stmt: SemanticStmt, attrs: dict, scope: LoweringScope, builder: IRBuilder) -> None:
    src_val, dst_val = _require_region_buffers(stmt, scope, op_name="tma_copy", roles=("src", "dst"))
    src_region = stmt.regions[0]
    dst_region = stmt.regions[1]
    # Use _region_tile_shape to preserve full rank (mirrors _lower_copy).
    tile_shape = _region_tile_shape(src_region.shape, tuple(src_val.type.shape))
    # Capture dst tile shape separately (mirrors _lower_copy).
    dst_tile_shape_raw = _region_tile_shape(dst_region.shape, tuple(dst_val.type.shape))
    dst_tile_shape = dst_tile_shape_raw if dst_tile_shape_raw != tile_shape else ()

    # Resolve real partition indices from TIR source.
    src_tir_indices = _extract_tir_region_indices(stmt.call_args, 0)
    dst_tir_indices = _extract_tir_region_indices(stmt.call_args, 1)
    src_idx, src_elem = _compute_tma_view_indices(src_tir_indices, tile_shape, scope, builder, what="tma_copy src")
    _dst_shape_for_idx = dst_tile_shape if dst_tile_shape else tile_shape
    dst_idx, dst_elem = _compute_tma_view_indices(dst_tir_indices, _dst_shape_for_idx, scope, builder, what="tma_copy dst")
    for role, buf, elementwise in (("src", src_val, src_elem), ("dst", dst_val, dst_elem)):
        if elementwise and buf.type.space != MemSpace.GLOBAL:
            raise _UnsupportedTileIRNode(
                f"tma_copy {role}: shared/register tile slices require an exactly divisible start offset; "
                "element-offset views are supported only for global buffers."
            )

    op = TmaCopy(
        src=src_val,
        dst=dst_val,
        tile_shape=tile_shape,
        dst_tile_shape=dst_tile_shape,
        src_indices=src_idx,
        dst_indices=dst_idx,
        src_elem_view=src_elem,
        dst_elem_view=dst_elem,
    )
    builder.create(op)


@tile_op_impl("tl.tileop.transpose")
def _lower_transpose(stmt: SemanticStmt, attrs: dict, scope: LoweringScope, builder: IRBuilder) -> None:
    """Lower a rank-2 ``tl.tileop.transpose`` (``T.transpose``) to a
    ``TransposeCopy`` op: ``dst[j, i] = src[i, j]``.

    The actual SHARED-permute vs GLOBAL-strided_view choice happens in
    ``TransposeCopy.emit_mlir`` (it needs the fully-materialised ``ctx`` to
    know whether ``src`` is a tile-map buffer).  This handler is the
    authoritative validation site and enforces, for BOTH src and dst
    regardless of memory space:

      - the declared buffer is exactly rank-2 (reject a (1,M,N)-style
        buffer loudly and clearly, rather than confusingly comparing an
        unsqueezed buffer shape against a squeezed region shape);
      - the region is the buffer's FULL extent, at a statically-zero offset
        (a dynamic-extent region like ``A[0:seq_len, :]`` must raise, not
        silently inflate to "the whole buffer" via ``_region_tile_shape``'s
        fallback; this applies to the dst side too — a partial GLOBAL dst
        region must not be silently overwritten at the wrong window);
      - dst's shape must be exactly the reverse of src's shape (the
        frontend's ``get_extent`` assert only checks rank >= 2, not the
        reversal, so a caller passing mismatched buffers must fail loudly
        here rather than emit a wrong-shape strided_view/permute).

    A missing region or a failed buffer lookup must raise loudly rather
    than warn-and-skip (silently dropping the transpose) -- the same
    no-silent-degrade contract as the gather4/scatter4 lowering path. Both
    raise ``TileIRLoweringError``, matching ``_lower_gather4_scatter4``'s
    style.
    """
    if len(stmt.regions) < 2:
        raise TileIRLoweringError(f"sem_to_ir: transpose requires 2 regions (src, dst); got {len(stmt.regions)}.")
    src_region = stmt.regions[0]
    dst_region = stmt.regions[1]
    try:
        src_val = scope.lookup_buffer(src_region.buffer)
        dst_val = scope.lookup_buffer(dst_region.buffer)
    except KeyError as exc:
        raise TileIRLoweringError(f"sem_to_ir: transpose buffer lookup failed: {exc}") from exc

    _require_full_buffer_region(src_region, src_val, op_name="tl.tileop.transpose", side="src", rank=2)
    _require_full_buffer_region(dst_region, dst_val, op_name="tl.tileop.transpose", side="dst", rank=2)

    src_shape = tuple(int(d) for d in src_val.type.shape)
    dst_shape = tuple(int(d) for d in dst_val.type.shape)

    if dst_shape != tuple(reversed(src_shape)):
        raise TileIRLoweringError(
            "tl.tileop.transpose requires dst shape to be the reverse of src shape "
            f"(dst[j,i] = src[i,j]); got src {list(src_shape)}, dst {list(dst_shape)}."
        )

    op = TransposeCopy(src=src_val, dst=dst_val, src_shape=src_shape, dst_shape=dst_shape)
    builder.create(op)


@tile_op_impl("tl.tileop.fill")
def _lower_fill(stmt: SemanticStmt, attrs: dict, scope: LoweringScope, builder: IRBuilder) -> None:
    (dst_val,) = _require_region_buffers(stmt, scope, op_name="fill", roles=("dst",))
    dst_region = stmt.regions[0]
    fill_value_str = attrs.get("value", "0.0")
    try:
        fill_value_float = float(fill_value_str)
    except (ValueError, TypeError):
        fill_value_float = 0.0
    # Use int value for integer dtypes so ct.constant doesn't reject float 0.0 for i32
    try:
        dst_dtype = dst_val.type.dtype
        if not dst_dtype._is_float:
            fill_value: Any = int(fill_value_float)
        else:
            fill_value = fill_value_float
    except (AttributeError, TypeError):
        fill_value = fill_value_float
    tile_shape = tuple(d for d in dst_region.shape if isinstance(d, int))
    op = Fill(dst=dst_val, value=fill_value, tile_shape=tile_shape)
    builder.create(op)


# Scale dtypes accepted by the plain-buffer block-scaled MMA path.
_BLOCKSCALED_SCALE_DTYPES = frozenset(
    {
        "float8_e8m0fnu",
        "float8_e4m3",
        "float8_e4m3fn",
        "uint8",
        "int8",
    }
)


def _lower_gemm_scaled(stmt: SemanticStmt, attrs: dict, scope: LoweringScope, builder: IRBuilder) -> None:
    """Lower the plain-buffer subset of ``T.tcgen05_gemm_blockscaled``.

    The supported form uses a single whole-K MMA, one CTA, unpacked scale
    buffers, and shapes matching the declared scale granularities.
    """
    try:
        a_val = scope.lookup_buffer(stmt.regions[0].buffer)
        b_val = scope.lookup_buffer(stmt.regions[1].buffer)
        c_val = scope.lookup_buffer(stmt.regions[2].buffer)
        sfa_val = scope.lookup_buffer(stmt.regions[3].buffer)
        sfb_val = scope.lookup_buffer(stmt.regions[4].buffer)
    except KeyError as exc:
        raise TileIRLoweringError(f"sem_to_ir: tcgen05_gemm_blockscaled buffer lookup failed: {exc}") from exc

    # (a) k_start must be statically 0.  Read the ORIGINAL TIR call args
    # (stmt.call_args), not the stringified attrs dict, so a non-constant
    # k_start expression is never mistaken for a literal zero (mirrors the
    # source-args idiom used by the atomic-CAS lowering for the same reason).
    k_start_val = _extract_static_int(stmt.call_args[15]) if len(stmt.call_args) > 15 else None
    if k_start_val != 0:
        raise TileIRLoweringNotImplementedError(
            "K-split tcgen05 blockscaled form (k_start != 0) is not yet supported by the TileIR backend; issue a single whole-K gemm."
        )

    # (b) use_2cta is not supported by this backend.
    if _parse_bool(attrs.get("annotation.use_2cta", "0")):
        raise TileIRLoweringNotImplementedError("use_2cta tcgen05 blockscaled gemm is not yet supported by the TileIR backend.")

    def _raw_dtype(region_idx: int, val: Any) -> str:
        raw = scope.lookup_raw_dtype(stmt.regions[region_idx].buffer)
        return raw if raw is not None else val.type.dtype.name

    sfa_raw = _raw_dtype(3, sfa_val)
    sfb_raw = _raw_dtype(4, sfb_val)

    # (c) SFA/SFB dtype gate — reject the packed-uint32 TMEM scale layout
    # the real tcgen05 examples use (see docstring above).
    if sfa_raw not in _BLOCKSCALED_SCALE_DTYPES:
        raise TileIRLoweringNotImplementedError(
            "packed-uint32 TMEM scale layout is not supported; declare plain e8m0/uint8 "
            f"scale buffers (SFA dtype must be one of {sorted(_BLOCKSCALED_SCALE_DTYPES)}, got {sfa_raw!r})."
        )
    if sfb_raw not in _BLOCKSCALED_SCALE_DTYPES:
        raise TileIRLoweringNotImplementedError(
            "packed-uint32 TMEM scale layout is not supported; declare plain e8m0/uint8 "
            f"scale buffers (SFB dtype must be one of {sorted(_BLOCKSCALED_SCALE_DTYPES)}, got {sfb_raw!r})."
        )

    # (d) SFA/SFB shape vs. granularity validation.
    sf_a_granularity_k = int(attrs.get("annotation.sf_a_granularity_k"))
    sf_b_granularity_k = int(attrs.get("annotation.sf_b_granularity_k"))

    def _try_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    K_val = _try_int(attrs.get("K"))
    if K_val is not None:
        if K_val % sf_a_granularity_k != 0:
            raise TileIRLoweringNotImplementedError(
                f"T.tcgen05_gemm_blockscaled: K ({K_val}) is not evenly divisible by sf_a_granularity_k ({sf_a_granularity_k})."
            )
        if K_val % sf_b_granularity_k != 0:
            raise TileIRLoweringNotImplementedError(
                f"T.tcgen05_gemm_blockscaled: K ({K_val}) is not evenly divisible by sf_b_granularity_k ({sf_b_granularity_k})."
            )

        c_shape = stmt.regions[2].shape
        if len(c_shape) == 2 and all(isinstance(d, int) for d in c_shape):
            M_val, N_val = c_shape
            sfa_shape = tuple(stmt.regions[3].shape)
            sfb_shape = tuple(stmt.regions[4].shape)
            expected_sfa = (M_val, K_val // sf_a_granularity_k)
            expected_sfb = (K_val // sf_b_granularity_k, N_val)
            if sfa_shape != expected_sfa:
                raise TileIRLoweringNotImplementedError(
                    f"T.tcgen05_gemm_blockscaled: SFA shape must be (M, K // sf_a_granularity_k) = {expected_sfa}, got {sfa_shape}."
                )
            if sfb_shape != expected_sfb:
                raise TileIRLoweringNotImplementedError(
                    f"T.tcgen05_gemm_blockscaled: SFB shape must be (K // sf_b_granularity_k, N) = {expected_sfb}, got {sfb_shape}."
                )

    trans_a = _parse_bool(attrs.get("transpose_A", "0"))
    trans_b = _parse_bool(attrs.get("transpose_B", "0"))
    clear = _parse_bool(attrs.get("clear_accum", "0"))

    op = GemmScaled(
        lhs=a_val,
        rhs=b_val,
        acc=c_val,
        lhs_scale=sfa_val,
        rhs_scale=sfb_val,
        trans_a=trans_a,
        trans_b=trans_b,
        clear=clear,
    )
    builder.create(op)


def _lower_gemm_common(stmt: SemanticStmt, attrs: dict, scope: LoweringScope, builder: IRBuilder, *, tcgen05: bool = False) -> None:
    # Both generic and instruction-specific blockscaled calls carry scale
    # granularities as annotations and append SFA/SFB/k_start to dense slots.
    blockscaled = "annotation.sf_a_granularity_k" in attrs

    if blockscaled:
        _require_region_buffers(
            stmt,
            scope,
            op_name="tcgen05 blockscaled gemm",
            roles=("A", "B", "C", "SFA", "SFB"),
        )
        _lower_gemm_scaled(stmt, attrs, scope, builder)
        return

    op_name = "tcgen05_gemm" if tcgen05 else "gemm"
    a_val, b_val, c_val = _require_region_buffers(stmt, scope, op_name=op_name, roles=("A", "B", "C"))

    trans_a = _parse_bool(attrs.get("transpose_A", "0"))
    trans_b = _parse_bool(attrs.get("transpose_B", "0"))
    clear = _parse_bool(attrs.get("clear_accum", "0"))

    # Detect unsigned-ness from the ORIGINAL SemanticBuffer.dtype string
    # stored in the scope before alias collapse (lookup_dtype("uint8") → int8,
    # so a_val.type.dtype.name would always be "int8" for uint8 inputs — wrong).
    lhs_raw = scope.lookup_raw_dtype(stmt.regions[0].buffer)
    rhs_raw = scope.lookup_raw_dtype(stmt.regions[1].buffer)
    lhs_unsigned = _is_unsigned_dtype(lhs_raw) if lhs_raw is not None else False
    rhs_unsigned = _is_unsigned_dtype(rhs_raw) if rhs_raw is not None else False

    if tcgen05:
        op = Tcgen05Gemm(
            lhs=a_val,
            rhs=b_val,
            acc=c_val,
            trans_a=trans_a,
            trans_b=trans_b,
            clear=clear,
            lhs_unsigned=lhs_unsigned,
            rhs_unsigned=rhs_unsigned,
        )
    else:
        op = Gemm(
            lhs=a_val,
            rhs=b_val,
            acc=c_val,
            trans_a=trans_a,
            trans_b=trans_b,
            clear=clear,
            lhs_unsigned=lhs_unsigned,
            rhs_unsigned=rhs_unsigned,
        )
    builder.create(op)


@tile_op_impl("tl.tileop.gemm", "tl.tileop.wgmma_gemm", "tl.tileop.gemm_blockscaled")
def _lower_gemm(stmt: SemanticStmt, attrs: dict, scope: LoweringScope, builder: IRBuilder) -> None:
    # "tl.tileop.wgmma_gemm" (T.wgmma_gemm) shares _gemm_impl's exact call
    # layout with "tl.tileop.gemm"; it only pins the Hopper WGMMA instruction
    # (plus wg_wait=-1, i.e. no implicit warpgroup wait).  Instruction
    # selection (wgmma vs mma) is the downstream cuda_tile optimizer's job in
    # this backend, and GEMM completion is ordered by TKO tokens, so both
    # lower to the same Gemm op and the wg_wait attr is intentionally unused.
    _lower_gemm_common(stmt, attrs, scope, builder, tcgen05=False)


@tile_op_impl("tl.tileop.tcgen05_gemm", "tl.tileop.tcgen05_gemm_blockscaled")
def _lower_tcgen05_gemm(stmt: SemanticStmt, attrs: dict, scope: LoweringScope, builder: IRBuilder) -> None:
    _lower_gemm_common(stmt, attrs, scope, builder, tcgen05=True)


@tile_op_impl("tl.tileop.reduce")
def _lower_reduce(stmt: SemanticStmt, attrs: dict, scope: LoweringScope, builder: IRBuilder) -> None:
    src_val, dst_val = _require_region_buffers(stmt, scope, op_name="reduce", roles=("src", "dst"))
    reduce_type = attrs.get("reduce_type", "sum")
    dim = attrs.get("dim", "0")
    try:
        dim = int(dim)
    except (ValueError, TypeError):
        dim = 0
    clear = _parse_bool(attrs.get("clear", "1"), default=True)
    # Detect unsigned-ness from the ORIGINAL SemanticBuffer.dtype string
    # (mirrors the gemm lhs_unsigned/rhs_unsigned threading above) —
    # `src_val.type.dtype.name` would always read back
    # "int32" for a uint32 buffer since the TileIR type registry alias-
    # collapses unsigned integers to signless MLIR types.
    src_raw = scope.lookup_raw_dtype(stmt.regions[0].buffer)
    src_unsigned = _is_unsigned_dtype(src_raw) if src_raw is not None else False
    op = Reduce(src=src_val, dst=dst_val, op=reduce_type, axis=dim, clear=clear, src_unsigned=src_unsigned)
    builder.create(op)


@tile_op_impl("tl.tileop.cumsum")
def _lower_cumsum(stmt: SemanticStmt, attrs: dict, scope: LoweringScope, builder: IRBuilder) -> None:
    src_val, dst_val = _require_region_buffers(stmt, scope, op_name="cumsum", roles=("src", "dst"))
    dim = attrs.get("dim", "0")
    try:
        dim = int(dim)
    except (ValueError, TypeError):
        dim = 0
    reverse = _parse_bool(attrs.get("reverse", "0"))
    op = Cumsum(src=src_val, dst=dst_val, axis=dim, reverse=reverse, kind="sum")
    builder.create(op)


@tile_op_impl("tl.tileop.cummax")
def _lower_cummax(stmt: SemanticStmt, attrs: dict, scope: LoweringScope, builder: IRBuilder) -> None:
    """Lower `tl.tileop.cummax` to a `Cumsum(kind="max")` op.

    A malformed statement (missing regions, or a region naming a buffer the
    lowering scope has never registered) must raise loudly rather than
    warn-and-skip (silently dropping the entire cummax computation). Mirrors
    `_lower_gather4_scatter4`'s style: raise `TileIRLoweringError` instead (a
    hand-built/malformed `SemanticStmt` reaching this handler is a
    lowering-pipeline bug, not a recoverable condition worth silently
    degrading).
    """
    if len(stmt.regions) < 2:
        raise TileIRLoweringError(f"sem_to_ir: cummax requires 2 regions (src, dst); got {len(stmt.regions)}.")
    try:
        src_val = scope.lookup_buffer(stmt.regions[0].buffer)
        dst_val = scope.lookup_buffer(stmt.regions[1].buffer)
    except KeyError as exc:
        raise TileIRLoweringError(f"sem_to_ir: cummax buffer lookup failed: {exc}") from exc
    dim = attrs.get("dim", "0")
    try:
        dim = int(dim)
    except (ValueError, TypeError):
        dim = 0
    reverse = _parse_bool(attrs.get("reverse", "0"))
    # Same alias-collapse trap as Gemm — thread true
    # unsigned-ness from the pre-collapse dtype string, since Cumsum's "max"
    # kind needs it for both the INT_MIN/0 identity and the ct.max
    # comparison signedness (see Cumsum.emit_mlir).
    src_raw = scope.lookup_raw_dtype(stmt.regions[0].buffer)
    src_unsigned = _is_unsigned_dtype(src_raw) if src_raw is not None else False
    op = Cumsum(src=src_val, dst=dst_val, axis=dim, reverse=reverse, kind="max", src_unsigned=src_unsigned)
    builder.create(op)


# Barrier-like tile ops


@tile_op_impl(
    "tir.ptx_arrive_barrier",
    "tir.ptx_commit_group",
    "tir.ptx_wait_group",
    "tl.mbarrier_wait_parity",
    # T.wait_wgmma(id) — waits for outstanding Hopper WGMMA groups.  GEMMs
    # are ordered by TKO tokens in this backend, so like the other waits
    # above this is a scheduling hint that emits nothing (Barrier is a no-op).
    "tl.wait_wgmma",
    "tl.tma_store_wait",
)
def _lower_barrier_op(stmt: SemanticStmt, attrs: dict, scope: LoweringScope, builder: IRBuilder) -> None:
    builder.create(Barrier())


@tile_op_impl("tir.tvm_storage_sync")
def _lower_storage_sync(stmt: SemanticStmt, attrs: dict, scope: LoweringScope, builder: IRBuilder) -> None:
    builder.create(GridSync())


@tile_op_impl("tir.tvm_thread_allreduce")
def _lower_thread_allreduce(stmt: SemanticStmt, attrs: dict, scope: LoweringScope, builder: IRBuilder) -> None:
    builder.create(ThreadAllreduce())


@tile_op_impl("tir.break_loop", "tl.loop_break")
def _lower_break(stmt: SemanticStmt, attrs: dict, scope: LoweringScope, builder: IRBuilder) -> None:
    # "tir.break_loop" comes from a Python `break` inside a TIR-scripted
    # `while`; "tl.loop_break" is the TileLang builtin `T.loop_break()` calls
    # directly (e.g. from `T.Persistent`'s generated wave-exhaustion guard).
    # Both terminate the innermost enclosing loop identically.
    builder.create(Break())


@tile_op_impl("tir.continue_loop")
def _lower_continue(stmt: SemanticStmt, attrs: dict, scope: LoweringScope, builder: IRBuilder) -> None:
    builder.create(Continue())


@tile_op_impl("tl.device_assert", "tl.device_assert_with_msg")
def _lower_device_assert(stmt: SemanticStmt, attrs: dict, scope: LoweringScope, builder: IRBuilder) -> None:
    """Lower device_assert / device_assert_with_msg to a DeviceAssert IR op.

    The condition is the first TIR call arg (``stmt.call_args[0]``).  The
    message is the second arg (if present and a StringImm) for
    ``tl.device_assert_with_msg``, or the fallback "device_assert" string.

    The SemanticStmt keeps call arguments as expression payloads rather than
    retaining the original TIR Call node.
    """
    cond_val = None
    message = "device_assert"
    if stmt.call_args:
        try:
            cond_val = lower_expr(stmt.call_args[0], scope, builder)
        except _UnsupportedTileIRNode:
            # Assert condition not lowerable → emit the assert without a condition
            # (a debug check, safe to weaken). Other errors propagate.
            cond_val = None
        if len(stmt.call_args) >= 2:
            arg1 = stmt.call_args[1]
            if isinstance(arg1, _tirx.StringImm):
                message = str(arg1.value)
    if cond_val is None:
        raise _UnsupportedTileIRNode(
            "device_assert condition could not be lowered: TIR call args not accessible in SemanticStmt.call_args."
        )
    op = DeviceAssert(cond=cond_val, message=message)
    builder.create(op)


# call_extern tile ops


@tile_op_impl("debug_print_msg", "debug_print_var", "debug_print_buffer_value")
def _lower_debug_print(stmt: SemanticStmt, attrs: dict, scope: LoweringScope, builder: IRBuilder) -> None:
    op = DebugPrint(message="debug_print")
    builder.create(op)


def _extract_access_ptr_buffer_name(tir_arg: Any) -> str | None:
    """Extract the buffer name from a ``tl.access_ptr(BufferLoad(buf, ...), ...)`` call.

    Returns the buffer name string, or None if the arg does not match the
    expected access_ptr structure.
    """
    if not isinstance(tir_arg, _tirx.Call):
        return None
    op_str = str(getattr(tir_arg.op, "name", ""))
    if "access_ptr" not in op_str:
        return None
    if not tir_arg.args:
        return None
    inner = tir_arg.args[0]
    if isinstance(inner, _tirx.BufferLoad):
        return inner.buffer.name
    return None


# main tile_op dispatcher


@impl("tile_op")
def _lower_tile_op(stmt: SemanticStmt, scope: LoweringScope, builder: IRBuilder) -> None:
    attrs = dict(stmt.attrs)
    op_name = attrs.get("op", "")
    extern_name = attrs.get("extern_name", "")

    # Try op_name first, then extern_name (for tir.call_extern variants).
    handler = TILE_OP_IMPL.get(op_name) or TILE_OP_IMPL.get(extern_name)
    if handler is None:
        raise _UnsupportedTileIRNode(f"Unsupported tile_op: op={op_name!r} extern_name={extern_name!r}")
    handler(stmt, attrs, scope, builder)


def _extract_region_buffer_name(tir_arg: Any) -> str | None:
    """Extract the buffer name from a region/access_ptr/BufferLoad TIR arg.

    Handles a direct ``BufferLoad`` or one wrapped by ``tl.region``,
    ``tl.access_ptr``, or another call whose first argument is the load.

    Returns the buffer name string, or None if extraction fails.
    """
    # Direct BufferLoad (e.g. val[0] as-is)
    if isinstance(tir_arg, _tirx.BufferLoad):
        return tir_arg.buffer.name
    if not isinstance(tir_arg, _tirx.Call):
        return None
    if not tir_arg.args:
        return None
    inner = tir_arg.args[0]
    if isinstance(inner, _tirx.BufferLoad):
        return inner.buffer.name
    return None


def _extract_static_int(tir_expr: Any) -> int | None:
    """Extract an integer constant from a TIR expression; None if non-static."""
    try:
        if isinstance(tir_expr, _tirx.IntImm):
            return int(tir_expr)
    except Exception:
        pass
    try:
        v = int(tir_expr)
        return v
    except Exception:
        return None

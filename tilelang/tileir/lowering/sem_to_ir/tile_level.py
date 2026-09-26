"""SemanticIR -> TileIR lowering: tile-level expression lowering.

Provides ``_lower_tile_level_expr`` (used inside T.Parallel loop bodies) and its
FMA helper.  Imports the shared foundation and scalar ``lower_expr``; the
``_extract_region_buffer_name`` / ``_extract_static_int`` helpers it needs from
``tile_ops`` are imported function-locally to keep the import graph acyclic.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from tvm import tirx as _tir

from tilelang.tileir.ir.builder import IRBuilder
from tilelang.tileir.errors import _UnsupportedTileIRNode
from tilelang.tileir.ir.types import MemSpace, TileType
from tilelang.tileir.ir.value import Value
from tilelang.tileir.ir.ops import (
    AtomicLoad,
    AtomicRMW,
    Broadcast,
    Cast,
    GatherLoad,
    Iota,
    Load,
    Permute,
    RepeatInterleave,
)

from ._base import (
    _canonical_dtype_str,
    LoweringScope,
    _get_binary_op_fn,
    _op_name_from_call,
    _tir_dtype_to_tile_type,
    _UNARY_CALL_FN,
)
from .expr import lower_expr, _lower_attr_expr, _make_elementwise, _make_select


def _try_lower_parallel_repeat_interleave_load(
    expr: Any,
    scope: LoweringScope,
    builder: IRBuilder,
    *,
    ordered_vars: list[str] | None,
    ordered_extents: list[int] | None,
) -> Value | None:
    """Lower ``buf[..., loop_var // C, ...]`` as a collective tile repeat.

    This recognizes one narrow packed-data pattern: every buffer axis maps to
    the parallel loop variable at the same position, except exactly one axis
    uses floor-division by a positive compile-time constant.  The source tile
    is loaded once and repeated adjacently on that axis.  Unmatched expressions
    return ``None`` so the caller can continue through the regular tile/scalar
    lowering paths.
    """
    if ordered_vars is None or ordered_extents is None:
        return None

    if not isinstance(expr, _tir.BufferLoad):
        return None
    if len(expr.indices) != len(ordered_vars) or len(expr.indices) != len(ordered_extents):
        return None

    try:
        buf_val = scope.lookup_buffer(expr.buffer.name)
    except KeyError:
        return None
    if buf_val.type.space not in (MemSpace.SHARED, MemSpace.REGISTER):
        return None

    src_shape = tuple(buf_val.type.shape)
    if len(src_shape) != len(expr.indices):
        return None

    repeat_axis: int | None = None
    repeats: int | None = None
    for axis, (index, var_name, target_extent, src_extent) in enumerate(zip(expr.indices, ordered_vars, ordered_extents, src_shape)):
        if isinstance(index, _tir.Var) and index.name == var_name:
            if src_extent != target_extent:
                return None
            continue
        if (
            isinstance(index, _tir.FloorDiv)
            and isinstance(index.a, _tir.Var)
            and index.a.name == var_name
            and isinstance(index.b, _tir.IntImm)
        ):
            factor = int(index.b)
            if factor <= 0 or repeat_axis is not None:
                return None
            if not isinstance(target_extent, int) or src_extent * factor != target_extent:
                return None
            repeat_axis = axis
            repeats = factor
            continue
        return None

    if repeat_axis is None or repeats is None:
        return None

    loaded_ty = TileType(
        dtype=buf_val.type.dtype,
        shape=src_shape,
        space=MemSpace.REGISTER,
        layout=None,
    )
    load = builder.create(
        Load(src=buf_val, tile_shape=src_shape, indices=tuple(0 for _ in src_shape)),
        result_types=(loaded_ty,),
    )
    result_ty = TileType(
        dtype=buf_val.type.dtype,
        shape=tuple(ordered_extents),
        space=MemSpace.REGISTER,
        layout=None,
    )
    repeated = builder.create(
        RepeatInterleave(src=load.results[0], axis=repeat_axis, repeats=repeats),
        result_types=(result_ty,),
    )
    return repeated.results[0]


def _lower_tile_level_expr(
    expr: Any,
    scope: LoweringScope,
    builder: IRBuilder,
    loop_vars: set,
    *,
    ordered_vars: list[str] | None = None,
    ordered_extents: list[int] | None = None,
) -> Value:
    """Lower a TIR PrimExpr in tile mode (used inside T.Parallel loop bodies).

    Differences from ``lower_expr``:
    * ``BufferLoad(buf, [i, j, ...])`` where *buf* is a SHARED/REGISTER buffer
      → returns the whole-tile buffer Value (not a scalar Load op), provided
      ALL indices in the BufferLoad are loop-variable names in *loop_vars*.
    * ``Cast(dtype, tile_expr)`` → ``Cast`` TileIR op producing a shaped tile.
    * ``Var(name)`` where ``name ∈ loop_vars`` AND ``ordered_vars`` is provided
      → emits an ``Iota`` op (an axis-indexed affine range tile, like numpy
      ``arange``).  This handles causal-mask conditions like
      ``m_idx * 64 + i >= k * 64 + j`` inside a T.Parallel body where ``i``
      and ``j`` are loop-variable names, not scalar Values in scope.
    * All other nodes delegate to ``lower_expr`` (scalar fallback).

    Parameters
    ----------
    expr :
        TIR ``PrimExpr`` to lower.
    scope :
        Current lowering scope (for Var lookups and buffer lookups).
    builder :
        IRBuilder to create ops.
    loop_vars :
        Set of TIR loop variable names for the enclosing T.Parallel nest.
        A BufferLoad is treated as a tile access iff ALL its index names
        are in *loop_vars*.
    ordered_vars :
        Ordered list of loop variable names (outer → inner), e.g. ["i", "j"].
        Required for iota emission; when None, loop-var Vars fall through to
        ``lower_expr`` (which raises _UnsupportedTileIRNode).
    ordered_extents :
        Corresponding integer extents for each element of *ordered_vars*,
        e.g. [64, 64].
    """

    # Helper: re-call self with same ordered_vars/extents forwarded.
    def _recurse(sub_expr: Any) -> Value:
        return _lower_tile_level_expr(
            sub_expr,
            scope,
            builder,
            loop_vars,
            ordered_vars=ordered_vars,
            ordered_extents=ordered_extents,
        )

    # BufferLoad in tile context (whole-tile / strided / partial-index loads).
    if isinstance(expr, _tir.BufferLoad):
        return _lower_buffer_load_tile(expr, scope, builder, loop_vars, ordered_vars, ordered_extents)

    # Cast in tile context
    if isinstance(expr, _tir.Cast):
        return _lower_cast_tile(expr, builder, _recurse)

    # Select in tile context (e.g. T.abs → select(x>=0, x, -x)); all three
    # sub-exprs lowered in tile mode so loop vars in the condition/branches
    # produce tile values instead of UnboundVar errors from scalar lower_expr.
    if isinstance(expr, _tir.Select):
        return _lower_select_tile(expr, builder, _recurse)

    # Binary ops in tile context (FMA-fused when possible).
    binary_fn = _get_binary_op_fn().get(type(expr))
    if binary_fn is not None:
        return _lower_binop_tile(expr, binary_fn, scope, builder, loop_vars, ordered_vars, ordered_extents, _recurse)

    # Unary/Ternary Call in tile context (if_then_else, atomics, intrinsics).
    if isinstance(expr, _tir.Call):
        handled = _lower_call_tile(expr, scope, builder, ordered_vars, ordered_extents, _recurse)
        if handled is not None:
            return handled

    # Var: may be a let-bound tile value OR a loop-var iota index.
    if isinstance(expr, _tir.Var):
        handled = _lower_var_tile(expr, scope, loop_vars, ordered_vars, ordered_extents, builder)
        if handled is not None:
            return handled

    # Fallback: delegate to scalar lower_expr
    return lower_expr(expr, scope, builder)


def _lower_buffer_load_tile(expr: Any, scope: LoweringScope, builder: IRBuilder, loop_vars: set, ordered_vars, ordered_extents) -> Value:
    """Lower a tile-context ``BufferLoad`` by trying each load pattern in order,
    falling back to the scalar ``lower_expr``. Always returns a ``Value``.
    """
    # Constant element indices remain scalar inside a parallel expression.
    # Treating e.g. scale[1] as a whole-tile access both loses the offset and
    # incorrectly broadcasts the scale buffer's extent to the parallel tile.
    # local.var is participant-private and may have been promoted to the
    # surrounding parallel shape; its syntactic [0] denotes each lane's value.
    if expr.buffer.scope() != "local.var" and all(isinstance(index, _tir.IntImm) for index in expr.indices):
        return lower_expr(expr, scope, builder)

    # A flat local/shared buffer may hold a row-major N-D parallel tile.
    # Preserve both its element offset and its logical parallel axes.
    if len(expr.indices) == 1 and ordered_vars and ordered_extents:
        from .parallel import _split_flattened_parallel_index

        buf = scope.lookup_buffer(expr.buffer.name)
        if buf.type.space in (MemSpace.SHARED, MemSpace.REGISTER):
            flat = _split_flattened_parallel_index(expr.indices[0], ordered_vars, ordered_extents)
            if flat is not None:
                base, extent = flat
                if isinstance(base, _tir.IntImm) and int(base) >= 0 and int(base) + extent <= buf.type.shape[0]:
                    result_ty = TileType(dtype=buf.type.dtype, shape=tuple(ordered_extents), space=MemSpace.REGISTER, layout=None)
                    return builder.create(
                        Load(src=buf, tile_shape=(extent,), indices=(int(base),), elem_view=True, reshape_to=tuple(ordered_extents)),
                        result_types=(result_ty,),
                    ).results[0]

    repeated = _try_lower_parallel_repeat_interleave_load(
        expr,
        scope,
        builder,
        ordered_vars=ordered_vars,
        ordered_extents=ordered_extents,
    )
    if repeated is not None:
        return repeated

    linearized = _try_lower_linearized_reg_load(expr, scope, builder, ordered_vars, ordered_extents)
    if linearized is not None:
        return linearized

    if _is_whole_tile_access(expr, scope, loop_vars):
        whole = _try_lower_whole_tile_load(expr, scope, builder, ordered_vars, ordered_extents)
        if whole is not None:
            return whole

    strided = _try_lower_block_strided_load(expr, scope, builder, loop_vars, ordered_vars, ordered_extents)
    if strided is not None:
        return strided

    partial = _try_lower_partial_index_load(expr, scope, builder, ordered_vars, ordered_extents)
    if partial is not None:
        return partial

    classified = _try_lower_classified_elem_load(expr, scope, builder, loop_vars, ordered_vars, ordered_extents)
    if classified is not None:
        return classified

    gathered = _try_lower_gather_load(expr, scope, builder, loop_vars, ordered_vars, ordered_extents)
    if gathered is not None:
        return gathered

    # Scalar load fallback
    return lower_expr(expr, scope, builder)


def _try_lower_linearized_reg_load(expr: Any, scope: LoweringScope, builder: IRBuilder, ordered_vars, ordered_extents):
    """Linearized read of a 2D SHARED/REGISTER tile under a 1D parallel var:
    ``h[v // D1, v % D1]`` with ``extent(v) == D0*D1`` is the whole (D0, D1)
    tile viewed as a flat (D0*D1,) tile. Returns None on non-matches.
    """
    if ordered_vars is None or ordered_extents is None or len(ordered_vars) != 1:
        return None
    if not isinstance(expr, _tir.BufferLoad) or len(expr.indices) != 2:
        return None
    v_name = ordered_vars[0]
    extent = ordered_extents[0]

    def _expand_let(e):
        # ``i_s = i_sk // D1`` style let aliases: resolve to the raw expr.
        if isinstance(e, _tir.Var) and e.name != v_name:
            binding = scope.get_scalar_expr_binding(e)
            if binding is not None:
                return binding
        return e

    i0, i1 = (_expand_let(i) for i in expr.indices)
    if not (isinstance(i0, _tir.FloorDiv) and isinstance(i0.a, _tir.Var) and i0.a.name == v_name and isinstance(i0.b, _tir.IntImm)):
        return None
    if not (isinstance(i1, _tir.FloorMod) and isinstance(i1.a, _tir.Var) and i1.a.name == v_name and isinstance(i1.b, _tir.IntImm)):
        return None
    d1 = int(i0.b)
    if int(i1.b) != d1:
        return None
    try:
        buf_val = scope.lookup_buffer(expr.buffer.name)
    except KeyError:
        return None
    if buf_val.type.space not in (MemSpace.SHARED, MemSpace.REGISTER):
        return None
    buf_shape = tuple(buf_val.type.shape)
    if len(buf_shape) != 2 or buf_shape[1] != d1 or buf_shape[0] * buf_shape[1] != extent:
        return None
    result_ty = TileType(
        dtype=buf_val.type.dtype,
        shape=(extent,),
        space=MemSpace.REGISTER,
        layout=None,
    )
    load_op = builder.create(
        Load(
            src=buf_val,
            tile_shape=buf_shape,
            indices=(0, 0),
            reshape_to=(extent,),
        ),
        result_types=(result_ty,),
    )
    return load_op.results[0]


def _contains_shaped_binding(expr: Any, scope: LoweringScope) -> bool:
    """Return whether *expr* consumes participant-private shaped SSA."""

    found = False

    def visit(node: Any) -> None:
        nonlocal found
        if isinstance(node, _tir.Var):
            bound_value = scope.lookup(node)
            found = found or (bound_value is not None and bool(tuple(bound_value.type.shape)))

    _tir.stmt_functor.post_order_visit(expr, visit)
    return found


def _classify_gather_dims(indices, scope: LoweringScope, builder: IRBuilder, loop_vars: set, ordered_vars, ordered_extents):
    """Classify per-dim gather index specs (see ``GatherLoad``).

    Returns ``(dim_kinds, dim_values, dim_axes, has_tile_dim)`` or ``None`` on
    a structural non-match.  Shared between the gather load pattern and the
    gather-form parallel atomic.
    """
    ov_list = list(ordered_vars)
    ov_set = set(ov_list)
    result_shape = tuple(int(e) for e in ordered_extents)

    def _contains_parallel_var(e: Any) -> bool:
        if isinstance(e, _tir.Var):
            return e.name in ov_set
        for _attr in ("a", "b", "args", "indices"):
            _c = getattr(e, _attr, None)
            if _c is None:
                continue
            if isinstance(_c, (list, tuple)) or type(_c).__name__ == "Array":
                if any(_contains_parallel_var(x) for x in _c):
                    return True
            elif _contains_parallel_var(_c):
                return True
        return False

    dim_kinds: list[str] = []
    dim_values: list[Any] = []
    dim_axes: list[int] = []
    has_tile_dim = False

    for idx in indices:
        e = idx
        if isinstance(e, _tir.Var) and e.name not in ov_set:
            bound_value = scope.lookup(e)
            if bound_value is not None and tuple(bound_value.type.shape):
                if tuple(bound_value.type.shape) != result_shape:
                    raise _UnsupportedTileIRNode(
                        f"shaped gather index has shape {bound_value.type.shape}, expected participant shape {result_shape}"
                    )
                dim_kinds.append("tile")
                dim_values.append(bound_value)
                dim_axes.append(-1)
                has_tile_dim = True
                continue
            binding = scope.get_scalar_expr_binding(e)
            if binding is not None:
                e = binding
        if _contains_shaped_binding(e, scope):
            tile_value = _lower_tile_level_expr(
                e,
                scope,
                builder,
                loop_vars,
                ordered_vars=ordered_vars,
                ordered_extents=ordered_extents,
            )
            if tuple(tile_value.type.shape) != result_shape:
                raise _UnsupportedTileIRNode(
                    f"shaped gather index lowered to shape {tile_value.type.shape}, expected participant shape {result_shape}; "
                    "refusing scalar fallback"
                )
            dim_kinds.append("tile")
            dim_values.append(tile_value)
            dim_axes.append(-1)
            has_tile_dim = True
            continue
        # Affine chain with one bare parallel var (``base₀ + var + base₁``,
        # pure var included) → iota along the var's axis at the summed base.
        from .parallel import _split_affine_parallel_term

        split = _split_affine_parallel_term(e, ov_set)
        if split is not None:
            base_tir, var_name = split
            if base_tir is None:
                base_val: Any = 0
            else:
                try:
                    base_val = lower_expr(base_tir, scope, builder)
                except _UnsupportedTileIRNode:
                    return None
            dim_kinds.append("iota")
            dim_values.append(base_val)
            dim_axes.append(ov_list.index(var_name))
            continue
        # No parallel vars at all → plain scalar index.
        if not _contains_parallel_var(e):
            if isinstance(e, _tir.IntImm):
                dim_kinds.append("const")
                dim_values.append(int(e))
                dim_axes.append(-1)
                continue
            try:
                scalar_val = lower_expr(e, scope, builder)
            except _UnsupportedTileIRNode:
                return None
            dim_kinds.append("scalar")
            dim_values.append(scalar_val)
            dim_axes.append(-1)
            continue
        # Tile-valued index (e.g. Indices[...] varying along parallel axes).
        try:
            tile_val = _lower_tile_level_expr(e, scope, builder, loop_vars, ordered_vars=ordered_vars, ordered_extents=ordered_extents)
        except _UnsupportedTileIRNode:
            return None
        if tuple(tile_val.type.shape) != result_shape:
            return None
        dim_kinds.append("tile")
        dim_values.append(tile_val)
        dim_axes.append(-1)
        has_tile_dim = True

    return tuple(dim_kinds), tuple(dim_values), tuple(dim_axes), has_tile_dim


def _try_lower_gather_load(expr: Any, scope: LoweringScope, builder: IRBuilder, loop_vars: set, ordered_vars, ordered_extents):
    """Data-dependent gather: a GLOBAL load where some dim's index is itself a
    TILE-valued expression (e.g. ``KV[b, Indices[b, s, g, i*BI + bi], g, d]``).

    Per buffer dim the index is classified as:

    - ``Add(base, parallel_var)`` / pure var → ``("iota", base, axis)``
    - no parallel vars → ``("scalar", lowered scalar)`` / ``("const", int)``
    - anything else that tile-level lowering can produce (an ``Indices`` load
      varying along parallel axes) → ``("tile", value)``

    At least one ``"tile"`` dim is required — simpler forms belong to the
    partition/strided-view patterns, which keep TMA eligibility.  Emitted as a
    ``GatherLoad`` (per-element pointers + ``load_ptr_tko``).  Returns ``None``
    on structural non-matches.
    """

    if ordered_vars is None or ordered_extents is None or not ordered_vars:
        return None
    if not all(isinstance(e, int) and e > 0 for e in ordered_extents):
        return None
    buf_name = expr.buffer.name
    try:
        buf_val = scope.lookup_buffer(buf_name)
    except KeyError:
        return None
    if buf_val.type.space != MemSpace.GLOBAL:
        return None
    raw_shape = tuple(buf_val.type.shape)
    if len(expr.indices) != len(raw_shape):
        return None

    specs = _classify_gather_dims(expr.indices, scope, builder, loop_vars, ordered_vars, ordered_extents)
    if specs is None:
        return None
    dim_kinds, dim_values, dim_axes, has_tile_dim = specs
    if not has_tile_dim:
        return None
    result_shape = tuple(int(e) for e in ordered_extents)

    result_ty = TileType(
        dtype=buf_val.type.dtype,
        shape=result_shape,
        space=MemSpace.REGISTER,
        layout=None,
    )
    op = builder.create(
        GatherLoad(
            src=buf_val,
            result_shape=result_shape,
            dim_kinds=tuple(dim_kinds),
            dim_values=tuple(dim_values),
            dim_axes=tuple(dim_axes),
        ),
        result_types=(result_ty,),
    )
    return op.results[0]


def _try_lower_classified_elem_load(expr: Any, scope: LoweringScope, builder: IRBuilder, loop_vars: set, ordered_vars, ordered_extents):
    """General GLOBAL tile load with per-dim classification and element-offset
    fallback.

    Each buffer dim's index is either ``Add(base, loop_var)`` / ``loop_var``
    (a tile dim of the loop extent) or a scalar expression (tile dim 1).  The
    bases go through ``_compute_view_indices``: exact-divisible bases become
    partition indices, anything else (e.g. varlen's ``cu_seqlens[b]+bx*bM``)
    becomes raw element offsets over a unit-stride strided view.

    This subsumes the aligned-only block-strided pattern for cases it rejects.
    Returns ``None`` (fall through to the scalar path) on structural
    non-matches; loop-var order mismatches also fall through (a transposed
    tile load must not be emitted silently).
    """
    from .parallel import _classify_store_dims, _compute_view_indices

    if ordered_vars is None or ordered_extents is None or not ordered_vars:
        return None
    buf_name = expr.buffer.name
    try:
        buf_val = scope.lookup_buffer(buf_name)
    except KeyError:
        return None
    if buf_val.type.space != MemSpace.GLOBAL:
        return None
    # A shaped let-bound index must take the gather path below.  The classified
    # element-load path only understands affine loop-variable indices and can
    # otherwise extract lane zero from a participant-private tile.
    if any(_contains_shaped_binding(index, scope) for index in expr.indices):
        return None
    raw_shape = tuple(buf_val.type.shape)
    if len(expr.indices) != len(raw_shape):
        return None

    ov_list = list(ordered_vars)
    ov_set = set(ov_list)

    # Expand let-bound index Vars (mirrors the parallel-store path).
    expanded: list[Any] = []
    for idx in expr.indices:
        e = idx
        if isinstance(e, _tir.Var) and e.name not in ov_set:
            binding = scope.get_scalar_expr_binding(e)
            if binding is not None:
                e = binding
        expanded.append(e)

    tile_sizes, base_tirs = _classify_store_dims(tuple(expanded), ov_list, list(ordered_extents), raw_shape)
    if not all(isinstance(s, int) and s > 0 for s in tile_sizes):
        return None
    if all(s == 1 for s in tile_sizes):
        return None  # pure scalar load — leave to the scalar path

    # Verify the loop vars appear in buffer-dim order matching ordered_vars —
    # ALL of them, exactly once. A transposed or partial mapping must not
    # produce a silently mis-laid-out tile.
    from .parallel import _split_affine_parallel_term

    def _dim_loop_var(e: Any) -> str | None:
        split = _split_affine_parallel_term(e, ov_set)
        return split[1] if split is not None else None

    found_names = [name for name in (_dim_loop_var(e) for e in expanded) if name is not None]
    ordered_found = [v for v in ov_list if v in found_names]
    if found_names != ordered_found or len(set(found_names)) != len(found_names):
        return None  # transposed / repeated var mapping — no silent mis-layout
    is_subset = found_names != ov_list
    if is_subset and len(found_names) != 1:
        return None  # multi-var subsets need N-D broadcast; not supported yet

    try:
        indices, elem_view = _compute_view_indices(
            tuple(base_tirs), tuple(tile_sizes), scope, builder, what=f"classified load from {buf_name!r}"
        )
    except _UnsupportedTileIRNode:
        return None  # scalar path raises its own (more specific) diagnostic

    from tilelang.tileir.emission_utils import _squeeze_shape

    squeezed = tuple(_squeeze_shape(list(tile_sizes)))
    load_result_ty = TileType(
        dtype=buf_val.type.dtype,
        shape=squeezed,
        space=MemSpace.REGISTER,
        layout=None,
    )
    load_op = builder.create(
        Load(
            src=buf_val,
            tile_shape=tuple(tile_sizes),
            indices=indices,
            elem_view=elem_view,
        ),
        result_types=(load_result_ty,),
    )
    loaded = load_op.results[0]
    if not is_subset:
        return loaded

    # The buffer is indexed by a SUBSET of the parallel vars (e.g.
    # ``scales_a[by*64 + i, k]`` under T.Parallel(i, j) — only ``i``): embed
    # the loaded tile into the full parallel tile shape, varying along the
    # found var's axis and broadcast along the others.
    full_shape = tuple(int(e) for e in (ordered_extents or []))
    if not all(isinstance(e, int) and e > 0 for e in full_shape):
        return None
    axis = ov_list.index(found_names[0])
    bcast_ty = TileType(
        dtype=buf_val.type.dtype,
        shape=full_shape,
        space=MemSpace.REGISTER,
        layout=None,
    )
    bcast_op = builder.create(
        Broadcast(
            src=loaded,
            src_shape=squeezed,
            target_shape=full_shape,
            axis=axis,
        ),
        result_types=(bcast_ty,),
    )
    return bcast_op.results[0]


def _is_whole_tile_access(expr: Any, scope: LoweringScope, loop_vars: set) -> bool:
    """True if every BufferLoad index is a parallel loop var / constant, or (for
    SHARED/REGISTER buffers) an affine combination of loop vars + scope scalars
    with at least one parallel loop var — i.e. a whole-tile register/shared read.
    """
    buf_name = expr.buffer.name
    # Check if all indices are loop vars (tile access) or constants
    all_tile_indices = all((isinstance(idx, _tir.Var) and idx.name in loop_vars) or (isinstance(idx, _tir.IntImm)) for idx in expr.indices)
    # For SHARED/REGISTER buffers the whole tile lives in registers and
    # Load.emit_mlir returns the entire tile-map entry regardless of the
    # index values.  So an affine index over loop vars / outer serial-loop
    # vars (e.g. ``x_frag[i, jj*4+j]`` where ``jj`` is an outer serial loop)
    # is still a whole-tile access — accept it here rather than dropping to
    # the scalar path, where the parallel loop variable is not bound. We require
    # at least one parallel loop var to appear so this only widens genuine
    # parallel-body tile reads.
    if not all_tile_indices:

        def _index_is_tile_affine(e: Any) -> bool:
            # Affine combination of constants, parallel loop vars, and
            # scope-bound scalars (outer serial-loop / block indices).
            if isinstance(e, _tir.IntImm):
                return True
            if isinstance(e, _tir.Var):
                return e.name in loop_vars or scope.lookup(e) is not None
            if isinstance(e, (_tir.Add, _tir.Sub, _tir.Mul)):
                return _index_is_tile_affine(e.a) and _index_is_tile_affine(e.b)
            return False

        def _references_loop_var(e: Any) -> bool:
            if isinstance(e, _tir.Var):
                return e.name in loop_vars
            if isinstance(e, (_tir.Add, _tir.Sub, _tir.Mul)):
                return _references_loop_var(e.a) or _references_loop_var(e.b)
            return False

        try:
            _buf_val_probe = scope.lookup_buffer(buf_name)
            _is_reg_or_shared = _buf_val_probe.type.space in (MemSpace.SHARED, MemSpace.REGISTER)
        except KeyError:
            _is_reg_or_shared = False
        if (
            _is_reg_or_shared
            and expr.indices
            and all(_index_is_tile_affine(idx) for idx in expr.indices)
            and any(_references_loop_var(idx) for idx in expr.indices)
        ):
            all_tile_indices = True
    return all_tile_indices


def _try_lower_whole_tile_load(expr: Any, scope: LoweringScope, builder: IRBuilder, ordered_vars, ordered_extents):
    """Lower a whole-tile SHARED/REGISTER (or extent-tiled GLOBAL) read.

    Handles the register/shared-fragment *row-slice* sub-case (``frag[k, j]``
    with scalar ``k``) and the outer-product *broadcast* sub-case (a lower-rank
    buffer embedded into the full parallel tile). Returns ``None`` (fall through)
    if the buffer is not in scope.
    """
    buf_name = expr.buffer.name

    # Try to emit a Load op for SHARED/REGISTER buffers.
    # We emit a Load TileOp (not just return the buffer Value) so that
    # the result Value is properly registered in emit_ctx.value_map.
    # Load.emit_mlir for SHARED/REGISTER returns ctx.get_tile(buf_val)
    # without emitting any MLIR instruction.
    try:
        buf_val = scope.lookup_buffer(buf_name)
        buf_shape = tuple(buf_val.type.shape)

        # Row-slice of a register/shared fragment: when some access dims are
        # indexed by a *scalar* (a serial-loop var or constant) rather than a
        # parallel/ordered var — e.g. ``frag[k, j]`` inside ``for k in serial``
        # with ``for j in T.Parallel`` — extract just that row instead of
        # returning the whole fragment tile. Returning the whole tile would
        # broaden a downstream accumulator (``acc[j] += frag[k,j]``) to the
        # fragment's shape and silently drop the loop accumulation (see the
        # loop-carry guard in ops.py). This is flash_decode's split-K combine.
        if buf_val.type.space in (MemSpace.SHARED, MemSpace.REGISTER) and ordered_vars and len(expr.indices) == len(buf_shape):
            from .parallel import _try_divide_expr

            _var_extent = dict(zip(ordered_vars, ordered_extents or []))

            def _has_ordered_var(e: Any) -> bool:
                if isinstance(e, _tir.Var):
                    return e.name in _var_extent
                for _attr in ("a", "b"):
                    _c = getattr(e, _attr, None)
                    if _c is not None and _has_ordered_var(_c):
                        return True
                return False

            _keep_shape: list[int] = []
            _keep_vars: list[str] = []
            _extract_shape: list[int] = []
            _extract_indices: list[Any] = []
            _ok = True
            for _d, _idx in enumerate(expr.indices):
                if isinstance(_idx, _tir.Var) and _idx.name in ordered_vars:
                    # Use the var's (possibly extent-clamped) parallel extent
                    # when it is smaller than the buffer dim: under an
                    # ``if i < K`` range clamp, ``acc[i, j]`` reads only the
                    # top K rows — returning the full tile would broaden the
                    # store value (reshape element-count mismatch downstream).
                    _ev = _var_extent.get(_idx.name)
                    if isinstance(_ev, int) and 0 < _ev < buf_shape[_d]:
                        _keep_shape.append(_ev)
                        _extract_shape.append(_ev)
                    else:
                        _keep_shape.append(buf_shape[_d])
                        _extract_shape.append(buf_shape[_d])
                    _keep_vars.append(_idx.name)
                    _extract_indices.append(0)
                    continue
                if isinstance(_idx, _tir.IntImm) or (isinstance(_idx, _tir.Var) and scope.lookup(_idx) is not None):
                    # Scalar-indexed dim: slice to extent 1 at this index.
                    _extract_shape.append(1)
                    _extract_indices.append(_lower_attr_expr(_idx, scope, builder))
                    continue
                # Strided sub-tile dim: Add(scalar_base, parallel_var) — e.g.
                # ``x_frag[i, jj*4 + j]`` with serial ``jj`` and parallel ``j``
                # (extent 4) is the (…, 4) tile at tile-granular index
                # base/extent = jj along this dim. ct.extract indexes tiles,
                # so the base must divide exactly by the var extent.
                _matched = False
                if isinstance(_idx, _tir.Add):
                    for _base, _var in ((_idx.a, _idx.b), (_idx.b, _idx.a)):
                        if not (isinstance(_var, _tir.Var) and _var.name in _var_extent):
                            continue
                        _ev = _var_extent[_var.name]
                        if not (isinstance(_ev, int) and _ev > 0) or _has_ordered_var(_base):
                            break
                        _q = _try_divide_expr(_base, _ev)
                        if _q is None:
                            break
                        _qv = _q if isinstance(_q, int) else _lower_attr_expr(_q, scope, builder)
                        _keep_shape.append(_ev)
                        _keep_vars.append(_var.name)
                        _extract_shape.append(_ev)
                        _extract_indices.append(_qv)
                        _matched = True
                        break
                if not _matched:
                    _ok = False
                    break
            if _ok and _keep_shape and _extract_shape != list(buf_shape):
                _slice_ty = TileType(
                    dtype=buf_val.type.dtype,
                    shape=tuple(_keep_shape),
                    space=MemSpace.REGISTER,
                    layout=None,
                )
                _slice_op = builder.create(
                    Load(src=buf_val, tile_shape=tuple(_extract_shape), indices=tuple(_extract_indices)),
                    result_types=(_slice_ty,),
                )
                _slice_val = _slice_op.results[0]

                # A serial index removes its buffer dimension from the loaded
                # tile, but the remaining dimensions still live on their
                # original T.Parallel axes.  For example, ``a[k, i]`` and
                # ``b[k, j]`` under ``T.Parallel(i, j)`` are shaped ``(I,)``
                # and ``(J,)`` after slicing; they must be embedded as
                # ``(I, 1)`` and ``(1, J)`` before elementwise arithmetic.
                # Returning the raw slices makes the generic emitter attempt
                # the invalid same-rank broadcast ``tile<I> -> tile<J>``.
                if ordered_vars is not None and ordered_extents is not None:
                    _full_shape = tuple(int(e) for e in ordered_extents)
                    _mapped_axes = [ordered_vars.index(v) for v in _keep_vars]
                    if (
                        tuple(_keep_shape) != _full_shape
                        and len(_keep_shape) == len(_mapped_axes)
                        and len(set(_mapped_axes)) == len(_mapped_axes)
                    ):
                        _reshape_shape = [1] * len(_full_shape)
                        for _size, _axis in zip(_keep_shape, _mapped_axes):
                            _reshape_shape[_axis] = _size
                        _broadcast_ty = TileType(
                            dtype=buf_val.type.dtype,
                            shape=_full_shape,
                            space=MemSpace.REGISTER,
                            layout=None,
                        )
                        return builder.create(
                            Broadcast(
                                src=_slice_val,
                                src_shape=tuple(_keep_shape),
                                target_shape=_full_shape,
                                axis=_mapped_axes[0],
                                reshape_shape=tuple(_reshape_shape),
                            ),
                            result_types=(_broadcast_ty,),
                        ).results[0]
                return _slice_val

        # For GLOBAL buffers accessed inside T.Parallel, use the
        # T.Parallel loop extents (ordered_extents) as the tile_shape, not the
        # full buffer shape.  Buffer dims like 257 are not powers of two and
        # would be rejected by cuda_tile.  The parallel extent (e.g. 128) is
        # always a power of two (tile block size) and matches what the store
        # side uses.  For SHARED/REGISTER buffers, Load.emit_mlir ignores
        # tile_shape (it just returns the tile map entry), so this change is
        # safe for those paths too.
        if ordered_extents is not None and len(ordered_extents) == len(buf_shape) and buf_val.type.space == MemSpace.GLOBAL:
            load_tile_shape = tuple(oe if (isinstance(oe, int) and oe > 0) else d for d, oe in zip(buf_shape, ordered_extents))
        else:
            load_tile_shape = buf_shape
        load_result_ty = (
            TileType(
                dtype=buf_val.type.dtype,
                shape=load_tile_shape,
                space=MemSpace.REGISTER,
                layout=None,
            )
            if load_tile_shape != buf_shape
            else buf_val.type
        )
        load_op = builder.create(
            Load(src=buf_val, tile_shape=load_tile_shape, indices=tuple(0 for _ in load_tile_shape)),
            result_types=(load_result_ty,),
        )
        # Re-sync buf_shape to the actual loaded tile shape for the broadcast
        # logic below (which checks len(ordered_extents) > len(buf_shape)).
        buf_shape = load_tile_shape
        loaded_val = load_op.results[0]

        # Transposed whole-tile read: a SHARED/REGISTER fragment indexed by the
        # parallel vars in a PERMUTED order (``x_local[j, i]`` inside
        # ``for i, j in T.Parallel(...)``).  The loaded tile carries the
        # buffer's axis order, but the parallel context lays tiles out in
        # ``ordered_vars`` order, so permute the axes to match.  Without this
        # the transpose is silently dropped — a square fragment passes the
        # shape-check and MISCOMPILES; a rectangular one trips the downstream
        # broadcast shape-check.  Read-side mirror of ``Store.val_perm``.
        if (
            buf_val.type.space in (MemSpace.SHARED, MemSpace.REGISTER)
            and ordered_vars is not None
            and len(expr.indices) == len(buf_shape) == len(ordered_vars)
            and all(isinstance(ix, _tir.Var) and ix.name in ordered_vars for ix in expr.indices)
        ):
            _idx_names = [ix.name for ix in expr.indices]
            if len(set(_idx_names)) == len(_idx_names):  # a genuine permutation
                # ct.permute: result axis k reads source axis perm[k]; we want
                # result axis k to hold ordered_vars[k].
                perm = tuple(_idx_names.index(ov) for ov in ordered_vars)
                if perm != tuple(range(len(perm))):
                    permuted_shape = tuple(buf_shape[p] for p in perm)
                    perm_ty = TileType(
                        dtype=buf_val.type.dtype,
                        shape=permuted_shape,
                        space=MemSpace.REGISTER,
                        layout=None,
                    )
                    loaded_val = builder.create(
                        Permute(src=loaded_val, perm=perm),
                        result_types=(perm_ty,),
                    ).results[0]
                    buf_shape = permuted_shape

        # Embed lower-rank buffer tiles in the axes selected by their parallel
        # indices, then broadcast them to the full parallel-loop tile shape.
        # This preserves outer-product semantics such as a row tile indexed by
        # ``i`` and a column tile indexed by ``j`` in a two-dimensional loop.
        if ordered_vars is not None and ordered_extents is not None and len(ordered_extents) > len(buf_shape):
            # The full parallel tile is nD (n = len(ordered_vars))
            # but the buffer is mD (m < n). Need to reshape + broadcast.
            full_shape = list(ordered_extents)
            reshape_shape = [1] * len(full_shape)
            shape_ok = True
            par_axis_used = -1  # which axis in the full tile this buffer maps to
            for buf_axis, idx in enumerate(expr.indices):
                if isinstance(idx, _tir.Var) and idx.name in ordered_vars:
                    par_axis = ordered_vars.index(idx.name)
                    reshape_shape[par_axis] = buf_shape[buf_axis]
                    par_axis_used = par_axis
                elif isinstance(idx, _tir.IntImm):
                    pass  # constant 0 index → stays 1 in reshape
                else:
                    shape_ok = False
                    break
            if shape_ok and reshape_shape != full_shape and par_axis_used >= 0:
                # Emit Broadcast: reshape the 1D tile to [1,…,extent,…,1]
                # at par_axis_used, then broadcast to full_shape.
                broadcast_result_ty = TileType(
                    dtype=buf_val.type.dtype,
                    shape=tuple(full_shape),
                    space=MemSpace.REGISTER,
                    layout=None,
                )
                bcast_op = builder.create(
                    Broadcast(
                        src=loaded_val,
                        src_shape=tuple(buf_shape),
                        target_shape=tuple(full_shape),
                        axis=par_axis_used,
                        # Multi-axis embed shape (e.g. weights (4,32) →
                        # (1,4,32)); the single-axis form only places ONE
                        # buffer dim and mis-shapes rank≥2 sources.
                        reshape_shape=tuple(reshape_shape),
                    ),
                    result_types=(broadcast_result_ty,),
                )
                loaded_val = bcast_op.results[0]

        return loaded_val
    except KeyError:
        pass  # fall through to scalar Load
    return None


def _try_lower_block_strided_load(expr: Any, scope: LoweringScope, builder: IRBuilder, loop_vars: set, ordered_vars, ordered_extents):
    """Lower a GLOBAL block-strided access ``buf[bx*N + i]`` (the canonical
    vec_add / T.Parallel pattern): tile_shape = loop extents, partition index =
    block_part / tile_dim. Returns ``None`` (fall through) on any non-match.
    """
    buf_name = expr.buffer.name

    # Block-strided tile access pattern
    # Recognise ``buf[block_part + loop_var]`` (or ``buf[loop_var + block_part]``)
    # in each dimension where ``loop_var ∈ loop_vars``.  This is the canonical
    # vec_add / T.Parallel access pattern ``A[bx * N + i]`` where ``bx`` is a
    # block-level scalar and ``N = len(T.Parallel(N))``.
    #
    # When matched we emit a tile-shaped Load with:
    #   tile_shape = (extent_of_loop_var, ...)
    #   indices    = (partition_index, ...)  [= block_part / tile_dim]
    #
    # The partition_index is computed by lowering the block_part expression
    # as a scalar and dividing by the tile_dim (via TIR Floor-div or Python int
    # division when the divisor is a known constant).
    if ordered_vars is not None and ordered_extents is not None and len(expr.indices) == len(ordered_extents):
        try:
            buf_val = scope.lookup_buffer(buf_name)
            # Only meaningful for GLOBAL buffers — SHARED/REGISTER access is
            # already handled by the all_tile_indices path above.
            if buf_val.type.space == MemSpace.GLOBAL:
                load_tile_shape: list[int] = []
                partition_tir_exprs: list[Any] = []
                pattern_ok = True
                for idx_expr, ov, oe in zip(expr.indices, ordered_vars, ordered_extents):
                    # Case 0: let-bound Var — look up its original TIR expression
                    # so that "idx = bx*N+i" is expanded to its definition and
                    # treated the same as Case 2 below.
                    resolved_idx_expr = idx_expr
                    if isinstance(idx_expr, _tir.Var) and idx_expr.name not in loop_vars:
                        tir_binding = scope.get_scalar_expr_binding(idx_expr)
                        if tir_binding is not None:
                            resolved_idx_expr = tir_binding
                    idx_expr = resolved_idx_expr
                    # Case 1: pure loop var → tile dimension = extent, partition = 0.
                    if isinstance(idx_expr, _tir.Var) and idx_expr.name in loop_vars:
                        load_tile_shape.append(oe)
                        partition_tir_exprs.append(_tir.IntImm("int32", 0))
                        continue
                    # Case 2: Add(block_part, loop_var) or Add(loop_var, block_part).
                    if isinstance(idx_expr, _tir.Add):
                        a, b = idx_expr.a, idx_expr.b

                        # Determine which side contains the loop var.
                        def _has_loop_var(e: Any) -> bool:
                            """True if any sub-expression is a loop var."""
                            if isinstance(e, _tir.Var) and e.name in loop_vars:
                                return True
                            if isinstance(e, _tir.Add):
                                return _has_loop_var(e.a) or _has_loop_var(e.b)
                            if isinstance(e, _tir.Mul):
                                return _has_loop_var(e.a) or _has_loop_var(e.b)
                            if isinstance(e, _tir.Sub):
                                return _has_loop_var(e.a) or _has_loop_var(e.b)
                            return False

                        if _has_loop_var(a) and not _has_loop_var(b):
                            iota_part, block_part = a, b
                        elif _has_loop_var(b) and not _has_loop_var(a):
                            iota_part, block_part = b, a
                        else:
                            pattern_ok = False
                            break
                        # Validate iota_part is exactly the loop var.
                        if not (isinstance(iota_part, _tir.Var) and iota_part.name == ov):
                            pattern_ok = False
                            break

                        # Compute partition index = block_part / tile_dim.
                        # If block_part is Mul(block_var, IntImm(N)) and N==oe, use block_var.
                        def _extract_partition_idx(bp: Any, tile_dim: int) -> Any | None:
                            """Return TIR expr for partition index, or None on failure."""
                            # Pattern: bp = scalar_expr * tile_dim → partition = scalar_expr.
                            if isinstance(bp, _tir.Mul):
                                if isinstance(bp.b, _tir.IntImm) and int(bp.b) == tile_dim:
                                    return bp.a
                                if isinstance(bp.a, _tir.IntImm) and int(bp.a) == tile_dim:
                                    return bp.b
                            # Pattern: bp is integer multiple of tile_dim.
                            if isinstance(bp, _tir.IntImm):
                                v = int(bp)
                                if v % tile_dim == 0:
                                    return _tir.IntImm("int32", v // tile_dim)
                            # Fallback: divide by tile_dim with floordiv.
                            try:
                                return _tir.floordiv(bp, _tir.IntImm("int32", tile_dim))
                            except (TypeError, ValueError):
                                return None

                        par_tir = _extract_partition_idx(block_part, oe)
                        if par_tir is None:
                            pattern_ok = False
                            break
                        load_tile_shape.append(oe)
                        partition_tir_exprs.append(par_tir)
                        continue
                    # No pattern matched for this dimension.
                    pattern_ok = False
                    break

                if pattern_ok and load_tile_shape:
                    load_shape_tuple = tuple(load_tile_shape)
                    # Build result TileType with the load tile shape.
                    load_result_ty = TileType(
                        dtype=buf_val.type.dtype,
                        shape=load_shape_tuple,
                        space=MemSpace.REGISTER,
                        layout=None,
                    )
                    # Lower each partition index expression as a scalar.
                    lowered_indices: list[Any] = []
                    for par_tir_expr in partition_tir_exprs:
                        lowered_indices.append(_lower_attr_expr(par_tir_expr, scope, builder))
                    load_op = builder.create(
                        Load(
                            src=buf_val,
                            tile_shape=load_shape_tuple,
                            indices=tuple(lowered_indices),
                        ),
                        result_types=(load_result_ty,),
                    )
                    return load_op.results[0]
        except (KeyError, TypeError, ValueError, _UnsupportedTileIRNode):
            # Expected non-matches: buffer not in scope, an unexpected TIR node
            # shape, or a partition index that can't be lowered → fall through to
            # the scalar load. Other errors (e.g. an AttributeError from a typo, or
            # a cuda_tile emit/verify failure) propagate rather than being masked.
            pass
    return None


def _try_lower_partial_index_load(expr: Any, scope: LoweringScope, builder: IRBuilder, ordered_vars, ordered_extents):
    """Lower a GLOBAL *partial-index* access where SOME dims are parallel loop
    vars and the rest are scalar (block / outer-loop) indices, broadcasting over
    any omitted parallel axis. Returns ``None`` (fall through) on any non-match.
    """
    buf_name = expr.buffer.name

    # Loop-variable dimensions use their parallel extent; scalar dimensions
    # use a unit tile extent and their lowered expression as partition index.
    if ordered_vars is not None and ordered_extents is not None and len(ordered_vars) > 0 and len(expr.indices) > len(ordered_extents):
        try:
            buf_val = scope.lookup_buffer(buf_name)
            if buf_val.type.space == MemSpace.GLOBAL:
                # Verify that all ordered_vars appear exactly once among indices.
                lv_set = set(ordered_vars)
                found_loop_axes: dict[str, int] = {}  # loop_var → index axis
                for ax_i, idx_e in enumerate(expr.indices):
                    # Expand let-var aliases.
                    e = idx_e
                    if isinstance(idx_e, _tir.Var) and idx_e.name not in lv_set:
                        tb = scope.get_scalar_expr_binding(idx_e)
                        if tb is not None:
                            e = tb
                    if isinstance(e, _tir.Var) and e.name in lv_set:
                        found_loop_axes[e.name] = ax_i

                # Accept the buffer being indexed by a SUBSET of the parallel
                # loop vars (the rest are broadcast over).  e.g. glse[bz,by,k]
                # inside T.Parallel(num_split, 128): only `k` indexes glse; `j`
                # is broadcast.  found_loop_axes must be non-empty and each found
                # loop var must appear exactly once.
                if 1 <= len(found_loop_axes) <= len(ordered_vars):
                    # Build the N-D tile_shape and partition indices.
                    p_tile_shape: list[int] = []
                    p_partition_exprs: list[Any] = []
                    shape_ok = True
                    ov_to_extent = dict(zip(ordered_vars, ordered_extents))
                    for idx_e in expr.indices:
                        # Expand let-var.
                        e = idx_e
                        if isinstance(idx_e, _tir.Var) and idx_e.name not in lv_set:
                            tb = scope.get_scalar_expr_binding(idx_e)
                            if tb is not None:
                                e = tb
                        if isinstance(e, _tir.Var) and e.name in lv_set:
                            # Loop-var dimension: tile extent = ordered_extent.
                            p_tile_shape.append(ov_to_extent[e.name])
                            p_partition_exprs.append(_tir.IntImm("int32", 0))
                        else:
                            # Scalar dimension: tile extent = 1, partition = scalar_idx.
                            p_tile_shape.append(1)
                            p_partition_exprs.append(idx_e)  # original expr for lowering

                    if shape_ok and p_tile_shape:
                        p_shape_tuple = tuple(p_tile_shape)
                        squeezed_p = tuple(d for d in p_shape_tuple if d != 1)
                        if not squeezed_p:
                            squeezed_p = (1,)
                        load_result_ty_p = TileType(
                            dtype=buf_val.type.dtype,
                            shape=squeezed_p,
                            space=MemSpace.REGISTER,
                            layout=None,
                        )
                        lowered_p_indices = []
                        for par_expr in p_partition_exprs:
                            _p_val = _lower_attr_expr(par_expr, scope, builder)
                            # A partition index must be SCALAR. A tile-valued
                            # expr here (e.g. a let-bound TopkIndices tile) is
                            # a data-dependent gather — fall through so the
                            # gather pattern handles it instead of silently
                            # scalarizing element 0.
                            if getattr(getattr(_p_val, "type", None), "shape", ()) != ():
                                return None
                            lowered_p_indices.append(_p_val)
                        load_op_p = builder.create(
                            Load(
                                src=buf_val,
                                tile_shape=p_shape_tuple,
                                indices=tuple(lowered_p_indices),
                            ),
                            result_types=(load_result_ty_p,),
                        )
                        loaded_p = load_op_p.results[0]
                        if len(found_loop_axes) == len(ordered_vars):
                            # Buffer indexed by every parallel loop var — the
                            # loaded tile already matches the parallel tile shape.
                            return loaded_p
                        # SUBSET: the buffer omits some parallel loop var(s); embed
                        # the loaded tile at the used loop var's parallel axis and
                        # broadcast over the missing axes.  Only the single-used-var
                        # case (1D loaded tile) is handled here.
                        if len(squeezed_p) == 1:
                            found_name = next(iter(found_loop_axes))
                            bcast_axis = ordered_vars.index(found_name)
                            full_shape = tuple(ordered_extents)
                            bcast_ty = TileType(
                                dtype=buf_val.type.dtype,
                                shape=full_shape,
                                space=MemSpace.REGISTER,
                                layout=None,
                            )
                            bcast_op = builder.create(
                                Broadcast(
                                    src=loaded_p,
                                    src_shape=squeezed_p,
                                    target_shape=full_shape,
                                    axis=bcast_axis,
                                ),
                                result_types=(bcast_ty,),
                            )
                            return bcast_op.results[0]
                        # multi-used-var subset: not handled — fall through.
        except (KeyError, TypeError, ValueError, _UnsupportedTileIRNode):
            # Expected non-matches (see _try_lower_block_strided_load) → fall
            # through to the scalar load; unexpected errors propagate.
            pass
    return None


def _lower_cast_tile(expr: Any, builder: IRBuilder, recurse) -> Value:
    """Lower a tile-context ``Cast`` to a shaped ``Cast`` TileIR op."""
    src_val = recurse(expr.value)
    tgt_dtype_str = _canonical_dtype_str(expr.dtype)
    src_dtype_str = _canonical_dtype_str(getattr(expr.value, "dtype", ""))
    scalar_ty = _tir_dtype_to_tile_type(tgt_dtype_str)
    result_ty = TileType(dtype=scalar_ty.dtype, shape=src_val.type.shape, space=MemSpace.REGISTER, layout=None)
    cast_op = builder.create(
        Cast(src=src_val, dtype=tgt_dtype_str, src_dtype=src_dtype_str),
        result_types=(result_ty,),
    )
    return cast_op.results[0]


def _lower_select_tile(expr: Any, builder: IRBuilder, recurse) -> Value:
    """Lower a tile-context ``Select`` (e.g. ``T.abs``); all three sub-exprs in tile mode."""
    cond_val = recurse(expr.condition)
    true_val = recurse(expr.true_value)
    false_val = recurse(expr.false_value)
    return _make_select(cond_val, true_val, false_val, expr, builder)


def _lower_binop_tile(
    expr: Any, binary_fn: Any, scope: LoweringScope, builder: IRBuilder, loop_vars: set, ordered_vars, ordered_extents, recurse
) -> Value:
    """Lower a tile-context binary op (FMA-fused when possible)."""
    fma_val = _try_lower_fma_tile(
        expr,
        scope,
        builder,
        loop_vars,
        ordered_vars=ordered_vars,
        ordered_extents=ordered_extents,
    )
    if fma_val is not None:
        return fma_val
    lhs = recurse(expr.a)
    rhs = recurse(expr.b)
    # uint signedness: "max"/"min"/comparisons need the OPERAND's
    # signedness (expr.dtype is "bool" for comparisons) -- see
    # `_make_elementwise`'s docstring.
    unsigned = str(getattr(expr.a, "dtype", "")).startswith("uint")
    return _make_elementwise(binary_fn, (lhs, rhs), expr, builder, unsigned=unsigned)


def _lower_var_tile(expr: Any, scope: LoweringScope, loop_vars: set, ordered_vars, ordered_extents, builder: IRBuilder):
    """Lower a tile-context ``Var``: let-bound value, or a loop-var ``Iota`` index.

    Returns ``None`` to fall through to the scalar ``lower_expr``.
    """
    bound = scope.lookup(expr)
    if bound is not None:
        return bound
    # Causal-mask case: a loop variable becomes an axis-indexed Iota tile so that
    # ``m_idx * 64 + i`` produces a tile where (row, col) holds ``m_idx * 64 + row``.
    if expr.name in loop_vars and ordered_vars is not None and ordered_extents is not None and expr.name in ordered_vars:
        axis = ordered_vars.index(expr.name)
        tile_shape = tuple(ordered_extents)
        result_ty = TileType(dtype="int32", shape=tile_shape, space=MemSpace.REGISTER, layout=None)
        iota_op = builder.create(Iota(tile_shape=tile_shape, axis=axis), result_types=(result_ty,))
        return iota_op.results[0]
    return None


def _lower_call_tile(expr: Any, scope: LoweringScope, builder: IRBuilder, ordered_vars, ordered_extents, recurse):
    """Lower a tile-context ``Call`` (if_then_else, atomic-return/-load, unary/binary
    intrinsics, reinterpret). Returns a ``Value``, or ``None`` to fall through."""
    from .tile_ops import _extract_region_buffer_name, _extract_static_int

    op_name = _op_name_from_call(expr)
    result_dtype_str = str(getattr(expr, "dtype", "float32"))
    result_ty = _tir_dtype_to_tile_type(result_dtype_str)

    # Return-value atomic anywhere in an RHS expression: emit the (masked)
    # gather-form AtomicRMW and use its previous-value tile as the result.
    # The enclosing parallel-if predicate travels via scope._parallel_mask.
    if "atomic_add_ret_elem_op" in op_name and ordered_vars is not None:
        from .parallel import _try_lower_ret_atomic_value

        _ret = _try_lower_ret_atomic_value(
            expr, scope, builder, set(ordered_vars), ordered_vars, ordered_extents, getattr(scope, "_parallel_mask", None)
        )
        if _ret is not None:
            return _ret

    # T.if_then_else in tile mode → Select over (possibly mask-shaped) tiles.
    if op_name in ("tir.if_then_else",) and len(expr.args) == 3:
        cond_val = recurse(expr.args[0])
        true_val = recurse(expr.args[1])
        false_val = recurse(expr.args[2])
        return _make_select(cond_val, true_val, false_val, expr, builder)

    # Atomic-return calls embedded in a buffer_store RHS → AtomicRMW(return_prev=True).
    _ATOMIC_RET_OPS = {
        "tl.atomic_add_ret_elem_op": "add",
        "tl.atomic_max_ret_elem_op": "max",
        "tl.atomic_min_ret_elem_op": "min",
    }
    if op_name in _ATOMIC_RET_OPS and len(expr.args) >= 2:
        _ret_kind = _ATOMIC_RET_OPS[op_name]
        _dst_name = _extract_region_buffer_name(expr.args[0])
        _val_name = _extract_region_buffer_name(expr.args[1])
        if _dst_name is not None and _val_name is not None:
            try:
                _dst_val = scope.lookup_buffer(_dst_name)
                _val_val = scope.lookup_buffer(_val_name)
                _mo = 0
                if len(expr.args) >= 3:
                    _mo_raw = _extract_static_int(expr.args[2])
                    if _mo_raw is not None:
                        _mo = _mo_raw
                _ret_tile_shape = tuple(ordered_extents) if ordered_extents else tuple(_dst_val.type.shape)
                _ret_result_ty = TileType(
                    dtype=_dst_val.type.dtype,
                    shape=_ret_tile_shape,
                    space=MemSpace.REGISTER,
                    layout=None,
                )
                _atomic_op = builder.create(
                    AtomicRMW(dst=_dst_val, val=_val_val, kind=_ret_kind, memory_order=_mo, return_prev=True),
                    result_types=(_ret_result_ty,),
                )
                if _atomic_op.results:
                    return _atomic_op.results[0]
            except KeyError:
                # Buffer not in scope → fall through to the scalar path. A real
                # emit/verify failures must not be swallowed: degrading a confirmed
                # atomic to a non-atomic scalar op would silently drop atomicity.
                pass

    # Atomic load call embedded in a buffer_store / let RHS.
    if op_name == "tl.atomic_load_elem_op" and len(expr.args) >= 2:
        _src_name = _extract_region_buffer_name(expr.args[0])
        if _src_name is not None:
            try:
                _src_val = scope.lookup_buffer(_src_name)
                _mo = 2  # default ACQUIRE
                _mo_raw = _extract_static_int(expr.args[1])
                if _mo_raw is not None:
                    _mo = _mo_raw
                _load_tile_shape = tuple(ordered_extents) if ordered_extents else tuple(_src_val.type.shape)
                _load_result_ty = TileType(
                    dtype=_src_val.type.dtype,
                    shape=_load_tile_shape,
                    space=MemSpace.REGISTER,
                    layout=None,
                )
                _load_op = builder.create(
                    AtomicLoad(src=_src_val, memory_order=_mo),
                    result_types=(_load_result_ty,),
                )
                if _load_op.results:
                    return _load_op.results[0]
            except KeyError:
                # Buffer not in scope → fall through to the scalar path; a real
                # emit failure must surface rather than silently becoming a
                # non-atomic load.
                pass

    unary_fn = _UNARY_CALL_FN.get(op_name)
    if unary_fn is not None and len(expr.args) == 1:
        arg_val = recurse(expr.args[0])
        return _make_elementwise(unary_fn, (arg_val,), expr, builder)

    # Binary intrinsic calls (bitwise / shift / pow) — recurse args in tile mode.
    if op_name in ("tir.bitwise_and", "tir.bitwise_or", "tir.bitwise_xor") and len(expr.args) == 2:
        lhs = recurse(expr.args[0])
        rhs = recurse(expr.args[1])
        _bw_fn = {"tir.bitwise_and": "andi", "tir.bitwise_or": "ori", "tir.bitwise_xor": "xori"}[op_name]
        return _make_elementwise(_bw_fn, (lhs, rhs), expr, builder)
    if op_name == "tir.bitwise_not" and len(expr.args) == 1:
        arg_val = recurse(expr.args[0])
        _bn_dtype = str(getattr(expr, "dtype", "")) or result_dtype_str
        _bn_fn = "not" if _bn_dtype in ("bool", "int1", "uint1") else "bitwise_not"
        return _make_elementwise(_bn_fn, (arg_val,), expr, builder)
    if op_name in ("tir.shift_left", "tir.shift_right") and len(expr.args) == 2:
        lhs = recurse(expr.args[0])
        rhs = recurse(expr.args[1])
        if op_name == "tir.shift_left":
            fn = "shl"
        else:
            expr_dtype_str = str(getattr(expr, "dtype", "")) or ""
            fn = "shr_unsigned" if expr_dtype_str.startswith("uint") else "shr"
        return _make_elementwise(fn, (lhs, rhs), expr, builder)
    if op_name in ("tir.pow", "tl.pow_of_int") and len(expr.args) == 2:
        lhs = recurse(expr.args[0])
        rhs = recurse(expr.args[1])
        return _make_elementwise("pow", (lhs, rhs), expr, builder)
    if op_name == "tir.atan2" and len(expr.args) == 2:
        lhs = recurse(expr.args[0])
        rhs = recurse(expr.args[1])
        return _make_elementwise("atan2", (lhs, rhs), expr, builder)
    # tir.reinterpret maps to a bit reinterpretation, not a numeric cast.
    if op_name == "tir.reinterpret" and len(expr.args) == 1:
        src_val = recurse(expr.args[0])
        src_dtype_str = str(getattr(expr.args[0], "dtype", ""))
        cast_op = builder.create(
            Cast(src=src_val, dtype=result_dtype_str, src_dtype=src_dtype_str, bitcast=True),
            result_types=(replace(result_ty, shape=src_val.type.shape),),
        )
        return cast_op.results[0]

    return None  # fall through to scalar lower_expr


def _try_lower_fma_tile(
    expr: Any,
    scope: LoweringScope,
    builder: IRBuilder,
    loop_vars: set,
    *,
    ordered_vars: list[str] | None = None,
    ordered_extents: list[int] | None = None,
) -> Value | None:
    """FMA detection for tile-level expressions.

    Gated behind scope.fast_math: only emit FMA when fast_math is enabled.
    """
    # Gate FMA fusion behind fast_math; precise mode uses separate mul+add.
    if not scope.fast_math:
        return None

    def _is_mul(e: Any) -> bool:
        return isinstance(e, _tir.Mul)

    def _is_float_tir(e: Any) -> bool:
        dtype_str = str(getattr(e, "dtype", ""))
        return any(dtype_str.startswith(p) for p in ("float", "bfloat"))

    def _r(e: Any) -> Value:
        return _lower_tile_level_expr(
            e,
            scope,
            builder,
            loop_vars,
            ordered_vars=ordered_vars,
            ordered_extents=ordered_extents,
        )

    if isinstance(expr, _tir.Add):
        if _is_mul(expr.a) and _is_float_tir(expr.a) and _is_float_tir(expr.b):
            a = _r(expr.a.a)
            b = _r(expr.a.b)
            c = _r(expr.b)
            return _make_elementwise("fma", (a, b, c), expr, builder)
        if _is_mul(expr.b) and _is_float_tir(expr.b) and _is_float_tir(expr.a):
            a = _r(expr.b.a)
            b = _r(expr.b.b)
            c = _r(expr.a)
            return _make_elementwise("fma", (a, b, c), expr, builder)
    return None

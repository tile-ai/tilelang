"""SemanticIR -> TileIR lowering: T.Parallel loop lowering.

Provides the parallel-loop machinery: detection (``_is_parallel_for``), loop-var
collection, the tile-level body walker, store-dim classification, partition-index
computation, and the region-shape / divide helpers.  Imports the shared
foundation, scalar ``lower_expr``, tile-level lowering, and ``lower_stmt``; the
``_extract_region_buffer_name`` / ``_extract_static_int`` helpers from
``tile_ops`` are imported function-locally to keep the import graph acyclic.
"""

from __future__ import annotations

from typing import Any

from tvm import tirx as _tir

from tilelang.tileir.ir.builder import IRBuilder
from tilelang.tileir.errors import TileIRLoweringError, _UnsupportedTileIRNode
from tilelang.tileir.ir.types import MemSpace, TileType
from tilelang.tileir.ir.value import Value
from tilelang.tileir.ir.ops import (
    AtomicLoad,
    AtomicRMW,
    AtomicStore,
    Broadcast,
    Constant,
    Elementwise,
    Loop,
    Select,
    Store,
)
from tilelang.tileir.semantic import SemanticStmt

from ._base import (
    LoweringScope,
    _binding_var,
    _make_placeholder,
    _next_power_of_two,
    _scalar_i32_type,
)
from .expr import lower_expr, _lower_attr_expr
from .tile_level import _lower_tile_level_expr
from .stmt import lower_stmt, _lower_loop_bounds


def _is_parallel_for(stmt: SemanticStmt) -> bool:
    """Return True if *stmt* is a T.Parallel (ForKind.PARALLEL) for loop."""
    if stmt.kind != "for":
        return False
    # ForKind.PARALLEL = 1.  The SemanticIR serialises "kind" in two ways:
    #   - string "1" (integer form), or
    #   - string "parallel" (name form).
    attrs = dict(stmt.attrs)
    kind_val = attrs.get("kind")
    if kind_val is None and stmt.loop_kind is not None:
        try:
            return stmt.loop_kind == _tir.ForKind.PARALLEL
        except Exception:
            pass
        return False
    # Accept string "parallel" as well as the numeric "1"
    kind_str = str(kind_val).lower().strip()
    if kind_str == "parallel":
        return True
    try:
        return int(kind_str) == 1  # ForKind.PARALLEL = 1
    except (ValueError, TypeError):
        return False


def _collect_parallel_loop_vars(
    stmt: SemanticStmt,
) -> tuple[set, SemanticStmt | None, list[str], list[int]]:
    """Peel nested T.Parallel for loops and collect loop var names and extents.

    Descends through the PARALLEL for nest until a non-PARALLEL stmt is found.

    Returns
    -------
    (loop_var_names: set[str], innermost_body: SemanticStmt or None,
     ordered_vars: list[str], ordered_extents: list[int])

    ``ordered_vars`` is the loop-variable names in outer→inner order (e.g.
    ``["i", "j"]`` for a 2-D parallel nest).  ``ordered_extents`` is the
    corresponding loop extent for each var (e.g. ``[64, 64]``).  These are
    used by ``_lower_tile_level_expr`` to emit ``Iota`` ops for loop-variable
    references in causal-mask / if_then_else expressions.
    """
    loop_vars: set[str] = set()
    ordered_vars: list[str] = []
    ordered_extents: list[int] = []
    cur = stmt
    while _is_parallel_for(cur):
        attrs = dict(cur.attrs)
        var_name = attrs.get("var", "")
        loop_vars.add(var_name)
        ordered_vars.append(var_name)
        # Extract loop extent (integer).
        extent_raw = cur.loop_extent if cur.loop_extent is not None else attrs.get("extent")
        extent_int = 0
        try:
            if isinstance(extent_raw, int):
                extent_int = extent_raw
            else:
                if isinstance(extent_raw, (_tir.IntImm, str)):
                    extent_int = int(extent_raw)
        except (TypeError, ValueError):
            extent_int = 0
        ordered_extents.append(extent_int)
        if not cur.children:
            return loop_vars, None, ordered_vars, ordered_extents
        cur = cur.children[0]
    return loop_vars, cur, ordered_vars, ordered_extents


def _lower_parallel_loop(stmt: SemanticStmt, scope: LoweringScope, builder: IRBuilder) -> None:
    """Lower a ``T.Parallel`` nest as collective tile operations."""
    loop_vars, body, ordered_vars, ordered_extents = _collect_parallel_loop_vars(stmt)
    if body is None:
        return  # empty parallel body

    # Pad each extent to the next power of 2.  CUDA Tile IR requires all tile
    # dimensions to be powers of two.  Without padding, iota/store ops with
    # non-pow2 extents (e.g. 24) fail
    # at MLIR creation time with "all dimensions must be powers of two".  The padded
    # tile is larger than the logical loop range; out-of-bounds lanes write to
    # padding positions (which the global store bounds-check will discard) or remain
    # in REGISTER-space padding (no visible effect).
    padded_extents = [_next_power_of_two(e) if isinstance(e, int) and e > 0 else e for e in ordered_extents]

    with scope.frame():
        # Add loop var names as available in scope (they are tile-range vars;
        # if an expression uses them via scope.lookup, the tile-buffer path above
        # takes precedence so the Var lookup is usually not hit).
        # No TileIR binding is needed because _lower_tile_level_expr handles
        # BufferLoad before variable lookup.
        _lower_parallel_body(
            body,
            loop_vars,
            scope,
            builder,
            ordered_vars=ordered_vars,
            ordered_extents=padded_extents,
        )


def _lower_parallel_let(stmt, scope, builder, loop_vars, ordered_vars, ordered_extents, recurse_body):
    """Bind a let-var in a parallel body (tile->scalar fallback) and record any
    block-strided partition expr for later make_partition_view."""
    attrs = dict(stmt.attrs)
    var_name = attrs.get("var", "")
    bind_key = _binding_var(stmt)
    if bind_key is None:
        bind_key = var_name
    tir_value: Any = stmt.value if stmt.value is not None else attrs.get("value")
    if tir_value is not None:
        try:
            bound_val = _lower_tile_level_expr(
                tir_value,
                scope,
                builder,
                loop_vars,
                ordered_vars=ordered_vars,
                ordered_extents=ordered_extents,
            )
        except _UnsupportedTileIRNode:
            # Value not expressible as a tile → try the scalar path. If that is
            # also unsupported, leave the let-var unbound (a downstream use then
            # fails loudly). Only _UnsupportedTileIRNode is a real fallback signal;
            # other errors (typos / emit failures) propagate.
            try:
                bound_val = _lower_attr_expr(tir_value, scope, builder)
            except _UnsupportedTileIRNode:
                bound_val = None
        if var_name and bound_val is not None:
            scope.bind(bind_key, bound_val)
            # Also persist the raw TIR expression in _tir_expr_bindings so that
            # _lower_tile_level_expr can pattern-match let-bound index vars.
            if tir_value is not None:
                scope._tir_expr_bindings[bind_key] = tir_value
            # Block-strided access tracking.
            # When a let-var is set to ``block_part + loop_var`` (the canonical
            # T.Parallel index pattern, e.g. ``idx = bx * N + i``), record the
            # partition TIR expression in scope so that subsequent BufferLoad /
            # BufferStore on this variable can compute the correct partition index
            # for ``make_partition_view`` instead of falling back to scalar loads.
            if var_name and ordered_vars is not None and ordered_extents is not None and tir_value is not None:
                try:
                    # Single-dimension case: check if tir_value = block_part + loop_var.
                    if isinstance(tir_value, _tir.Add) and len(ordered_vars) == 1 and len(ordered_extents) == 1:
                        a, b = tir_value.a, tir_value.b
                        ov, oe = ordered_vars[0], ordered_extents[0]

                        def _is_loop_var_only(e: Any) -> bool:
                            return isinstance(e, _tir.Var) and e.name == ov

                        if _is_loop_var_only(b) and not _is_loop_var_only(a):
                            block_part = a
                        elif _is_loop_var_only(a) and not _is_loop_var_only(b):
                            block_part = b
                        else:
                            block_part = None
                        if block_part is not None:
                            # Compute partition TIR = block_part / oe.
                            # Pattern: block_part = Mul(v, IntImm(oe)) → partition = v.
                            if isinstance(block_part, _tir.Mul) and isinstance(block_part.b, _tir.IntImm) and int(block_part.b) == oe:
                                partition_tir = block_part.a
                            elif isinstance(block_part, _tir.Mul) and isinstance(block_part.a, _tir.IntImm) and int(block_part.a) == oe:
                                partition_tir = block_part.b
                            elif isinstance(block_part, _tir.IntImm):
                                v = int(block_part)
                                if v % oe == 0:
                                    partition_tir = _tir.IntImm("int32", v // oe)
                                else:
                                    partition_tir = None
                            else:
                                # Generic: floordiv(block_part, oe).
                                try:
                                    partition_tir = _tir.floordiv(block_part, _tir.IntImm("int32", oe))
                                except (TypeError, ValueError):
                                    partition_tir = None
                            if partition_tir is not None:
                                # Lower the partition index to a scalar TileIR Value and store.
                                try:
                                    par_val = _lower_attr_expr(partition_tir, scope, builder)
                                    scope.bind(f"__part_idx__{var_name}", par_val)
                                except _UnsupportedTileIRNode:
                                    pass  # index not lowerable → skip the __part_idx__ cache
                except (ImportError, AttributeError, TypeError, ValueError, _UnsupportedTileIRNode):
                    pass  # block-strided partition detection didn't apply → skip the optimization
    if stmt.children:
        for child in stmt.children:
            recurse_body(child)
    return


def _lower_parallel_if(stmt, scope, builder, loop_vars, ordered_vars, ordered_extents, mask, recurse_body):
    """Lower an if/else in a parallel body: the (tile-wide) condition becomes a
    boolean mask predicating the branch stores (cuda_tile.if needs a scalar)."""
    attrs = dict(stmt.attrs)
    tir_cond = stmt.condition if stmt.condition is not None else attrs.get("condition")
    # Static extent-clamp guard: ``if var < K`` where ``var`` is a parallel
    # loop var and K is a static power-of-two ≤ its extent is a pure range
    # restriction, not a data-dependent predicate — lower the branch with the
    # var's extent CLAMPED to K instead of masking. This keeps tile shapes
    # aligned with the (smaller) destination buffers (e.g.
    # ``if bi_i < BS//split: dst16[bi_i, d] = src32[bi_i + s*16, d]``) and
    # enables tile-granular sub-tile extracts on the read side.  Only applies
    # without an else-branch (the complement is not representable as a clamp).
    if tir_cond is not None and len(stmt.children) == 1 and ordered_vars and ordered_extents:
        try:
            if isinstance(tir_cond, _tir.LT) and isinstance(tir_cond.a, _tir.Var) and isinstance(tir_cond.b, _tir.IntImm):
                _v_name = tir_cond.a.name
                _k = int(tir_cond.b)
                if _v_name in ordered_vars:
                    _axis = list(ordered_vars).index(_v_name)
                    _ext = ordered_extents[_axis]
                    if isinstance(_ext, int) and 0 < _k <= _ext and _k == _next_power_of_two(_k):
                        _clamped = list(ordered_extents)
                        _clamped[_axis] = _k
                        _lower_parallel_body(
                            stmt.children[0], loop_vars, scope, builder, ordered_vars=ordered_vars, ordered_extents=_clamped, mask=mask
                        )
                        return
        except ImportError:
            pass

    then_mask: Any = mask
    if tir_cond is not None:
        # The condition becomes a tile-wide mask for every store in the branch.
        # A dropped or weakened mask silently mis-guards the stores, so failures
        # propagate rather than degrading to a weaker mask.
        cond_tile = _lower_tile_level_expr(
            tir_cond,
            scope,
            builder,
            loop_vars,
            ordered_vars=ordered_vars,
            ordered_extents=ordered_extents,
        )
        # Combine with any outer mask: AND of both predicates.
        # We use a TileIR Elementwise("andi") op since the cuda_tile
        # module is not accessible in the sem_to_ir phase.
        if mask is not None:
            bool_type = TileType(
                dtype=cond_tile.type.dtype,
                shape=cond_tile.type.shape,
                space=MemSpace.REGISTER,
                layout=None,
            )
            combined = builder.create(
                Elementwise(fn="andi", inputs=(mask, cond_tile)),
                result_types=(bool_type,),
            )
            cond_tile = combined.results[0]
        then_mask = cond_tile
    # Lower then-branch with the combined mask
    if stmt.children:
        recurse_body(stmt.children[0], child_mask=then_mask)
    # else-branch: use the negated condition as mask so that REGISTER buffer
    # stores inside the else branch apply select(not cond, else_val, current).
    # This implements `if cond: x=a else: x=b` as two sequential masked stores:
    #   then: x = select(cond, a, x_init)
    #   else: x = select(not cond, b, x_after_then)
    # which correctly gives x = select(cond, a, b) when a/b are constants.
    if len(stmt.children) > 1:
        else_mask = mask  # default: use outer mask (e.g. None for top-level if)
        if then_mask is not None:
            # Negate then_mask with Elementwise("not"). This must succeed for
            # the else stores to be predicated correctly.
            bool_type = TileType(
                dtype=then_mask.type.dtype,
                shape=then_mask.type.shape,
                space=MemSpace.REGISTER,
                layout=None,
            )
            not_op = builder.create(
                Elementwise(fn="not", inputs=(then_mask,)),
                result_types=(bool_type,),
            )
            negated = not_op.results[0]
            # Combine with outer mask if present.
            if mask is not None:
                combined_not = builder.create(
                    Elementwise(fn="andi", inputs=(negated, mask)),
                    result_types=(bool_type,),
                )
                else_mask = combined_not.results[0]
            else:
                else_mask = negated
        recurse_body(stmt.children[1], child_mask=else_mask)
    return


def _try_lower_ret_atomic_value(
    tir_value: Any, scope: LoweringScope, builder: IRBuilder, loop_vars: set, ordered_vars, ordered_extents, mask
) -> Value | None:
    """Lower a return-value atomic used as an RHS: ``T.atomic_add(dst[...],
    v, return_prev=True)``.

    Emits a (masked) gather-form AtomicRMW returning the previous-value tile.
    Under a mask, masked lanes add the identity (0) — their previous values
    are garbage, which is fine because every consumer of the result is under
    the same predicate. Returns None when *tir_value* is not a ret-atomic.
    """
    if not isinstance(tir_value, _tir.Call):
        return None
    op_name = str(getattr(tir_value.op, "name", ""))
    if "atomic_add_ret_elem_op" not in op_name:
        return None
    if ordered_vars is None or ordered_extents is None:
        return None
    args = list(tir_value.args)
    dst_load = _extract_region_buffer_load(args[0])
    if dst_load is None:
        raise _UnsupportedTileIRNode(f"ret-atomic: cannot extract dst BufferLoad from {args[0]!r}.")
    dst_val = scope.lookup_buffer(dst_load.buffer.name)
    from .tile_level import _classify_gather_dims

    specs = _classify_gather_dims(list(dst_load.indices), scope, builder, loop_vars, ordered_vars, ordered_extents)
    if specs is None:
        raise _UnsupportedTileIRNode(f"ret-atomic: cannot classify dst indices {[str(i) for i in dst_load.indices]}.")
    gk, gv, ga, _ = specs
    val = _lower_tile_level_expr(args[1], scope, builder, loop_vars, ordered_vars=ordered_vars, ordered_extents=ordered_extents)
    if mask is not None:
        val = _apply_atomic_add_mask(val, mask, scope, builder, ordered_extents)
    full_shape = tuple(int(e) for e in ordered_extents)
    result_ty = TileType(dtype=dst_val.type.dtype, shape=full_shape, space=MemSpace.REGISTER, layout=None)
    op = builder.create(
        AtomicRMW(
            dst=dst_val,
            val=val,
            kind="add",
            memory_order=0,
            return_prev=True,
            gather_dim_kinds=gk,
            gather_dim_values=gv,
            gather_dim_axes=ga,
            tile_shape=full_shape,
        ),
        result_types=(result_ty,),
    )
    return op.results[0]


def _apply_atomic_add_mask(val: Value, mask: Value, scope: LoweringScope, builder: IRBuilder, eff_extents: list) -> Value:
    """Predicate an atomic-ADD value with select(mask, val, 0).

    Adding zero is the exact identity, so masked lanes perform a no-op RMW.
    The value is broadcast to the parallel extents first when scalar-like.
    """
    full_shape = tuple(int(e) for e in eff_extents)
    val_shape = tuple(val.type.shape)
    if val_shape != full_shape:
        numel = 1
        for d in val_shape:
            numel *= d
        if numel != 1:
            raise _UnsupportedTileIRNode(
                f"masked atomic add: value shape {val_shape} does not match the "
                f"parallel extents {full_shape} and is not scalar-broadcastable."
            )
        bcast_ty = TileType(dtype=val.type.dtype, shape=full_shape, space=MemSpace.REGISTER, layout=None)
        val = builder.create(
            Broadcast(src=val, src_shape=val_shape, target_shape=full_shape, axis=0),
            result_types=(bcast_ty,),
        ).results[0]
    scalar_zero_ty = TileType(dtype=val.type.dtype, shape=(), space=MemSpace.REGISTER, layout=None)
    zero_scalar = builder.create(Constant(value=0, dtype=val.type.dtype.name), result_types=(scalar_zero_ty,)).results[0]
    full_ty = TileType(dtype=val.type.dtype, shape=full_shape, space=MemSpace.REGISTER, layout=None)
    zero = builder.create(Broadcast(src=zero_scalar, src_shape=(), target_shape=full_shape, axis=0), result_types=(full_ty,)).results[0]
    return builder.create(
        Select(cond=mask, true_val=val, false_val=zero),
        result_types=(full_ty,),
    ).results[0]


def _extract_region_buffer_load(tir_arg):
    """Return the underlying BufferLoad node from a region/access_ptr/BufferLoad
    TIR arg (same patterns as ``_extract_region_buffer_name``), or None."""
    if isinstance(tir_arg, _tir.BufferLoad):
        return tir_arg
    if not isinstance(tir_arg, _tir.Call) or not tir_arg.args:
        return None
    inner = tir_arg.args[0]
    if isinstance(inner, _tir.BufferLoad):
        return inner
    return None


def _lower_parallel_atomic(stmt, scope, builder, loop_vars, ordered_vars, ordered_extents, mask=None) -> bool:
    """Lower an element-wise atomic store/load in a parallel body. Returns True
    if it emitted the op, False to fall through to generic lowering."""
    from .tile_ops import _extract_region_buffer_name, _extract_static_int

    _tn = str(dict(stmt.attrs).get("op", ""))
    _args = list(stmt.call_args)
    if _tn:
        # tl.atomic_{add,max,min}_elem_op: element-wise RMW over the parallel
        # nest. The dst BufferLoad indices mix parallel loop vars (tile dims)
        # with block offsets (partition indices), e.g.
        # ``C[by*128 + i, bx*128 + j]`` under ``T.Parallel(128, 128)`` is the
        # (128,128) tile at partition (by, bx). Dropping the offsets would
        # silently reduce every CTA into partition (0,0), so any pattern this
        # branch cannot classify must raise, never fall through.
        _is_elem_rmw = (
            "atomic_add_elem_op" in _tn
            or "atomic_addx2_elem_op" in _tn
            or "atomic_addx4_elem_op" in _tn
            or "atomic_max_elem_op" in _tn
            or "atomic_min_elem_op" in _tn
        ) and "_ret_" not in _tn
        if _is_elem_rmw and len(_args) >= 2:
            if "atomic_max" in _tn:
                _kind = "max"
            elif "atomic_min" in _tn:
                _kind = "min"
            else:
                _kind = "add"
            _dst_load = _extract_region_buffer_load(_args[0])
            if _dst_load is None:
                raise _UnsupportedTileIRNode(f"parallel atomic {_kind}: cannot extract dst BufferLoad from {_args[0]!r}.")
            _dst_name = _dst_load.buffer.name
            try:
                _dst_val = scope.lookup_buffer(_dst_name)
            except KeyError as exc:
                raise _UnsupportedTileIRNode(f"parallel atomic {_kind}: dst buffer lookup failed: {exc}") from exc
            _raw_shape = tuple(_dst_val.type.shape)
            # Vectorized atomics (addx2/addx4) step the innermost index by the
            # vector width and add `width` consecutive elements per iteration.
            # Tile equivalent: expand that var's extent by the width and drop
            # the Mul(var, width) scale from every index (dst AND val).
            _vec_w = 4 if "addx4" in _tn else (2 if "addx2" in _tn else 1)
            _eff_extents = list(ordered_extents or [])
            _val_node = _args[1]
            if _vec_w > 1:
                _scaled = _find_vec_scaled_var(_dst_load.indices, set(ordered_vars or []), _vec_w)
                if _scaled is None:
                    raise _UnsupportedTileIRNode(
                        f"parallel atomic {_kind} (x{_vec_w}): no parallel var scaled by "
                        f"{_vec_w} found in dst indices {[str(i) for i in _dst_load.indices]}."
                    )
                _vname = _scaled[0]
                _axis = list(ordered_vars).index(_vname)
                _eff_extents[_axis] = int(_eff_extents[_axis]) * _vec_w
                _indices_tir = [_unscale_vec_index(i, _vname, _vec_w) for i in _dst_load.indices]
                _val_bl = _extract_region_buffer_load(_args[1])
                if _val_bl is None:
                    raise _UnsupportedTileIRNode(f"parallel atomic {_kind} (x{_vec_w}): cannot extract val BufferLoad from {_args[1]!r}.")
                _val_node = _unscale_vec_index(_val_bl, _vname, _vec_w)
            else:
                _indices_tir = list(_dst_load.indices)
            if len(_indices_tir) != len(_raw_shape):
                raise _UnsupportedTileIRNode(
                    f"parallel atomic {_kind}: dst `{_dst_name}` has {len(_raw_shape)} dims but the store uses {len(_indices_tir)} indices."
                )
            _gather_specs = None
            _pindices: tuple = ()
            _elem_view = False
            _tile_sizes: list = []
            try:
                _tile_sizes, _base_tirs = _classify_store_dims(_indices_tir, ordered_vars, _eff_extents, _raw_shape)
                if not all(isinstance(s, int) and s > 0 for s in _tile_sizes):
                    raise _UnsupportedTileIRNode(
                        f"parallel atomic {_kind}: could not classify dst indices "
                        f"{[str(i) for i in _indices_tir]} into tile/partition dims "
                        f"(tile sizes {_tile_sizes})."
                    )
                _pindices, _elem_view = _compute_view_indices(
                    tuple(_base_tirs), tuple(_tile_sizes), scope, builder, what=f"parallel atomic {_kind}"
                )
            except _UnsupportedTileIRNode:
                # Data-dependent scatter (a dim's index is TILE-valued, e.g.
                # ``dKV[b, Indices[...], g, d] += v``): classify per-dim gather
                # specs and go through per-element pointers instead.
                from .tile_level import _classify_gather_dims

                _gather_specs = _classify_gather_dims(_indices_tir, scope, builder, loop_vars, ordered_vars, _eff_extents)
                if _gather_specs is None or not _gather_specs[3]:
                    raise
            # Lower the value expr in tile context (loud on failure — a dropped
            # or partition-0 atomic is a silent miscompile).
            _val_tile_val = _lower_tile_level_expr(
                _val_node,
                scope,
                builder,
                loop_vars,
                ordered_vars=ordered_vars,
                ordered_extents=_eff_extents,
            )
            _mo = 0
            if len(_args) >= 3:
                _mo_raw = _extract_static_int(_args[2])
                if _mo_raw is not None:
                    _mo = _mo_raw
            # A mask from an enclosing parallel-if must predicate the atomic.
            # atomic_rmw_tko has no mask operand; for ADD, select(mask, val, 0)
            # is an exact predicate (masked lanes add the identity). Other
            # kinds have no cheap identity here — reject loudly.
            if mask is not None:
                if _kind != "add":
                    raise _UnsupportedTileIRNode(
                        f"parallel atomic {_kind} under a mask: only 'add' supports identity-masking (select(mask, val, 0))."
                    )
                _val_tile_val = _apply_atomic_add_mask(_val_tile_val, mask, scope, builder, _eff_extents)
            if _gather_specs is not None:
                _gk, _gv, _ga, _ = _gather_specs
                builder.create(
                    AtomicRMW(
                        dst=_dst_val,
                        val=_val_tile_val,
                        kind=_kind,
                        memory_order=_mo,
                        gather_dim_kinds=_gk,
                        gather_dim_values=_gv,
                        gather_dim_axes=_ga,
                        # Scatter domain (the parallel extents): a scalar val
                        # (e.g. histogram `+= 1`) broadcasts to it at emit.
                        tile_shape=tuple(int(e) for e in _eff_extents),
                    )
                )
                return True
            builder.create(
                AtomicRMW(
                    dst=_dst_val,
                    val=_val_tile_val,
                    kind=_kind,
                    memory_order=_mo,
                    dst_indices=tuple(_pindices),
                    tile_shape=tuple(_tile_sizes),
                    elem_view=_elem_view,
                )
            )
            return True

        # tl.atomic_store_elem_op: args[0]=dst_ptr, args[1]=value_expr, args[2]=memory_order
        if "atomic_store_elem_op" in _tn and len(_args) >= 3:
            _dst_name = _extract_region_buffer_name(_args[0])
            if _dst_name is not None:
                try:
                    _dst_val = scope.lookup_buffer(_dst_name)
                    # Lower the value expr in tile context.
                    _val_tir = _args[1]
                    try:
                        _val_tile_val = _lower_tile_level_expr(
                            _val_tir,
                            scope,
                            builder,
                            loop_vars,
                            ordered_vars=ordered_vars,
                            ordered_extents=ordered_extents,
                        )
                    except _UnsupportedTileIRNode:
                        _val_tile_val = None  # not tile-lowerable → fall to generic lower_stmt
                    _mo = 0
                    _mo_raw = _extract_static_int(_args[2])
                    if _mo_raw is not None:
                        _mo = _mo_raw
                    if _val_tile_val is not None:
                        # The value tile is already lowered (a builder result), so it is
                        # registered in the value_map. Pass it directly as AtomicStore.val
                        # — AtomicStore.emit_mlir resolves the operand via tile_map /
                        # buffer_map / value_map — rather than staging it through a Store
                        # into a temporary REGISTER buffer first.
                        builder.create(AtomicStore(dst=_dst_val, val=_val_tile_val, memory_order=_mo))
                        return True
                except (KeyError, TypeError, ValueError, _UnsupportedTileIRNode):
                    pass  # specialized atomic-store didn't apply → generic lower_stmt (also atomic)

        # tl.atomic_load_elem_op: args[0]=src_ptr, args[1]=memory_order
        # (This case is usually handled via buffer_store RHS; here if it appears as Evaluate)
        if "atomic_load_elem_op" in _tn and len(_args) >= 2:
            _src_name = _extract_region_buffer_name(_args[0])
            if _src_name is not None:
                try:
                    _src_val = scope.lookup_buffer(_src_name)
                    _mo = 2
                    _mo_raw = _extract_static_int(_args[1])
                    if _mo_raw is not None:
                        _mo = _mo_raw
                    builder.create(AtomicLoad(src=_src_val, memory_order=_mo))
                    return True
                except (KeyError, TypeError, ValueError, _UnsupportedTileIRNode):
                    pass  # specialized atomic-load didn't apply → generic lower_stmt (also atomic)
    return False


def _lower_parallel_body(
    stmt: SemanticStmt,
    loop_vars: set,
    scope: LoweringScope,
    builder: IRBuilder,
    *,
    ordered_vars: list[str] | None = None,
    ordered_extents: list[int] | None = None,
    mask: Any = None,
) -> None:
    """Recursively process the body of a T.Parallel nest.

    ``mask`` is an optional tile-wide boolean (tile<Nxi1>) that is applied as
    a predicate to buffer stores inside this branch.  It is set when an
    ``if_stmt`` is encountered inside a parallel body — e.g. ``if idx < n``
    generates a tile-wide condition that must guard the conditional stores as
    a select rather than as a ``cuda_tile.if`` (which requires a scalar 0-D
    condition).
    """
    # Expose the current predicate to expression-level consumers (e.g. a
    # ret-atomic nested inside an RHS expression lowered via tile_level).
    scope._parallel_mask = mask

    # Deferred import to break the parallel <- tile_ops cycle: tile_ops imports
    # parallel, so these extractor helpers (in tile_ops) are imported here.

    # Helper to re-call with same ordered_vars/extents/mask.
    def _recurse_body(child_stmt: SemanticStmt, child_loop_vars: set = loop_vars, child_mask: Any = mask) -> None:
        _lower_parallel_body(
            child_stmt,
            child_loop_vars,
            scope,
            builder,
            ordered_vars=ordered_vars,
            ordered_extents=ordered_extents,
            mask=child_mask,
        )

    if stmt.kind == "seq":
        for child in stmt.children:
            _recurse_body(child)
        return

    if stmt.kind == "let":
        _lower_parallel_let(stmt, scope, builder, loop_vars, ordered_vars, ordered_extents, _recurse_body)
        return
    if stmt.kind == "buffer_store":
        _lower_parallel_buffer_store(
            stmt,
            loop_vars,
            scope,
            builder,
            ordered_vars=ordered_vars,
            ordered_extents=ordered_extents,
            mask=mask,
        )
        return

    if _is_parallel_for(stmt):
        # Nested parallel loop inside the body — peel further
        inner_vars, inner_body, inner_ov, inner_oe = _collect_parallel_loop_vars(stmt)
        if inner_body is not None:
            all_vars = loop_vars | inner_vars
            # Merge ordered_vars: outer first, then inner (for iota axis computation).
            combined_ov = (ordered_vars or []) + inner_ov
            combined_oe = (ordered_extents or []) + inner_oe
            with scope.frame():
                _lower_parallel_body(
                    inner_body,
                    all_vars,
                    scope,
                    builder,
                    ordered_vars=combined_ov,
                    ordered_extents=combined_oe,
                    mask=mask,
                )
        return

    if stmt.kind in ("block",):
        for child in stmt.children:
            _recurse_body(child)
        return

    if stmt.kind == "thread_extent":
        # ty/tz (extent 1) AttrStmts nested inside a SIMT threadIdx.x region:
        # bind the var to scalar 0 and KEEP the body in the parallel lane
        # context (falling back to lower_stmt would drop it). A >1 extent here
        # would be a 2D SIMT nest — not supported.
        _te_attrs = dict(stmt.attrs)
        _te_var = _te_attrs.get("var", "")
        _te_raw = _te_attrs.get("extent")
        try:
            _te_ext = int(_te_raw) if not hasattr(_te_raw, "value") else int(_te_raw.value)
        except (TypeError, ValueError):
            _te_ext = -1
        if _te_ext != 1:
            raise _UnsupportedTileIRNode(
                f"nested thread binding `{_te_var}` (extent {_te_raw!r}) inside a SIMT "
                "threadIdx.x region — multi-axis SIMT nests are not supported."
            )
        if _te_var:
            _zero_op = builder.create(Constant(value=0, dtype="int32"), result_types=(_scalar_i32_type(),))
            _te_key = _binding_var(stmt)
            scope.bind(_te_key if _te_key is not None else _te_var, _zero_op.results[0])
        for child in stmt.children:
            _recurse_body(child)
        return

    if stmt.kind == "for":
        # Serial loop nested inside the T.Parallel body (e.g. a per-tile
        # reduction ``for k in T.serial(n): mixes[j] += src[k, i, j]``):
        # emit a real Loop op whose body is lowered WITH the parallel tile
        # context, so parallel loop vars stay tile axes and fragment updates
        # are loop-carried by the emit-side tile-carry machinery.
        _lower_serial_for_in_parallel(
            stmt, loop_vars, scope, builder, ordered_vars=ordered_vars, ordered_extents=ordered_extents, mask=mask
        )
        return

    if stmt.kind == "if":
        _lower_parallel_if(stmt, scope, builder, loop_vars, ordered_vars, ordered_extents, mask, _recurse_body)
        return
    if stmt.kind == "atomic_rmw" and _lower_parallel_atomic(stmt, scope, builder, loop_vars, ordered_vars, ordered_extents, mask=mask):
        return
    # Fallback: lower as a normal stmt. Never silently skip — a dropped
    # reduce/copy/store inside a parallel body produces a numerically-wrong
    # kernel with no diagnostic. Surface the failure loudly.
    try:
        lower_stmt(stmt, scope, builder)
    except _UnsupportedTileIRNode:
        raise
    except Exception as exc:
        raise _UnsupportedTileIRNode(f"cannot lower stmt {stmt.kind!r} inside a parallel body: {exc}") from exc


def _find_vec_scaled_var(indices, ov_set: set, width: int) -> tuple | None:
    """Find the single parallel var used as ``Mul(var, width)`` in *indices*.

    Vectorized atomics (``atomic_addx2/x4``) step their innermost index by the
    vector width; the tile equivalent expands that var's extent by ``width``
    and drops the scale.  Returns ``(var_name,)`` or None.
    """
    found: set[str] = set()

    def _scan(e: Any) -> None:
        if isinstance(e, _tir.Mul):
            a, b = e.a, e.b
            for var, c in ((a, b), (b, a)):
                if isinstance(var, _tir.Var) and var.name in ov_set and isinstance(c, _tir.IntImm) and int(c) == width:
                    found.add(var.name)
                    return
        for _attr in ("a", "b"):
            _c = getattr(e, _attr, None)
            if _c is not None:
                _scan(_c)
        idxs = getattr(e, "indices", None)
        if idxs is not None:
            for i in idxs:
                _scan(i)

    for idx in indices:
        _scan(idx)
    if len(found) != 1:
        return None
    return (next(iter(found)),)


def _unscale_vec_index(e: Any, var_name: str, width: int) -> Any:
    """Rewrite ``Mul(var, width)`` → ``var`` throughout an index expression
    (the var's extent is expanded by ``width`` by the caller)."""
    if isinstance(e, _tir.Mul):
        a, b = e.a, e.b
        for var, c in ((a, b), (b, a)):
            if isinstance(var, _tir.Var) and var.name == var_name and isinstance(c, _tir.IntImm) and int(c) == width:
                return var
        return _unscale_vec_index(a, var_name, width) * _unscale_vec_index(b, var_name, width)
    if isinstance(e, _tir.Add):
        return _unscale_vec_index(e.a, var_name, width) + _unscale_vec_index(e.b, var_name, width)
    if isinstance(e, _tir.Sub):
        return _unscale_vec_index(e.a, var_name, width) - _unscale_vec_index(e.b, var_name, width)
    if isinstance(e, _tir.BufferLoad):
        return _tir.BufferLoad(e.buffer, [_unscale_vec_index(i, var_name, width) for i in e.indices])
    return e


def _split_affine_parallel_term(expr: Any, ov_set: set) -> tuple | None:
    """Split an additive chain into ``(base, var_name)``.

    Matches ``t₀ + … + loop_var + … + tₙ`` where exactly ONE term is a bare
    parallel loop var and every other term is loop-var-free; ``base`` is the
    sum of the remaining terms (``None`` when there are none, i.e. a pure
    loop var).  Returns ``None`` for anything else — e.g. scaled
    (``2*loop_var``) or repeated vars, which are not unit-stride affine.
    """
    terms: list = []

    def _collect(e: Any) -> None:
        if isinstance(e, _tir.Add):
            _collect(e.a)
            _collect(e.b)
        else:
            terms.append(e)

    _collect(expr)

    def _has_parallel_var(e: Any) -> bool:
        if isinstance(e, _tir.Var):
            return e.name in ov_set
        for _attr in ("a", "b", "args", "indices"):
            _c = getattr(e, _attr, None)
            if _c is None:
                continue
            if isinstance(_c, (list, tuple)) or type(_c).__name__ == "Array":
                if any(_has_parallel_var(x) for x in _c):
                    return True
            elif _has_parallel_var(_c):
                return True
        return False

    var_terms = [t for t in terms if isinstance(t, _tir.Var) and t.name in ov_set]
    if len(var_terms) != 1:
        return None
    rest = [t for t in terms if t is not var_terms[0]]
    if any(_has_parallel_var(t) for t in rest):
        return None
    if not rest:
        return None, var_terms[0].name
    base = rest[0]
    for t in rest[1:]:
        base = base + t
    return base, var_terms[0].name


def _lower_serial_for_in_parallel(
    stmt: SemanticStmt,
    loop_vars: set,
    scope: LoweringScope,
    builder: IRBuilder,
    *,
    ordered_vars: list[str] | None,
    ordered_extents: list[int] | None,
    mask: Any = None,
) -> None:
    """Lower a serial ``for`` nested inside a T.Parallel body.

    Reuses the generic loop scaffolding (``_lower_loop_bounds`` + Loop op) but
    walks the body with ``_lower_parallel_body`` so the enclosing parallel
    loop vars remain tile axes.  The loop induction var is bound as a scalar
    in a fresh scope frame, exactly like the generic serial ``for``.
    """
    attrs = dict(stmt.attrs)
    var_name = attrs.get("var", "")
    bind_key = _binding_var(stmt)
    if bind_key is None:
        bind_key = var_name
    loop_var = _make_placeholder(builder, _scalar_i32_type(), name=var_name)

    start_val, stop_val, is_pipelined = _lower_loop_bounds(stmt, scope, builder)
    if stop_val is None:
        stop_val = _make_placeholder(builder, _scalar_i32_type(), name="stop")

    with builder.block_scope(params=[loop_var]) as body_block, scope.frame():
        scope.bind(bind_key, loop_var)
        for child in stmt.children:
            _lower_parallel_body(child, loop_vars, scope, builder, ordered_vars=ordered_vars, ordered_extents=ordered_extents, mask=mask)

    builder.create(
        Loop(
            start=start_val,
            stop=stop_val,
            step=None,
            init=None,
            body=body_block,
            pipelined=is_pipelined,
        )
    )


def _classify_store_dims(
    store_indices_tir: Any,
    ordered_vars: list[str] | None,
    ordered_extents: list[int] | None,
    raw_shape: tuple,
) -> tuple[list[int], list[Any]]:
    """Classify each buffer dim as parallel or scalar for a T.Parallel store.

    Returns
    -------
    tile_sizes : list[int]
        Per-dim tile size: loop extent if parallel, 1 if scalar.
    partition_idx_tirs : list[Any | None]
        Per-dim partition TIR index: the scalar TIR expr for scalar dims,
        None for parallel dims (will use partition index 0).
    """
    ov_set = set(ordered_vars or [])
    ov_list = list(ordered_vars or [])
    oe_list = list(ordered_extents or [])
    # Build a var_name → extent mapping.
    var_to_extent: dict[str, int] = {}
    for vname, ext in zip(ov_list, oe_list):
        var_to_extent[vname] = ext

    def _contains_loop_var(expr: Any) -> bool:
        """Return True if expr contains any loop var."""
        if isinstance(expr, _tir.Var):
            return expr.name in ov_set
        for child_attr in ("a", "b"):
            child = getattr(expr, child_attr, None)
            if child is not None and _contains_loop_var(child):
                return True
        return False

    def _get_loop_var_extent(expr: Any) -> int | None:
        """If expr is an additive chain containing exactly one bare loop var
        (``base₀ + … + loop_var + … + baseₙ``), return the loop extent."""
        split = _split_affine_parallel_term(expr, ov_set)
        if split is None:
            return None
        return var_to_extent.get(split[1])

    def _get_scalar_part(expr: Any) -> Any | None:
        """For an additive chain with one bare loop var, return the sum of the
        remaining (loop-var-free) terms; None when there is no scalar part."""
        split = _split_affine_parallel_term(expr, ov_set)
        if split is None:
            return None
        return split[0]

    tile_sizes: list[int] = []
    base_idx_tirs: list[Any | None] = []
    ndim = len(raw_shape)
    try:
        indices = list(store_indices_tir)
    except TypeError:
        # Can't iterate: fall back to all-scalar
        return [1] * ndim, list(store_indices_tir) if store_indices_tir else [None] * ndim

    for ax in range(ndim):
        if ax >= len(indices):
            tile_sizes.append(1)
            base_idx_tirs.append(None)
            continue
        idx = indices[ax]
        ext = _get_loop_var_extent(idx)
        if ext is not None and ext > 0:
            # Parallel dim: tile size = loop extent, base = the scalar part of
            # Add(scalar, loop_var) (None → 0 for a pure loop var).
            tile_sizes.append(ext)
            base_idx_tirs.append(_get_scalar_part(idx))
        else:
            # Scalar dim: tile size = 1, base = this TIR expr
            tile_sizes.append(1)
            base_idx_tirs.append(idx)

    return tile_sizes, base_idx_tirs


def _compute_store_shape_and_indices(
    scope, builder, store_indices_tir, expanded_indices_tir, eff_ordered_vars, eff_ordered_extents, raw_shape
):
    """Compute ``(tile_shape, store_indices, elem_view)`` for a parallel buffer
    store from its exact per-dim parallel-vs-scalar classification.

    ``elem_view=True`` means ``store_indices`` are raw element offsets for a
    unit-stride strided view (misaligned bases, e.g. varlen ``cu_seqlens``);
    otherwise they are tile-granular partition indices."""
    if store_indices_tir is None or eff_ordered_vars is None or eff_ordered_extents is None or len(store_indices_tir) != len(raw_shape):
        raise TileIRLoweringError(
            "parallel store requires raw indices, ordered loop metadata, and "
            "matching buffer/index rank; refusing a positional partition-0 fallback."
        )

    tile_sizes, base_idx_tirs = _classify_store_dims(
        expanded_indices_tir,
        eff_ordered_vars,
        eff_ordered_extents,
        raw_shape,
    )
    if not all(isinstance(size, int) and size > 0 for size in tile_sizes):
        raise TileIRLoweringError(f"parallel store could not determine a positive static tile shape: {tile_sizes}.")

    tile_shape = tuple(tile_sizes)
    store_indices, elem_view = _compute_view_indices(tuple(base_idx_tirs), tile_shape, scope, builder, what="parallel store")
    return tile_shape, store_indices, elem_view


def _lower_parallel_buffer_store(
    stmt: SemanticStmt,
    loop_vars: set,
    scope: LoweringScope,
    builder: IRBuilder,
    *,
    ordered_vars: list[str] | None = None,
    ordered_extents: list[int] | None = None,
    mask: Any = None,
) -> None:
    """Emit a tile-level Store op for a buffer_store inside a T.Parallel loop.

    ``mask`` is an optional tile-wide boolean (tile<Nxi1>) from an enclosing
    ``if_stmt`` inside the parallel body (e.g. ``if idx < n``).  When present,
    the value is wrapped with ``ct.select(mask, val, zeros)`` so that
    out-of-bounds lanes write zero rather than raising a ``cuda_tile.if``
    error for non-scalar conditions.
    """
    attrs = dict(stmt.attrs)
    buf_name = attrs.get("buffer", "")
    try:
        buf_val = scope.lookup_buffer(buf_name)
    except KeyError as exc:
        raise TileIRLoweringError(f"sem_to_ir: parallel buffer_store references unknown buffer {buf_name!r}.") from exc

    tir_value: Any = stmt.value if stmt.value is not None else attrs.get("value")

    if tir_value is None:
        raise TileIRLoweringError(f"sem_to_ir: parallel buffer_store to {buf_name!r} has no value.")

    # Causal-mask case: if ordered_vars is not provided but
    # loop_vars is non-empty, try to infer ordered_vars from the buffer shape
    # so that Var("i"), Var("j") references in the RHS can be lowered as Iota.
    # The buffer's tile shape gives us the extents; the loop var names come from
    # loop_vars (converted to a list in encounter order — approximated by sorted).
    eff_ordered_vars = ordered_vars
    eff_ordered_extents = ordered_extents
    if eff_ordered_vars is None and loop_vars:
        buf_shape = buf_val.type.shape
        if buf_shape and len(loop_vars) == len(buf_shape):
            # Best-effort: sort loop_vars alphabetically (i < j < k in most kernels).
            eff_ordered_vars = sorted(loop_vars)
            eff_ordered_extents = list(buf_shape)
    if eff_ordered_vars is None or eff_ordered_extents is None:
        raise TileIRLoweringError(
            f"sem_to_ir: parallel buffer_store to {buf_name!r} has no ordered loop metadata; refusing to assume partition 0."
        )

    # Lower the RHS in tile mode. A return-value atomic anywhere in the RHS is
    # handled by tile_level's Call branch, with the predicate carried on
    # scope._parallel_mask.
    try:
        val = _lower_tile_level_expr(
            tir_value,
            scope,
            builder,
            loop_vars,
            ordered_vars=eff_ordered_vars,
            ordered_extents=eff_ordered_extents,
        )
    except _UnsupportedTileIRNode as tile_exc:
        # Value not expressible as a tile → try the scalar path. If that is
        # also unsupported, never silently drop the store (the output buffer
        # would never be written) — raise loudly. Unexpected errors propagate.
        try:
            val = _lower_attr_expr(tir_value, scope, builder)
        except _UnsupportedTileIRNode as attr_exc:
            raise _UnsupportedTileIRNode(
                f"parallel buffer_store to {buf_name!r}: cannot lower value {tir_value!r} (tile-level: {tile_exc}; attr: {attr_exc})"
            ) from attr_exc

    raw_shape = buf_val.type.shape

    # Extract store TIR indices (for partition classification).
    store_indices_tir = stmt.indices
    if not store_indices_tir:
        raise TileIRLoweringError(
            f"sem_to_ir: parallel buffer_store to {buf_name!r} has no raw TIR indices; refusing to assume partition 0."
        )
    if len(store_indices_tir) != len(raw_shape):
        raise TileIRLoweringError(
            f"sem_to_ir: parallel buffer_store to {buf_name!r} has index rank {len(store_indices_tir)} but buffer rank {len(raw_shape)}."
        )

    # Classify dims: parallel (loop var) vs. scalar (partition index)
    # When store_indices_tir is available, classify each buffer dim by inspecting
    # which TIR index expressions contain loop variables.
    # For Output[by, bx, i] in T.Parallel(512):
    #   dim 0 (by): scalar → tile_size=1, partition=by
    #   dim 1 (bx): scalar → tile_size=1, partition=bx
    #   dim 2 (i):  parallel → tile_size=512, partition=0
    # Expand let-bound index Vars to their TIR definitions before classifying.
    # A T.Parallel store like ``C[idx]`` with ``idx = bx*threads + i`` arrives
    # here with store_indices_tir = (Var("idx"),). ``idx`` is not a loop var, so
    # _classify_store_dims would mis-classify it as a scalar (tile_size=1) and the
    # 128-element value tile would then be reshaped to 1 element.  Expand each
    # non-loop-var index Var via scope.get_scalar_expr_binding so the classifier
    # sees the real ``Add(Mul(bx, threads), i)`` form.
    expanded_indices_tir = store_indices_tir
    if store_indices_tir is not None:
        try:
            _ov_set = set(eff_ordered_vars or [])
            expanded = []
            for idx in store_indices_tir:
                e = idx
                if isinstance(e, _tir.Var) and e.name not in _ov_set:
                    binding = scope.get_scalar_expr_binding(e)
                    if binding is not None:
                        e = binding
                expanded.append(e)
            expanded_indices_tir = tuple(expanded)
        except (ImportError, AttributeError, TypeError):
            expanded_indices_tir = store_indices_tir

    tile_shape, store_indices, elem_view = _compute_store_shape_and_indices(
        scope,
        builder,
        store_indices_tir,
        expanded_indices_tir,
        eff_ordered_vars,
        eff_ordered_extents,
        raw_shape,
    )
    # Transposed parallel store (``B_t[i_k, i_s] = B[i_s, i_k]``): when the
    # dst dims map the parallel vars in a PERMUTED order, the value tile
    # (laid out in ordered_vars order) must be permuted — a reshape would
    # scramble the data. dst axis k reads val axis perm[k].
    val_perm: tuple = ()
    if expanded_indices_tir is not None and eff_ordered_vars:
        _ov_set = set(eff_ordered_vars)
        _dim_vars = []
        for _idx in expanded_indices_tir:
            _split = _split_affine_parallel_term(_idx, _ov_set)
            _dim_vars.append(_split[1] if _split is not None else None)
        _found = [v for v in _dim_vars if v is not None]
        if len(_found) == len(eff_ordered_vars) and set(_found) == _ov_set and _found != list(eff_ordered_vars):
            val_perm = tuple(list(eff_ordered_vars).index(v) for v in _found)
    # Pass the mask Value (if any) to Store so emit_mlir can apply the
    # predicated-store path for out-of-bounds element suppression.
    store_op = Store(
        dst=buf_val,
        val=val,
        tile_shape=tile_shape,
        indices=store_indices,
        mask=mask,
        elem_view=elem_view,
        val_perm=val_perm,
    )
    builder.create(store_op)


def _extract_tir_region_indices(call_args: tuple[Any, ...], region_index: int) -> tuple | None:
    """Extract raw TIR PrimExpr indices from a ``tl.region`` call arg.

    ``stmt.call_args`` for a ``tile_op copy`` contains region calls of the form::

        tl.tileop.copy(tl.region(A[bx*64, k*64], 1, 64, 64),
                       tl.region(sa[0, 0], 2, 64, 64))

    This helper drills into ``call_args[region_index].args[0]``
    (the ``BufferLoad``) to return its ``.indices`` tuple of TIR PrimExpr
    objects.  Returns ``None`` if the source node does not have the expected
    shape.
    """
    try:
        region_call = call_args[region_index]
        buf_load = region_call.args[0]
        if hasattr(buf_load, "indices"):
            return tuple(buf_load.indices)
    except (IndexError, AttributeError):
        pass
    return None


def _compute_partition_indices(
    tir_indices: tuple | None,
    tile_shape: tuple,
    scope: LoweringScope,
    builder: IRBuilder,
    *,
    what: str = "partition view",
) -> tuple:
    """Compute exact tile-partition indices without a partition-0 fallback.

    ``TmaCopy`` cannot encode the element-offset mode supported by ``Copy``.
    Reuse the authoritative view-index computation, but reject any start that
    requires element offsets rather than silently addressing partition zero.
    """
    if tir_indices is None:
        raise _UnsupportedTileIRNode(f"{what}: raw TIR region indices are unavailable; refusing to assume partition 0.")
    if len(tir_indices) != len(tile_shape):
        raise _UnsupportedTileIRNode(f"{what}: region index rank {len(tir_indices)} does not match tile rank {len(tile_shape)}.")

    indices, elementwise = _compute_view_indices(tir_indices, tile_shape, scope, builder, what=what)
    if elementwise:
        raise _UnsupportedTileIRNode(
            f"{what} start offset is not exactly divisible by its tile extent; TmaCopy has no element-offset indexing mode."
        )
    return indices


def _compute_view_indices(
    tir_indices: tuple | None,
    tile_shape: tuple,
    scope: LoweringScope,
    builder: IRBuilder,
    *,
    what: str = "view",
) -> tuple[tuple, bool]:
    """Compute view indices without silently falling back to partition zero.

    Returns ``(indices, elementwise)``:

    - ``elementwise=False`` — every start offset divided exactly by its tile
      extent; ``indices`` are tile-granular partition indices for
      ``make_partition_view`` (index k addresses elements
      ``[k*extent, (k+1)*extent)``).
    - ``elementwise=True`` — some start offset is not statically divisible by
      the tile extent (e.g. varlen attention's ``cu_seqlens[b] + bx*block_M``
      bases); ``indices`` are raw ELEMENT offsets for ``make_strided_view``
      with unit traversal strides.

    Raises ``_UnsupportedTileIRNode`` when an index cannot be lowered at all —
    a silently-zeroed index reads/writes the wrong tile.
    """
    if tir_indices is None:
        return tuple(0 for _ in tile_shape), False

    quotients: list[Any] = []  # int | Value | ("expr", tir) | None (=needs element mode)
    raws: list[Any] = []  # int | Value | tir expr
    elementwise = False

    for tir_idx, extent in zip(tir_indices, tile_shape):
        if tir_idx is None:
            quotients.append(0)
            raws.append(0)
            continue
        if isinstance(tir_idx, (int, Value)):
            # Pre-lowered index (int constant or already a scalar Value).
            raws.append(tir_idx)
            if isinstance(tir_idx, int) and tir_idx % max(extent, 1) == 0:
                quotients.append(tir_idx // max(extent, 1))
            elif extent == 1:
                quotients.append(tir_idx)
            else:
                quotients.append(None)
                elementwise = True
            continue
        if isinstance(tir_idx, _tir.IntImm):
            v = int(tir_idx)
            raws.append(v)
            if extent > 0 and v % extent == 0:
                quotients.append(v // extent)
            else:
                quotients.append(None)
                elementwise = True
            continue

        effective_idx = tir_idx
        if isinstance(tir_idx, _tir.Var):
            if extent == 1:
                bound_val = scope.lookup(tir_idx)
                if bound_val is not None:
                    # extent 1: partition index == element index == the Value.
                    quotients.append(bound_val)
                    raws.append(bound_val)
                    continue
            bound_expr = scope.get_scalar_expr_binding(tir_idx)
            if bound_expr is not None:
                effective_idx = bound_expr

        raws.append(effective_idx)
        quotient_expr = _try_divide_expr(effective_idx, extent)
        if quotient_expr is None:
            quotients.append(None)
            elementwise = True
        elif isinstance(quotient_expr, int):
            quotients.append(quotient_expr)
        else:
            quotients.append(("expr", quotient_expr))

    def _lower_loud(expr: Any, dim: int) -> Any:
        try:
            return lower_expr(expr, scope, builder)
        except _UnsupportedTileIRNode as exc:
            raise _UnsupportedTileIRNode(f"{what}: cannot lower start offset `{expr}` for dim {dim}: {exc}") from exc

    if not elementwise:
        out: list[Any] = []
        for _dim, q in enumerate(quotients):
            if isinstance(q, tuple):
                try:
                    out.append(lower_expr(q[1], scope, builder))
                except _UnsupportedTileIRNode:
                    # Quotient unlowerable — retry the whole tuple in element mode.
                    elementwise = True
                    break
            else:
                out.append(q)
        if not elementwise:
            return tuple(out), False

    out = []
    for dim, r in enumerate(raws):
        if isinstance(r, (int, Value)):
            out.append(r)
        else:
            out.append(_lower_loud(r, dim))
    return tuple(out), True


def _prove_region_in_bounds(
    tir_indices: tuple | None,
    tile_shape: tuple,
    buffer_shape: tuple,
    scope: LoweringScope,
    builder: IRBuilder,
) -> bool:
    """Conservatively prove a rectangular region is inside a static buffer.

    The proof runs while raw TIR start-offset expressions and launch-axis
    ranges are still available.  The resulting boolean is carried by TileIR;
    emission does not re-discover bounds from lowered scalar arithmetic.
    """
    if len(tile_shape) != len(buffer_shape):
        return False
    if not all(isinstance(d, int) and d > 0 for d in (*tile_shape, *buffer_shape)):
        return False

    if tir_indices is None:
        return all(tile <= size for tile, size in zip(tile_shape, buffer_shape))
    if len(tir_indices) != len(tile_shape):
        return False

    try:
        from tvm import arith as _arith
        from tvm import ir as _ir

        analyzer = _arith.Analyzer()
        root = builder.module_block()
        for axis, extent in root.block_extents.items():
            var = scope._axis_bind_vars.get(axis)
            if isinstance(var, _tir.Var) and isinstance(extent, int) and extent > 0:
                analyzer.bind(var, _ir.Range.from_min_extent(0, extent))

        # Let-bound bases such as ``row = bx * block_m`` retain their raw TIR
        # expressions in the lowering scope and can participate in the proof.
        for var, expr in scope._tir_expr_bindings.items():
            if isinstance(var, _tir.Var) and hasattr(expr, "dtype"):
                try:
                    analyzer.bind(var, expr)
                except (TypeError, ValueError):
                    return False

        for start, tile, size in zip(tir_indices, tile_shape, buffer_shape):
            if isinstance(start, int):
                start = _tir.IntImm("int32", start)
            if not hasattr(start, "dtype"):
                return False
            if not analyzer.can_prove(start >= 0):
                return False
            if not analyzer.can_prove(start + tile <= size):
                return False
        return True
    except (AttributeError, TypeError, ValueError):
        return False


def _try_divide_expr(expr: Any, divisor: int) -> Any:
    """Return ``expr / divisor`` if expr is exactly divisible, else None.

    Handles ``IntImm * k``, ``k * IntImm``, ``Var`` (divisor == 1), and plain
    ``IntImm`` cases.  Returns an ``int`` for constant quotients and a TIR
    ``PrimExpr`` for symbolic ones.  Returns ``None`` when the division is not
    exact or the expression structure is unrecognised.
    """
    if divisor <= 0:
        return None

    if isinstance(expr, _tir.IntImm):
        val = int(expr)
        if val % divisor == 0:
            return val // divisor
        return None

    if divisor == 1:
        return expr

    if isinstance(expr, _tir.Mul):
        # Case: a * b where one operand is an IntImm = divisor → the other
        a, b = expr.a, expr.b
        if isinstance(b, _tir.IntImm) and int(b) == divisor:
            return a
        if isinstance(a, _tir.IntImm) and int(a) == divisor:
            return b
        # Case: a * (k * divisor) — e.g. bx * 128 / 64 = bx * 2
        if isinstance(b, _tir.IntImm) and int(b) % divisor == 0:
            factor = int(b) // divisor
            if factor == 1:
                return a
            return _tir.Mul(a, _tir.IntImm("int32", factor))
        if isinstance(a, _tir.IntImm) and int(a) % divisor == 0:
            factor = int(a) // divisor
            if factor == 1:
                return b
            return _tir.Mul(_tir.IntImm("int32", factor), b)

    if isinstance(expr, _tir.FloorMod):
        # FloorMod(x, m) is always in [0, m-1].  When m <= divisor the integer
        # quotient floor((m-1)/divisor) == 0, so FloorMod(x, m) / divisor == 0.
        # This handles the paged-KV pattern: k * block_N % block_size / block_N == 0
        # (when block_N == block_size, FloorMod(k*64, 64) is in [0,63] → / 64 = 0).
        try:
            if isinstance(expr.b, _tir.IntImm) and int(expr.b) <= divisor:
                return 0
        except Exception:
            pass

    if isinstance(expr, _tir.Add):
        # Case: a + b where both are divisible by divisor
        q_a = _try_divide_expr(expr.a, divisor)
        q_b = _try_divide_expr(expr.b, divisor)
        if q_a is not None and q_b is not None:
            if isinstance(q_a, int) and isinstance(q_b, int):
                return q_a + q_b
            if isinstance(q_a, int):
                q_a = _tir.IntImm("int32", q_a)
            if isinstance(q_b, int):
                q_b = _tir.IntImm("int32", q_b)
            return _tir.Add(q_a, q_b)

    return None


def _region_tile_shape(region_shape: tuple, buf_type_shape: tuple) -> tuple:
    """Compute a full-rank tile_shape for a copy region.

    The region shape may contain non-int TIR expressions for dynamic dims (e.g.
    ``kv_end - kv_start``).  The original filter ``isinstance(d, int)`` dropped
    such dims, reducing rank (e.g. 4D region → 3D tile_shape) and causing
    ``make_partition_view`` to fail with a rank mismatch on 4D TensorViews.

    Iterate over region_shape and for each dim:
      - If int: use it directly.
      - If non-int (TIR expr): fall back to the corresponding buffer dim from
        buf_type_shape (which is always static for GLOBAL buffers in TileLang).
        If buf_type_shape is also non-static, use 1 as a safe sentinel.

    This preserves rank: a 4D region with one dynamic dim produces a 4D tile_shape.
    """
    result: list[int] = []
    for i, d in enumerate(region_shape):
        if isinstance(d, int) and d > 0:
            result.append(d)
        else:
            # Non-int dimension: use the buffer's static shape for this axis.
            fallback = buf_type_shape[i] if i < len(buf_type_shape) else 1
            if isinstance(fallback, int) and fallback > 0:
                result.append(fallback)
            else:
                result.append(1)
    return tuple(result)

"""SemanticIR -> TileIR lowering: statement dispatch + control-flow handlers.

Provides ``lower_stmt`` (the ``IMPL`` dispatcher) and the control-flow / scoping
handlers registered via ``@impl`` (seq, block, thread_extent, swizzle,
reduce_scope, warp_specialize, let, for, while, if, buffer_store).  Imports the
shared foundation and scalar ``lower_expr``; the parallel-loop entry points it
calls are imported function-locally to keep the import graph acyclic.
"""

from __future__ import annotations

import re
from typing import Any

from tvm import tirx as _tir

from tilelang.tileir.ir.builder import IRBuilder
from tilelang.tileir.errors import (
    TileIRLoweringError,
    TileIRLoweringNotImplementedError,
    _UnboundScopeVariable,
    _UnsupportedTileIRNode,
)
from tilelang.tileir.ir.value import Block, Value
from tilelang.tileir.ir.ops import Break, Constant, Elementwise, IfElse, Loop, Select, Store
from tilelang.tileir.semantic import SemanticStmt

from ._base import (
    IMPL,
    LoweringScope,
    impl,
    _binding_var,
    _make_placeholder,
    _scalar_bool_type,
    _scalar_i32_type,
)
from .expr import lower_expr, _lower_attr_expr, _make_elementwise


# stmt dispatchers


def lower_stmt(stmt: SemanticStmt, scope: LoweringScope, builder: IRBuilder) -> None:
    """Dispatch to IMPL[stmt.kind], raising _UnsupportedTileIRNode if missing."""
    handler = IMPL.get(stmt.kind)
    if handler is None:
        raise _UnsupportedTileIRNode(f"Unsupported SemanticStmt kind: {stmt.kind!r}")
    handler(stmt, scope, builder)


# Kind handlers


@impl("seq")
def _lower_seq(stmt: SemanticStmt, scope: LoweringScope, builder: IRBuilder) -> None:
    for child in stmt.children:
        lower_stmt(child, scope, builder)


@impl("block")
def _lower_block(stmt: SemanticStmt, scope: LoweringScope, builder: IRBuilder) -> None:
    with scope.frame():
        for child in stmt.children:
            lower_stmt(child, scope, builder)


@impl("thread_extent")
def _lower_thread_extent(stmt: SemanticStmt, scope: LoweringScope, builder: IRBuilder) -> None:
    # Bind the thread/block index var so that expressions using threadIdx.*/blockIdx.*
    # can be lowered (instead of hitting the unbound-Var raise in lower_expr).
    attrs = dict(stmt.attrs)
    var_name = attrs.get("var", "")
    tag = attrs.get("tag", "")  # e.g. "blockIdx.x", "threadIdx.y"

    # An INNER SIMT thread binding — an explicit ``T.thread_binding(threadIdx.*)``
    # inside a T.Kernel whose launch nest already bound this axis — is
    # semantically a single-axis T.Parallel over the thread lanes: lower the
    # body with the parallel tile machinery (masks, affine chains, masked
    # element scatters) instead of a scalar placeholder.
    _INNER_THREAD_AXIS = {"threadIdx.x": "tx", "threadIdx.y": "ty", "threadIdx.z": "tz"}
    _axis = _INNER_THREAD_AXIS.get(tag)
    if _axis is not None and _axis in builder.block.thread_extents and var_name:
        outer_extent = builder.block.thread_extents[_axis]
        raw_extent = attrs.get("extent")
        try:
            inner_extent = int(raw_extent) if not hasattr(raw_extent, "value") else int(raw_extent.value)
        except (TypeError, ValueError) as exc:
            raise _UnsupportedTileIRNode(f"inner thread binding `{var_name}` ({tag}): non-static extent {raw_extent!r}.") from exc
        if inner_extent != outer_extent:
            raise _UnsupportedTileIRNode(
                f"inner thread binding `{var_name}` ({tag}): extent {inner_extent} != kernel thread extent {outer_extent}."
            )
        from .parallel import _lower_parallel_body

        with scope.frame():
            for child in stmt.children:
                _lower_parallel_body(
                    child,
                    {var_name},
                    scope,
                    builder,
                    ordered_vars=[var_name],
                    ordered_extents=[inner_extent],
                )
        return
    if var_name:
        # Represent as a typed i32 placeholder Value — there is no MLIR context at
        # this stage; the value is an abstract SSA id that MLIR emission wires up.
        idx_val = _make_placeholder(builder, _scalar_i32_type(), name=var_name)
        # Bind by the TIR Var (object identity) so a same-named user Var
        # (e.g. a T.Persistent `bx` coordinate) never aliases this block axis.
        bind_key = _binding_var(stmt)
        if bind_key is None:
            bind_key = var_name
        scope.bind(bind_key, idx_val)
        # Record block index Values in block.index_values so emit_module can bind
        # them to the MLIR get_tile_block_id() components.
        # Use the tag ("blockIdx.x/y/z") to detect block axes — var_name varies
        # per kernel (e.g. "bx", "by", "tile_m") but tag is always canonical.
        _BLOCK_TAG_TO_AXIS_KEY = {
            "blockIdx.x": "bx",
            "blockIdx.y": "by",
            "blockIdx.z": "bz",
        }
        axis_key = _BLOCK_TAG_TO_AXIS_KEY.get(tag)
        if axis_key is not None:
            # Remember the binding key so the swizzle pass can identity-rebind.
            scope._axis_bind_vars[axis_key] = bind_key
            builder.block.index_values[axis_key] = idx_val
            # Also store the static extent so _lower_threadblock_swizzle_pattern can
            # compute the swizzled block ID arithmetic (grid_x * grid_y product).
            raw_extent = attrs.get("extent")
            if raw_extent is not None:
                try:
                    extent_val = int(raw_extent) if not hasattr(raw_extent, "value") else int(raw_extent.value)
                    builder.block.block_extents[axis_key] = extent_val
                except (TypeError, ValueError):
                    pass
        # Also record threadIdx.* vars so emit_module can bind them to
        # ct.iota(extent, Int32) tiles — giving each thread its real lane
        # index for per-thread SIMT scatter kernels (e.g. dequant_gemm).
        _THREAD_TAG_TO_AXIS_KEY = {
            "threadIdx.x": "tx",
            "threadIdx.y": "ty",
            "threadIdx.z": "tz",
        }
        thread_key = _THREAD_TAG_TO_AXIS_KEY.get(tag)
        if thread_key is not None:
            builder.block.index_values[thread_key] = idx_val
            # Store the static extent so emit_module can size the iota.
            raw_extent = attrs.get("extent")
            if raw_extent is not None:
                try:
                    extent_val = int(raw_extent) if not hasattr(raw_extent, "value") else int(raw_extent.value)
                    builder.block.thread_extents[thread_key] = extent_val
                except (TypeError, ValueError):
                    pass
            # Record this var_name as a thread-index variable so
            # _is_thread_index_only_cond can erase warp-specialize guards regardless
            # of the user-chosen variable name (e.g. "tid" in minference vs "tx").
            if var_name:
                scope._thread_var_names.add(var_name)
        # Note: reserved launch-axis names ("bx"/"by"/"bz"/"tx"/"ty"/"tz")
        # need no separate bookkeeping here -- `scope.bind(var_name, idx_val)`
        # above already registers them in `_binding_stack`, and `_lower_let`'s
        # Bind-deferral branch rejects ANY deferred binding that collides
        # with an existing `scope.lookup(...)` entry, reserved or not.
    for child in stmt.children:
        lower_stmt(child, scope, builder)


@impl("threadblock_swizzle_pattern")
def _lower_threadblock_swizzle_pattern(stmt: SemanticStmt, scope: LoweringScope, builder: IRBuilder) -> None:
    """Lower T.use_swizzle(panel_size, order) block-ID remapping.

    Implements rasterization2DRow and rasterization2DColumn swizzle patterns
    by computing remapped (bx, by) from the raw blockIdx values, using
    Elementwise + Select ops.
    """
    # Parse swizzle function name and panel_size from the semantic value.
    func_name: str | None = None
    panel_size: int | None = None
    try:
        if stmt.value is not None:
            tvm_tuple = stmt.value  # tirx.Call("tirx.tvm_tuple", args=[StringImm, IntImm])
            if isinstance(tvm_tuple, _tir.Call) and len(tvm_tuple.args) >= 2:
                name_arg = tvm_tuple.args[0]
                panel_arg = tvm_tuple.args[1]
                if isinstance(name_arg, _tir.StringImm):
                    func_name = name_arg.value
                if isinstance(panel_arg, _tir.IntImm):
                    panel_size = int(panel_arg)
    except Exception:
        pass

    if func_name is None or panel_size is None or func_name not in ("rasterization2DRow", "rasterization2DColumn"):
        # Unknown/un-parseable swizzle — pass through without remapping.
        for child in stmt.children:
            lower_stmt(child, scope, builder)
        return

    # Get raw bx/by placeholder Values and static grid extents
    bx_val = builder.block.index_values.get("bx")
    by_val = builder.block.index_values.get("by")
    grid_x = builder.block.block_extents.get("bx")
    grid_y = builder.block.block_extents.get("by")

    if bx_val is None or by_val is None or grid_x is None or grid_y is None:
        # Cannot swizzle without block index values or static extents.
        for child in stmt.children:
            lower_stmt(child, scope, builder)
        return

    # Build arithmetic helpers (all ops use i32 scalars)
    i32_ty = _scalar_i32_type()

    def _const(val: int) -> Value:
        op = builder.create(Constant(value=val, dtype="int32"), result_types=(i32_ty,))
        return op.results[0]

    def _binop(fn: str, lhs: Value, rhs: Value) -> Value:
        op = builder.create(Elementwise(fn=fn, inputs=(lhs, rhs)), result_types=(i32_ty,))
        return op.results[0]

    def _select(cond: Value, true_v: Value, false_v: Value) -> Value:
        op = builder.create(Select(cond=cond, true_val=true_v, false_val=false_v), result_types=(i32_ty,))
        return op.results[0]

    def _cmp_lt(lhs: Value, rhs: Value) -> Value:
        """Return bool tile: lhs < rhs (signed)."""
        bool_ty = _scalar_bool_type()
        op = builder.create(Elementwise(fn="lt", inputs=(lhs, rhs)), result_types=(bool_ty,))
        return op.results[0]

    def _cmp_ne(lhs: Value, rhs: Value) -> Value:
        """Return bool tile: lhs != rhs (signed)."""
        bool_ty = _scalar_bool_type()
        op = builder.create(Elementwise(fn="ne", inputs=(lhs, rhs)), result_types=(bool_ty,))
        return op.results[0]

    def _is_odd(v: Value) -> Value:
        """Return bool tile: v % 2 != 0."""
        return _cmp_ne(_binop("floormod", v, _const(2)), _const(0))

    # linear index: block_idx = bx + by * grid_x
    block_idx = _binop("add", bx_val, _binop("mul", by_val, _const(grid_x)))
    grid_size = grid_x * grid_y

    if func_name == "rasterization2DRow":
        panel_sz = panel_size * grid_x
        panel_offset = _binop("floormod", block_idx, _const(panel_sz))
        panel_idx = _binop("floordiv", block_idx, _const(panel_sz))
        total_panel = (grid_size + panel_sz - 1) // panel_sz
        last_panel_blocks = _binop("sub", _const(grid_size), _binop("mul", panel_idx, _const(panel_sz)))
        last_stride = _binop("floordiv", last_panel_blocks, _const(grid_x))
        stride = _select(
            _cmp_lt(_binop("add", panel_idx, _const(1)), _const(total_panel)),
            _const(panel_size),
            last_stride,
        )
        panel_col = _binop("floordiv", panel_offset, stride)
        reverse_col = _binop("sub", _binop("sub", _const(grid_x), _const(1)), panel_col)
        col_idx = _select(_is_odd(panel_idx), reverse_col, panel_col)
        row_idx = _binop(
            "add",
            _binop("floormod", panel_offset, stride),
            _binop("mul", panel_idx, _const(panel_size)),
        )
        swizzled_bx = col_idx
        swizzled_by = row_idx

    else:  # rasterization2DColumn
        panel_sz = panel_size * grid_y
        panel_offset = _binop("floormod", block_idx, _const(panel_sz))
        panel_idx = _binop("floordiv", block_idx, _const(panel_sz))
        total_panel = (grid_size + panel_sz - 1) // panel_sz
        last_panel_blocks = _binop("sub", _const(grid_size), _binop("mul", panel_idx, _const(panel_sz)))
        last_stride = _binop("floordiv", last_panel_blocks, _const(grid_y))
        stride = _select(
            _cmp_lt(_binop("add", panel_idx, _const(1)), _const(total_panel)),
            _const(panel_size),
            last_stride,
        )
        panel_row = _binop("floordiv", panel_offset, stride)
        reverse_row = _binop("sub", _binop("sub", _const(grid_y), _const(1)), panel_row)
        row_idx = _select(_is_odd(panel_idx), reverse_row, panel_row)
        col_idx = _binop(
            "add",
            _binop("floormod", panel_offset, stride),
            _binop("mul", panel_idx, _const(panel_size)),
        )
        swizzled_bx = col_idx
        swizzled_by = row_idx

    # Rebind bx/by in scope to swizzled values for children. Rebind the SAME
    # key _lower_thread_extent used (the block axis' TIR Var, by identity) so
    # child references resolve to the swizzled coord; fall back to the Value's
    # name only if no Var was recorded.
    bx_key = scope._axis_bind_vars.get("bx", getattr(bx_val, "name", None))
    by_key = scope._axis_bind_vars.get("by", getattr(by_val, "name", None))
    if bx_key is not None:
        scope.bind(bx_key, swizzled_bx)
    if by_key is not None:
        scope.bind(by_key, swizzled_by)

    for child in stmt.children:
        lower_stmt(child, scope, builder)


@impl("reduce_scope")
def _lower_reduce_scope(stmt: SemanticStmt, scope: LoweringScope, builder: IRBuilder) -> None:
    for child in stmt.children:
        lower_stmt(child, scope, builder)


@impl("warp_specialize")
def _lower_warp_specialize(stmt: SemanticStmt, scope: LoweringScope, builder: IRBuilder) -> None:
    """Lower warp_specialize (T.ws) as a transparent sequential pass-through.

    The ``warp_specialize`` attribute marks warp-group sections in a
    producer/consumer kernel (warp-group id 0 = consumer/compute, 1 =
    producer/TMA).  The cuda_tile dialect owns warp-specialization scheduling
    in the downstream lowering pipeline, so at this stage we simply lower each
    section's body in program order; the cuTile assembler routes the sections
    to the correct warp groups.

    Lowering both sections in program order preserves the sequential semantics
    required at this stage; warp-group routing occurs downstream.
    """
    for child in stmt.children:
        lower_stmt(child, scope, builder)


def _expr_contains_buffer_load(expr: Any) -> bool:
    """Return True if the TIR expression tree *expr* contains a BufferLoad.

    Used by `_lower_let`'s Bind-deferral (replay) registration to reject a
    deferred value that reads from a buffer -- see the call site for why.
    Traversal failures are rejected. Treating an unanalyzable expression as
    pure would allow replay to violate single-evaluation semantics.
    """
    found = False

    def _visit(node: Any) -> None:
        nonlocal found
        if isinstance(node, _tir.BufferLoad):
            found = True

    try:
        _tir.stmt_functor.post_order_visit(expr, _visit)
    except Exception as exc:
        raise TileIRLoweringError(f"replay purity analysis failed for expression {expr!r}; refusing to classify it as replay-safe") from exc
    return found


@impl("let")
def _lower_let(stmt: SemanticStmt, scope: LoweringScope, builder: IRBuilder) -> None:
    attrs = dict(stmt.attrs)
    var_name = attrs.get("var", "")
    # Key the binding by the TIR Var (object identity) when available: this is
    # what lets a `T.Persistent` coordinate named `bx` coexist with the kernel's
    # `blockIdx.x` (also named `bx`) — they are distinct Var objects.
    bind_key = _binding_var(stmt)
    if bind_key is None:
        bind_key = var_name
    # Prefer the raw semantic PrimExpr over the string-serialized attribute.
    tir_value: Any = stmt.value if stmt.value is not None else attrs.get("value")

    if tir_value is not None:
        # A Bind may reference a variable introduced by a later statement in
        # the same scope. Defer only that case and replay its raw expression
        # once the referenced variable is available.
        # Use _lower_attr_expr to handle both TIR PrimExpr and string-serialized attrs.
        try:
            bound_val = _lower_attr_expr(tir_value, scope, builder)
        except _UnboundScopeVariable as exc:
            if var_name:
                # Object-identity keys let a deferred coordinate coexist with
                # same-named live variables such as block axes. Replay is safe
                # only for pure expressions because it is not cached.
                if _expr_contains_buffer_load(tir_value):
                    raise TileIRLoweringNotImplementedError(
                        f"let binding '{var_name}' cannot be deferred for replay: its "
                        "value expression reads from a buffer (BufferLoad), and a "
                        "deferred Bind is re-lowered from scratch at every reference "
                        "site -- single-evaluation semantics cannot be guaranteed "
                        "under replay. Only pure index arithmetic is supported."
                    ) from exc
                scope.set_replay_binding(bind_key, tir_value)
            bound_val = None
        else:
            if var_name and bound_val is not None:
                # Bind directly into the CURRENT scope (not a new frame).
                # SemanticIR `let` has children=0 and the "body" (subsequent stmts
                # that use the variable) are siblings in the parent seq. A new frame
                # would be popped before those siblings are lowered, losing the binding.
                scope.bind(bind_key, bound_val)
                # Also store the raw TIR PrimExpr so that
                # _compute_partition_indices can symbolically divide let-bound vars
                # (e.g. "m_start = bx * block_M") to derive tile-level indices.
                scope._tir_expr_bindings[bind_key] = tir_value

    # Lower any explicit children (rare; occurs when let wraps a body expr).
    if stmt.children:
        with scope.frame():
            for child in stmt.children:
                lower_stmt(child, scope, builder)


def _lower_loop_bounds(stmt: SemanticStmt, scope: LoweringScope, builder: IRBuilder) -> tuple:
    """Lower a ``for`` stmt's bounds and pipeline flag: ``(start, stop, pipelined)``.

    Must be called before entering the loop's block_scope so the bound values
    land in the outer block, not the loop body.  Shared between the generic
    serial ``for`` lowering and the serial-for-inside-T.Parallel path.
    """
    attrs = dict(stmt.attrs)

    # Raw loop bounds are explicit SemanticStmt payloads. Serialized attrs are
    # retained only for hand-built semantic tests.
    tir_min = stmt.loop_min if stmt.loop_min is not None else attrs.get("min")
    tir_extent = stmt.loop_extent if stmt.loop_extent is not None else attrs.get("extent")

    start_val = None
    stop_val = None

    # Let loop-bound lowering failures propagate so they remain diagnosable.
    # The only legitimate stop_val=None is when tir_extent is genuinely absent.
    # Use _lower_attr_expr to handle both TIR PrimExpr and string-serialized attrs.
    if tir_min is not None:
        start_val = _lower_attr_expr(tir_min, scope, builder)

    if tir_extent is not None:
        # Compute stop = min + extent when min != 0, otherwise stop = extent.
        # Detect zero-min via string "0" (from _attrs) or IntImm(0).
        # Check isinstance before equality to avoid TIR __eq__ against str.
        if tir_min is None:
            min_is_zero = True
        elif isinstance(tir_min, str):
            min_is_zero = tir_min == "0"
        elif isinstance(tir_min, _tir.IntImm):
            min_is_zero = int(tir_min) == 0
        else:
            min_is_zero = False
        extent_val = _lower_attr_expr(tir_extent, scope, builder)
        if min_is_zero or start_val is None:
            stop_val = extent_val
        else:
            # stop = start + extent
            stop_val = _make_elementwise("add", (start_val, extent_val), None, builder)

    # ``SemanticStmt("for")`` carries ``kind="pipelined"`` in its attrs
    # when the TIR For node has ``num_stages`` / pipeline annotations (set in
    # ``semantic.py``).  Loop.emit_mlir threads a loop-external token as a
    # ForOp iter-arg for pipelined loops, creating the cross-iteration token
    # chain the tileiras assembler needs.  Pipelining is purely token-driven —
    # no ``num_stages`` MLIR attribute is added to the ForOp.
    is_pipelined = attrs.get("kind", "serial") == "pipelined"

    # Reject explicit pipeline schedule annotations (tl_pipeline_order,
    # tl_pipeline_stage, tl_pipeline_group).  These require a per-op pipeline
    # schedule that the TileIR pipeline does not yet implement.
    _UNSUPPORTED_PIPELINE_ANNOTATION_KEYS = (
        "annotation.tl_pipeline_order",
        "annotation.tl_pipeline_stage",
        "annotation.tl_pipeline_group",
    )
    _found_unsupported = [k for k in _UNSUPPORTED_PIPELINE_ANNOTATION_KEYS if k in attrs]
    if _found_unsupported:
        raise _UnsupportedTileIRNode(
            "TileIR pipeline lowering currently supports `num_stages`; explicit pipeline "
            f"schedule annotations are not representable yet: "
            f"{', '.join(k.replace('annotation.', '') for k in _found_unsupported)}."
        )

    return start_val, stop_val, is_pipelined


@impl("for")
def _lower_for(stmt: SemanticStmt, scope: LoweringScope, builder: IRBuilder) -> None:
    # Deferred import to break the stmt <- parallel cycle: parallel imports stmt
    # (for lower_stmt), so the parallel-loop entry points are imported here.
    from .parallel import _is_parallel_for, _lower_parallel_loop

    # Delegate T.Parallel for loops to the tile-level parallel handler.
    # Parallel loops represent whole-tile operations rather than Loop ops.
    if _is_parallel_for(stmt):
        _lower_parallel_loop(stmt, scope, builder)
        return

    attrs = dict(stmt.attrs)
    var_name = attrs.get("var", "")
    bind_key = _binding_var(stmt)
    if bind_key is None:
        bind_key = var_name
    loop_var_type = _scalar_i32_type()

    # Use block_scope so that body ops go into a nested Block.
    loop_var = _make_placeholder(builder, loop_var_type, name=var_name)

    start_val, stop_val, is_pipelined = _lower_loop_bounds(stmt, scope, builder)
    # stop_val=None → while-style (extent genuinely absent); create placeholder
    # only in this legitimate case, not on exception.
    if stop_val is None:
        stop_val = _make_placeholder(builder, loop_var_type, name="stop")

    with builder.block_scope(params=[loop_var]) as body_block, scope.frame():
        scope.bind(bind_key, loop_var)
        if stmt.children:
            lower_stmt(stmt.children[0], scope, builder)

    loop_op = Loop(
        start=start_val,
        stop=stop_val,
        step=None,
        init=None,
        body=body_block,
        pipelined=is_pipelined,
    )
    builder.create(loop_op)


@impl("while")
def _lower_while(stmt: SemanticStmt, scope: LoweringScope, builder: IRBuilder) -> None:
    # Guard each iteration with an IfElse whose false branch breaks the loop.
    tir_cond = stmt.condition if stmt.condition is not None else dict(stmt.attrs).get("condition")

    with builder.block_scope() as body_block, scope.frame():
        # Emit the condition check guard at the top of the while body.
        if tir_cond is not None:
            try:
                cond_val = _lower_attr_expr(tir_cond, scope, builder)
            except _UnsupportedTileIRNode:
                raise
            except Exception as exc:
                # Dropping the termination guard yields an infinite-loop
                # kernel that hangs with no diagnostic. A while-loop
                # without its guard is always wrong — raise loudly.
                raise _UnsupportedTileIRNode(f"while-loop: cannot lower termination condition {tir_cond!r}: {exc}") from exc
            # Then branch: empty (condition true → continue body).
            with builder.block_scope() as _then_block:
                pass  # no-op
            # Else branch: break out of the loop.
            with builder.block_scope() as _else_block:
                builder.create(Break())
            builder.create(IfElse(cond=cond_val, then_block=_then_block, else_block=_else_block))
        # Lower the body statements.
        if stmt.children:
            lower_stmt(stmt.children[0], scope, builder)

    loop_op = Loop(start=None, stop=None, step=None, init=None, body=body_block)
    builder.create(loop_op)


def _is_thread_index_only_cond(
    tir_cond: Any,
    extra_thread_var_names: set[str] | None = None,
) -> bool:
    """Return True if *tir_cond* involves only threadIdx.* variables.

    When True, the condition is a warp-specialization guard (e.g. ``T.ws(0)`` →
    ``tx >= 0 and tx < 128``).  In the CUDA Tile IR model, tile operations
    are warp-uniform; the warp-group routing is handled by the optimizer/
    assembler.  The condition can therefore be ERASED at the TileIR level —
    both branches are emitted as sequential code and the optimizer later
    splits them into warp groups.

    Detection: collect all Var names referenced in the expression and check
    they are all in the set of known threadIdx variable names.

    ``extra_thread_var_names`` is the set of ACTUAL TIR
    variable names for threadIdx.* that ``_lower_thread_extent`` accumulated
    in ``scope._thread_var_names``.  This allows detection of warp-specialize
    guards that use non-standard variable names (e.g. ``tid >= 128`` in
    minference, where ``tid`` is the threadIdx.x variable, not ``tx``).
    """
    if tir_cond is None:
        return False
    # Build the full set of known thread-index var names.
    # Always include the canonical names; merge in any kernel-specific names.
    thread_vars = {"tx", "ty", "tz"}
    if extra_thread_var_names:
        thread_vars = thread_vars | extra_thread_var_names
    try:
        if isinstance(tir_cond, str):
            # String-serialized condition — check if it mentions only thread vars
            vars_in = set(re.findall(r"\b[a-zA-Z_]\w*\b", tir_cond))
            keywords = {"and", "or", "not", "True", "False"}
            vars_in -= keywords
            return bool(vars_in and vars_in.issubset(thread_vars))
        # Real TIR node: collect all Var names
        vars_seen: set[str] = set()

        def _visit(node: Any) -> None:
            if isinstance(node, _tir.Var):
                vars_seen.add(node.name)

        _tir.stmt_functor.post_order_visit(tir_cond, _visit)
        return bool(vars_seen) and vars_seen.issubset(thread_vars)
    except Exception:
        return False


@impl("if")
def _lower_if(stmt: SemanticStmt, scope: LoweringScope, builder: IRBuilder) -> None:
    attrs = dict(stmt.attrs)
    tir_cond = stmt.condition if stmt.condition is not None else attrs.get("condition")
    if tir_cond is None:
        raise _UnsupportedTileIRNode("if-statement without a condition in SemanticIR")

    # Warp-specialize guard: thread-index-only conditions (e.g. T.ws(0) →
    # ``tx >= 0 and tx < 128``) cannot be represented as a scalar cuda_tile.if
    # condition because tx is a vector tile<256xi1>.  Erase the guard and emit
    # both branches as sequential code.  The cuda_tile optimizer routes threads
    # to the correct warp group during its warp-specialization pass.
    if _is_thread_index_only_cond(tir_cond, scope._thread_var_names):
        with scope.frame():
            if stmt.children:
                lower_stmt(stmt.children[0], scope, builder)
        if len(stmt.children) > 1:
            with scope.frame():
                lower_stmt(stmt.children[1], scope, builder)
        return

    # No silent fallback: a condition that fails to lower must raise loudly so
    # numerical failures are diagnosable.
    # Use _lower_attr_expr to handle both TIR PrimExpr and string-serialized attrs.
    cond_val = _lower_attr_expr(tir_cond, scope, builder)

    with builder.block_scope() as then_block, scope.frame():
        if stmt.children:
            lower_stmt(stmt.children[0], scope, builder)

    else_block: Block | None = None
    if len(stmt.children) > 1:
        with builder.block_scope() as else_block, scope.frame():
            lower_stmt(stmt.children[1], scope, builder)

    if_op = IfElse(cond=cond_val, then_block=then_block, else_block=else_block)
    builder.create(if_op)


@impl("buffer_store")
def _lower_buffer_store(stmt: SemanticStmt, scope: LoweringScope, builder: IRBuilder) -> None:
    attrs = dict(stmt.attrs)
    buf_name = attrs.get("buffer", "")
    try:
        buf_val = scope.lookup_buffer(buf_name)
    except KeyError as exc:
        raise TileIRLoweringError(f"sem_to_ir: buffer_store references unknown buffer {buf_name!r}.") from exc

    # Prefer the raw semantic PrimExpr over the string-serialized attribute.
    tir_value: Any = stmt.value if stmt.value is not None else attrs.get("value")

    if tir_value is None:
        raise _UnsupportedTileIRNode(f"buffer_store to {buf_name!r} without a value in SemanticIR")
    # No silent fallback: a store value that fails to lower must raise loudly.
    # Use _lower_attr_expr to handle both TIR PrimExpr and string-serialized attrs.
    val = _lower_attr_expr(tir_value, scope, builder)

    # Lower the explicit semantic store indices to TileIR Values so
    # that per-thread scatter (e.g. B_local[v] = ... where v depends on threadIdx.x)
    # uses the correct runtime element offset.
    # Store.emit_mlir handles tile_shape=() + non-empty indices via select-mask scatter.
    tir_store_indices: tuple | None = stmt.indices or None

    store_indices: tuple
    store_tile_shape: tuple
    if tir_store_indices:
        # Attempt to lower each index to a TileIR Value.
        lowered_store_indices = []
        for dim, idx in enumerate(tir_store_indices):
            try:
                idx_val = lower_expr(idx, scope, builder)
                lowered_store_indices.append(idx_val)
            except _UnsupportedTileIRNode as exc:
                raise _UnsupportedTileIRNode(f"buffer_store to {buf_name!r}: cannot lower index {idx!r} for dim {dim}: {exc}") from exc
        # Use tile_shape=() to signal scalar indexed store (scatter).
        # Store.emit_mlir dispatches to the select-mask scatter path for
        # SHARED/REGISTER tile buffers with non-empty indices.
        store_tile_shape = ()
        store_indices = tuple(lowered_store_indices)
    else:
        # Replace -1 sentinel dims with 1 to keep tile_shape positive.
        store_tile_shape = tuple(d if isinstance(d, int) and d > 0 else 1 for d in buf_val.type.shape)
        store_indices = tuple(0 for _ in buf_val.type.shape)

    store_op = Store(
        dst=buf_val,
        val=val,
        tile_shape=store_tile_shape,
        indices=store_indices,
    )
    builder.create(store_op)

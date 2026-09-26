"""TileIR control-flow ops (loops, branches, terminators)."""

from __future__ import annotations

import dataclasses
import functools
import operator
from typing import Any

from tilelang.tileir.emission_utils import (
    _as_tile,
    _as_token,
    _block_ends_with_terminator,
    _reshape_tile_to,
    _walk_block,
)
from tilelang.tileir.errors import TileIRLoweringNotImplementedError, _UnsupportedTileIRNode
from tilelang.tileir.ir.types import MemSpace
from tilelang.tileir.ir.value import Block, Value
from tilelang.tileir.ir.ops._base import (
    Effect,
    TileOp,
    attribute,
    nested_block,
    operand,
)


def _collect_loop_token_bufs(body: Block) -> list:
    """Export tokens for every global access, including nested reads.

    Even read-only tokens must leave the region through loop results: enclosing
    control flow may carry them, and a later store may depend on those reads.
    The dependency plan still permits independent loads to overlap.
    """
    from tilelang.tileir.passes.base import walk_block

    buffers = []

    def visit(op):
        for buf, effect in op.buffer_effects():
            if effect != Effect.NONE and buf.type.space == MemSpace.GLOBAL and buf not in buffers:
                buffers.append(buf)

    walk_block(body, visit)
    return buffers


def _redirect_op_token_map(ctx: Any, body: Block, written_bufs: list) -> None:
    """Redirect ``ctx._op_token`` entries for ops inside *body* that touched
    any buffer in *written_bufs* to that buffer's CURRENT ``ctx._token_map``
    entry after the caller rebinds it to the enclosing loop's dominating
    result.

    ``_ensure_token``'s plan-based path (see ``emission_utils.py``)
    resolves a token dependency via ``ctx._op_token[id(dep_op)]`` -- the raw
    MLIR value a SPECIFIC op produced when it was emitted -- independent of
    ``ctx._token_map``.  Once a structured loop region closes, any op inside
    its body that touched a GLOBAL buffer left its ``_op_token`` entry
    pointing at an in-region SSA value that does not dominate anything AFTER
    the loop.  Without this redirect, a not-yet-emitted post-loop op whose
    token-order dependency plan names that in-loop op as a dependency would
    embed the stale value directly into its own operands, producing an
    "operand does not dominate this use" MLIR verifier error. Mirrors
    ``IfElse.emit_mlir``'s identical ``_op_token`` fix-up for its own nested
    region.

    Ops already emitted INSIDE *body* (any intra-loop dependents) already
    baked their operand values into the MLIR at emission time and are
    unaffected by this later dict mutation -- only not-yet-emitted (post-
    loop) lookups observe the redirect.
    """
    if not written_bufs:
        return
    from tilelang.tileir.passes.base import walk_block
    from tilelang.tileir.passes.loop_carry import _buffer_reads_writes

    written_ids = {id(b) for b in written_bufs}

    def _visit(op: Any) -> None:
        op_key = id(op)
        if op_key not in ctx._op_token:
            return
        reads, writes = _buffer_reads_writes(op)
        for buf in (*reads, *writes):
            if id(buf) in written_ids:
                ctx._op_token[op_key] = ctx._token_map[buf]

    walk_block(body, _visit)


def _loop_body_has_direct_break(block: Any) -> bool:
    """Return True if *block* contains a ``Break`` reachable without crossing
    a nested ``Loop`` boundary.

    A CUDA Tile IR ``break`` returns control to the innermost enclosing loop,
    including when it is nested within another control-flow construct. A
    ``Break`` inside a nested ``Loop`` therefore belongs to that inner loop,
    not this one -- do not recurse into it.
    ``IfElse`` branches do not start a new loop scope, so they are
    recursed into (this is how ``T.Persistent``'s traced form
    reaches its ``Break``: guarded by an ``IfElse``, inside the outer
    ``for``'s body).
    """
    for op in block.ops:
        if isinstance(op, Break):
            return True
        if isinstance(op, Loop):
            continue
        if isinstance(op, IfElse):
            if _loop_body_has_direct_break(op.then_block):
                return True
            if op.else_block is not None and _loop_body_has_direct_break(op.else_block):
                return True
    return False


@dataclasses.dataclass(eq=False)
class Loop(TileOp, opcode="loop", effect=Effect.NONE):
    """Structured counted (for) or unbounded (while) loop.

    ``is_for`` distinguishes the two variants: when ``True`` the loop is a
    counted for-loop with ``start``/``stop``/``step`` bounds; when ``False``
    it is an open-ended while-loop (bounds are ignored).

    ``init`` carries the loop-carried initial values that flow into the body
    as block arguments (mirroring the CUDA Tile IR ForOp / LoopOp pattern).
    ``init`` is a list of Value operands (or None for no loop-carried values);
    None is treated as an empty list.

    ``pipelined``: when True, the assembler is free to pipeline
    this loop across iterations.  ``emit_mlir`` threads a single root token
    from OUTSIDE the ForOp (via ``ct.make_token()`` before the ForOp) through
    the ForOp iter-args.  Body memory-ops that have no ordering constraints
    (``deps=[]`` in the token_plan) see this loop-external token via
    ``ctx._pipelined_loop_token_arg``, creating cross-iteration token
    dependencies that the tileiras assembler requires for pipelining.
    """

    start: Any = operand(default=None)
    stop: Any = operand(default=None)
    step: Any = operand(default=None)
    init: Any = operand(default=None)
    body: Block = nested_block()
    pipelined: bool = attribute(default=False)

    @property
    def is_for(self) -> bool:
        """Return ``True`` when this is a counted for-loop."""
        return self.stop is not None

    def emit_mlir(self, ctx: Any) -> None:
        """Lower Loop to MLIR: scf.for (is_for=True) or scf.while/loop (is_for=False).

        Body ops are walked via ``_walk_block`` inside the appropriate
        ``ir.InsertionPoint``.  Loop-carried VALUES from ``init`` are threaded
        as iter-args, as are SHARED/REGISTER tile buffers and per-buffer
        ordering tokens for GLOBAL buffers written inside the loop.

        For-loop body is auto-terminated with ``loop_continue([])`` unless the
        last op in the body is already a terminator (Break/Continue).

        A counted for-loop (``is_for``) whose body directly contains a
        ``Break`` (e.g. ``T.Persistent``'s wave-exhaustion guard, which
        calls ``T.loop_break()``) cannot use ``cuda_tile.for`` at all: the
        CUDA Tile IR ForOp does not support early termination, whereas LoopOp
        does. Such a loop is therefore
        emitted via ``_emit_for_with_break``, which re-expresses it as a
        ``cuda_tile.loop`` with the induction variable threaded as the loop's
        sole iter-arg. See that method's docstring for the exact shape.
        """
        # Normalise init to loop-carried Values and resolve their MLIR tiles.
        init_list = []
        if self.init is not None:
            if isinstance(self.init, (list, tuple)):
                init_list = list(self.init)
            else:
                init_list = [self.init]
        init_mlir = [_as_tile(ctx, ctx.lookup(v)) for v in init_list]
        init_types = [m.tile_type for m in init_mlir]

        if self.is_for:
            if _loop_body_has_direct_break(self.body):
                self._emit_for_with_break(ctx, init_list)
            else:
                self._emit_for(ctx, init_list, init_mlir, init_types)
        else:
            self._emit_while(ctx, init_list, init_mlir, init_types)
        return None  # side-effect only

    def _emit_for(self, ctx: Any, init_list, init_mlir, init_types) -> None:
        """Emit the scf.for path: a counted loop with tile/token iter-args."""
        ct = ctx.ct
        ct_gen = ctx.ct_gen
        ir = ctx.ir
        loc = ctx.loc
        # scf.for path
        i32 = ir.IntegerType.get_signless(32)
        tile_i32 = ct.TileType.get([], i32)

        def _to_i32_tile(v):
            if v is None:
                return ct.constant(0, tile_type=tile_i32, loc=loc)
            raw = ctx.lookup(v)
            return _as_tile(ctx, raw)

        lb = _to_i32_tile(self.start)
        ub = _to_i32_tile(self.stop)
        step = _to_i32_tile(self.step) if self.step is not None else ct.constant(1, tile_type=tile_i32, loc=loc)

        # Thread loop-live shared/register tiles through ForOp iter-args so
        # updates dominate uses after the loop. Standalone emission falls back
        # to carrying every tile when loop-carry analysis is unavailable.
        _all_tile_keys = list(ctx._tile_map.keys())
        _carry_ids = getattr(self, "_carry_tile_value_ids", None)
        if _carry_ids is None:
            tile_keys = _all_tile_keys
        else:
            tile_keys = [k for k in _all_tile_keys if id(k) in _carry_ids]
        # Restore non-carried entries after the body mutates the shared map.
        _dropped_tile_snapshot = {k: ctx._tile_map[k] for k in _all_tile_keys if k not in tile_keys}
        tile_init_mlir = [_as_tile(ctx, ctx._tile_map[k]) for k in tile_keys]

        # Thread last-op and last-store tokens for written global buffers to
        # preserve cross-iteration RAW, WAR, and WAW dependencies.
        _written_global_bufs: list[Any] = _collect_loop_token_bufs(self.body)

        # Each written buffer carries separate last-op and last-store tokens.
        _tok_type_mlir = None
        _token_init_mlir: list[Any] = []  # 2 tokens per written buf: [last_op0, last_store0, last_op1, ...]
        for _wbuf in _written_global_bufs:
            try:
                _last_op_tok = ct.make_token(loc=loc)
                _last_store_tok = ct.make_token(loc=loc)
                _token_init_mlir.append(_last_op_tok)
                _token_init_mlir.append(_last_store_tok)
                if _tok_type_mlir is None:
                    _tok_type_mlir = _last_op_tok.type
            except AttributeError:
                # A returned token lacks a usable .type → give up token tracking
                # for this region. A real ct.make_token emit failure propagates.
                _written_global_bufs = []
                _token_init_mlir = []
                break

        all_init_mlir = init_mlir + tile_init_mlir + _token_init_mlir
        tok_mlir_type = _tok_type_mlir
        if tok_mlir_type is not None:
            all_init_types = init_types + [m.tile_type for m in tile_init_mlir] + [tok_mlir_type] * len(_token_init_mlir)
        else:
            all_init_types = init_types + [m.tile_type for m in tile_init_mlir]

        # A loop-external token enables pipelining while dominating all body
        # uses without becoming an iter-arg.
        pipelined_tok = None
        if self.pipelined:
            pipelined_tok = ct.make_token(loc=loc)

        for_op = ct_gen.ForOp(
            resultValues=tuple(all_init_types),
            lowerBound=lb,
            upperBound=ub,
            step=step,
            initValues=all_init_mlir,
            loc=loc,
        )

        # Body block: first arg is the induction var (tile<i32>, must match
        # bounds type), then iter-args (one per init + tile value).
        # ForOp bounds and step are tile<i32>, so the IV must also be
        # tile<i32> — using raw i32 would trigger a type-mismatch verifier
        # error from the CUDA Tile IR optimizer.
        # all_init_mlir may include raw MLIR token values (not ct.Tile).
        # Use m.type directly when m has no .value attribute (raw MLIR value).
        body_arg_types = [tile_i32] + [m.value.type if hasattr(m, "value") else m.type for m in all_init_mlir]
        body_block = ir.Block.create_at_start(for_op.region, body_arg_types)

        # Bind the induction variable (body_block.arguments[0]) to the
        # loop_var TileIR Value stored as the first param of self.body.
        # This allows body ops that reference the loop variable (e.g. Copy
        # with runtime indices, buffer_store with loop-var indexing) to look
        # it up in ctx.
        iv_arg = body_block.arguments[0]
        if self.body.params:
            ctx.bind(self.body.params[0], iv_arg)

        # Bind iter-arg results (arguments[1:1+len(init_list)]) to init_list Values.
        iter_args = list(body_block.arguments[1:])
        for v, mlir_arg in zip(init_list, iter_args[: len(init_list)]):
            ctx.bind(v, mlir_arg)

        # Bind iter-arg results for tile-map entries.
        tile_iter_args = iter_args[len(init_list) :]
        for k, mlir_arg in zip(tile_keys, tile_iter_args[: len(tile_keys)]):
            ctx._tile_map[k] = mlir_arg

        # Bind token iter-args to ctx._token_map for written GLOBAL bufs.
        # _token_init_mlir has 2 entries per buf: [last_op0, last_store0, ...].
        # The iter-args carry them in the same order.
        _n_tile = len(tile_keys)
        _n_tok_total = len(_token_init_mlir)  # 2 * len(_written_global_bufs)
        _tok_iter_args = tile_iter_args[_n_tile : _n_tile + _n_tok_total]
        _saved_token_map_entries: dict = {}
        for _bidx, _wbuf in enumerate(_written_global_bufs):
            _saved_token_map_entries[_wbuf] = ctx._token_map.get(_wbuf)
            # Use the LAST_OP token (first of the pair) as the current token.
            # This makes subsequent store ops depend on the loop's last-op token.
            _last_op_iter_arg = _tok_iter_args[_bidx * 2] if _bidx * 2 < len(_tok_iter_args) else None
            if _last_op_iter_arg is not None:
                # The iter-arg is a raw token-typed MLIR block argument; wrap
                # it as ct.Token so store/load TKO helpers accept it as their
                # input_token (they reject plain mlir.ir.Value).
                ctx._token_map[_wbuf] = _as_token(ctx, _last_op_iter_arg)

        # Expose the loop-external token so _ensure_token can return it
        # instead of emitting a fresh in-loop make_token.
        # The token is defined before the ForOp and dominates all body ops.
        saved_entry = getattr(ctx, "_pipelined_loop_token_entry_arg", None)
        if pipelined_tok is not None:
            ctx._pipelined_loop_token_entry_arg = pipelined_tok
        else:
            ctx._pipelined_loop_token_entry_arg = None

        # Expose the count of loop-carried iter-args (init + tile + token)
        # so a Break/Continue in the body can reject itself loudly rather
        # than emit operands_=[] and produce IR the MLIR verifier rejects
        # (yield operand count must equal the ForOp result count).
        _saved_iter_arg_count = getattr(ctx, "_loop_iter_arg_count", 0)
        ctx._loop_iter_arg_count = len(init_list) + len(tile_keys) + len(_token_init_mlir)
        # Isolate `_loop_break_forward_tiles` from this (ForOp) nesting
        # level: it is ONLY meaningful for the specific LoopOp instance that
        # `_emit_for_with_break` set it up for (its value is keyed to THAT
        # loop's counter tile, forwarded through `Break.emit_mlir`). A plain
        # `cuda_tile.for` never itself needs it (Break inside a bare ForOp
        # body is already rejected/handled purely via `_loop_iter_arg_count`)
        # but if this ForOp is NESTED inside an outer `_emit_for_with_break`
        # loop, the outer loop's stale forward-tiles tuple must not leak into
        # this body: an inner Break here could otherwise coincidentally match
        # the outer loop's iter-arg count and silently forward the WRONG
        # (outer) counter instead of hitting the loud arity-mismatch
        # rejection. Set to None for the duration of this body and restore
        # the outer value afterward (mirrors `_loop_iter_arg_count` above).
        _saved_forward_tiles = getattr(ctx, "_loop_break_forward_tiles", None)
        ctx._loop_break_forward_tiles = None
        _saved_forward_token_bufs = getattr(ctx, "_loop_break_forward_token_bufs", None)
        ctx._loop_break_forward_token_bufs = None

        with ir.InsertionPoint(body_block):
            _walk_block(self.body, ctx)
            if not _block_ends_with_terminator(self.body):
                # Auto-terminate: loop_continue with all carry-out values.
                # init_list values come from value_map; tile_keys from _tile_map.
                init_carry = [_as_tile(ctx, ctx.lookup(v)) for v in init_list] if init_list else []
                # Carry-out tile values: reshape to the expected ForOp iter-arg type
                # if the tile map was updated with a smaller tile (e.g. a staged
                # 3D shared buffer whose slot was written with a 2D tile by Copy).
                tile_carry = []
                for k, expected_init in zip(tile_keys, tile_init_mlir):
                    cur = _as_tile(ctx, ctx._tile_map[k])
                    expected_shape = list(expected_init.tile_type.shape)
                    cur_shape = list(cur.tile_type.shape)
                    if cur_shape != expected_shape:
                        # Reshape to expected (iter-arg) shape so loop_continue types match.
                        cur_elems = functools.reduce(operator.mul, cur_shape, 1) if cur_shape else 1
                        exp_elems = functools.reduce(operator.mul, expected_shape, 1) if expected_shape else 1
                        if cur_elems == exp_elems:
                            cur = _reshape_tile_to(ct, cur, expected_shape, loc)
                        else:
                            # A loop-carried tile whose element count changed inside the
                            # body cannot be a valid iter-arg (the yield type must match).
                            # Silently substituting a zero tile here is a SILENT MISCOMPILE
                            # (it drops the loop's accumulation) — never acceptable. The
                            # usual trigger is reading a register/shared fragment with a
                            # leading scalar/serial index (e.g. ``frag[k, j]`` with ``k`` a
                            # serial-loop var): the read currently returns the whole tile
                            # instead of slicing row ``k`` (tile_level.py only applies the
                            # parallel extents to GLOBAL buffers), so an accumulator
                            # ``acc[j] += frag[k, j]`` broadens to the fragment's shape.
                            # Fail loudly so the autotuner rejects this config rather than
                            # emitting wrong results.
                            raise TileIRLoweringNotImplementedError(
                                f"loop-carried tile shape changed in body: got {cur_shape}, "
                                f"iter-arg expects {expected_shape}. This usually means a "
                                f"register/shared fragment was read with a leading non-parallel "
                                f"index (row-slice of a fragment is not yet lowered). Refusing to "
                                f"silently zero the accumulator."
                            )
                    tile_carry.append(cur)
                # Carry out the updated tokens for written GLOBAL buffers.
                # Yield 2 tokens per written buf: LAST_OP, LAST_STORE.
                # Both are the current ctx._token_map[buf] (the store's out-token);
                # They remain separate to preserve the LAST_OP/LAST_STORE contract.
                token_carry: list[Any] = []
                for _bidx, _wbuf in enumerate(_written_global_bufs):
                    _last_op_init = _token_init_mlir[_bidx * 2] if _bidx * 2 < len(_token_init_mlir) else None
                    _last_store_init = _token_init_mlir[_bidx * 2 + 1] if _bidx * 2 + 1 < len(_token_init_mlir) else None
                    cur_tok = ctx._token_map.get(_wbuf)
                    token_carry.append(cur_tok if cur_tok is not None else _last_op_init)
                    token_carry.append(cur_tok if cur_tok is not None else _last_store_init)
                ct.loop_continue(init_carry + tile_carry + token_carry, loc=loc)

        # Restore the outer pipelined token context (handles nesting).
        ctx._pipelined_loop_token_entry_arg = saved_entry
        ctx._loop_iter_arg_count = _saved_iter_arg_count  # restore for nesting
        ctx._loop_break_forward_tiles = _saved_forward_tiles  # restore for nesting
        ctx._loop_break_forward_token_bufs = _saved_forward_token_bufs  # restore for nesting

        # Restore token_map entries overwritten by the token iter-arg binding.
        for _wbuf, _saved_tok in _saved_token_map_entries.items():
            ctx._token_map[_wbuf] = _saved_tok

        # Bind ForOp results back: first len(init_list) results → init_list,
        # remaining len(tile_keys) results → _tile_map,
        # final len(_written_global_bufs) results → _token_map.
        for_results = list(for_op.results)
        for v, result in zip(init_list, for_results[: len(init_list)]):
            ctx.bind(v, result)
        for k, result in zip(tile_keys, for_results[len(init_list) : len(init_list) + len(tile_keys)]):
            ctx._tile_map[k] = result
        # Restore the pre-loop tile for every dropped (non-carried) key so no
        # in-region SSA value emitted by the body leaks past the ForOp.  A
        # dropped key is, by construction, scratch that is never read after
        # the loop, so the restored pre-loop value is never consumed — this
        # only keeps ctx._tile_map free of dangling cross-region references.
        for k, pre_loop_tile in _dropped_tile_snapshot.items():
            ctx._tile_map[k] = pre_loop_tile
        # Bind ForOp token results back to _token_map for written GLOBAL bufs.
        # 2 token results per written buf: LAST_OP, LAST_STORE (in that order).
        _tok_start = len(init_list) + len(tile_keys)
        for _bidx, _wbuf in enumerate(_written_global_bufs):
            _last_op_result = for_results[_tok_start + _bidx * 2] if _tok_start + _bidx * 2 < len(for_results) else None
            # Bind the LAST_OP result as the current token for the buffer.
            # Wrap the raw ForOp ir.Value result as ct.Token so a store/load
            # AFTER the loop accepts it as input_token.
            if _last_op_result is not None:
                ctx._token_map[_wbuf] = _as_token(ctx, _last_op_result)
        _redirect_op_token_map(ctx, self.body, _written_global_bufs)

    def _emit_for_with_break(self, ctx: Any, init_list) -> None:
        """Emit a counted for-loop whose body directly contains a ``Break``
        (e.g. ``T.Persistent``) via ``cuda_tile.loop`` (LoopOp) instead of
        ``cuda_tile.for`` (ForOp) -- ForOp structurally forbids ``break``,
        whereas LoopOp supports it. The induction variable is threaded as the LoopOp's
        SOLE iter-arg, mirroring the shape ``_lower_while`` already builds
        for a user-authored ``while`` loop's guard:

            %counter0 = start
            %result = loop iter_values(%counter = %counter0) : i32 -> i32 {
                %cond = %counter >= stop
                if %cond { break %counter }
                <body ops, with the loop var bound to %counter>
                %next = %counter + step
                continue %next
            }

        The counter is always threaded as an iter-arg, plus TWO further
        iter-args (LAST_OP, LAST_STORE tokens) per GLOBAL buffer written
        directly in the body (see ``_collect_loop_token_bufs`` /
        ``_emit_for``'s matching pipelined-loop token machinery, which this
        mirrors) so that ordering established by writes inside the loop
        survives the loop's structured region regardless of which iteration
        ``break`` fires on. SHARED/REGISTER tile buffers
        touched in the body (A_shared/B_shared/C_local/... in the
        persistent-GEMM shape this exists for) are PHYSICAL storage, not
        pure SSA values -- exactly like the scalar ``alloc_var`` tiles the
        existing while+break tests rely on (see ``_emit_while``'s docstring)
        -- so they need no iter-arg threading here either.  A for-loop
        needing genuine additional loop-carried Value operands (``init``)
        alongside a body break is not implemented: `Break`'s existing loud
        rejection (``ctx._loop_iter_arg_count`` mismatch) still guards that
        case, narrowing this extension to exactly the persistent-loop
        pattern (plus GLOBAL-buffer token threading).

        The loop-carried-tile guard: "no iter-arg threading needed" above is
        true ONLY for tiles that are write-first scratch within each
        iteration (cleared/overwritten before use, dead after the loop --
        the persistent-GEMM shape this method was built for: ``T.clear``
        immediately before the inner accumulation loop). A tile that is
        genuinely loop-carried across iterations of THIS loop -- read before
        it is (re)written in the body (an accumulator with no per-iteration
        reset), or written in the body and read again after the loop
        (live-out) -- would, if silently allowed through, have its Load
        resolve to the SAME loop-invariant pre-loop SSA value on every
        iteration (since no iter-arg threads the previous iteration's Store
        result back in), or leak an in-region SSA value past the LoopOp
        (see the ctx._tile_map/_token_map snapshot/restore below). The
        former is a *silent* wrong answer, not a build failure -- e.g.
        ``T.clear(acc)`` then ``for w in T.serial(4): if w >= 3:
        T.loop_break(); acc[i] += 1.0`` compiles cleanly today and returns
        1.0 instead of 3.0. Reject it loudly instead.
        """
        from tilelang.tileir.passes.base import walk_block
        from tilelang.tileir.passes.loop_carry import _buffer_reads_writes

        if init_list:
            raise _UnsupportedTileIRNode(
                "a serial `for` loop with a body `break` AND explicit loop-carried "
                "Value operands (`init`) is not supported by the TileIR backend's "
                "break-capable loop lowering; only the plain induction-variable "
                "counter may be threaded through `break` today."
            )
        if self.pipelined:
            raise _UnsupportedTileIRNode(
                "a pipelined `for` loop with a body `break` is not supported by the "
                "TileIR backend: `_emit_for_with_break` threads per-buffer ordering "
                "tokens as GLOBAL-write iter-args, but does not implement the "
                "loop-invariant pipelined-token machinery (`_pipelined_loop_token_"
                "entry_arg`) `_emit_for` uses to enable cross-iteration load overlap."
            )

        # Reject a body that loop-carries a genuine SHARED/REGISTER
        # accumulator -- this loop threads ONLY the counter as an iter-arg
        # (see docstring), so a carried tile would silently read a stale
        # (loop-invariant) value every iteration instead of raising.
        #
        # ``_carry_tile_value_ids`` is set by ``loop_carry_pass`` (see
        # passes/loop_carry.py), which runs on every ``Loop`` op --
        # including this break-capable one -- as part of the standard
        # ``build_tileir_module`` pipeline (pipeline.py) before emission, so
        # in the normal compile path this attribute is always present here.
        _carry_ids = getattr(self, "_carry_tile_value_ids", None)
        if _carry_ids is None:
            # The pass did not run over this Loop -- e.g. a standalone emit
            # test that builds a `Loop` node directly and calls `emit_mlir`,
            # bypassing the pass pipeline (see the matching fallback note on
            # `_emit_for` above). Conservatively recompute just the
            # read-before-write half of `loop_carry_pass`'s analysis
            # directly over this loop's own body: a buffer read before it is
            # written INSIDE the body is a genuine cross-iteration
            # accumulator no matter what happens outside the loop, so this
            # scan cannot miss the silent-miscompile case above. It
            # deliberately skips the "written in body AND read after the
            # loop" (live-out) half of the full analysis -- that half needs
            # whole-root context this method does not have, but a missed
            # live-out tile fails LOUD (an MLIR dominance error once the
            # in-region SSA value is referenced post-loop -- see the
            # snapshot/restore below), never silently, so it does not need
            # to be caught here too.
            _seen_written: set[int] = set()
            _read_before_write: set[int] = set()

            def _visit(op: Any) -> None:
                reads, writes = _buffer_reads_writes(op)
                for r in reads:
                    if id(r) not in _seen_written:
                        _read_before_write.add(id(r))
                for w in writes:
                    _seen_written.add(id(w))

            walk_block(self.body, _visit)
            _carry_ids = _read_before_write

        _carried_tile_keys = [k for k in ctx._tile_map.keys() if id(k) in _carry_ids]
        if _carried_tile_keys:
            raise _UnsupportedTileIRNode(
                "loops containing T.loop_break() with loop-carried SHARED/REGISTER "
                "accumulators are not yet supported; hoist the accumulation or "
                "remove the break"
            )

        ct = ctx.ct
        ct_gen = ctx.ct_gen
        ir = ctx.ir
        loc = ctx.loc
        i32 = ir.IntegerType.get_signless(32)
        tile_i32 = ct.TileType.get([], i32)

        def _to_i32_tile(v):
            if v is None:
                return ct.constant(0, tile_type=tile_i32, loc=loc)
            raw = ctx.lookup(v)
            return _as_tile(ctx, raw)

        lb = _to_i32_tile(self.start)
        ub = _to_i32_tile(self.stop)
        step = _to_i32_tile(self.step) if self.step is not None else ct.constant(1, tile_type=tile_i32, loc=loc)

        # Thread a per-buffer ordering token as an ADDITIONAL LoopOp iter-arg
        # for every GLOBAL buffer written directly in the body -- mirrors
        # `_emit_for`'s pipelined-loop token machinery (see
        # `_collect_loop_token_bufs`).  Without this, a write to a
        # GLOBAL buffer inside a break-capable loop would have its ordering
        # token captured only in `ctx._token_map` during the body walk, then
        # discarded by the snapshot/restore below once the LoopOp's region
        # closes: any post-loop op referencing that buffer would either
        # violate MLIR dominance (referencing the in-region SSA token
        # directly) or silently fall back to a stale pre-loop token, losing
        # the ordering dependency (a dominance error or a stale read -- what
        # this threading fixes).  Carry TWO tokens per
        # written buffer (LAST_OP, LAST_STORE) exactly like `_emit_for`, even
        # though only LAST_OP is consulted via `ctx._token_map`; this remains
        # symmetric with `_emit_for`'s LAST_OP/LAST_STORE contract.
        _written_global_bufs = _collect_loop_token_bufs(self.body)
        _tok_type_mlir = None
        _token_init_mlir: list[Any] = []
        for _wbuf in _written_global_bufs:
            try:
                _last_op_tok = ct.make_token(loc=loc)
                _last_store_tok = ct.make_token(loc=loc)
                _token_init_mlir.append(_last_op_tok)
                _token_init_mlir.append(_last_store_tok)
                if _tok_type_mlir is None:
                    _tok_type_mlir = _last_op_tok.type
            except AttributeError:
                # A returned token lacks a usable .type -> give up token
                # tracking for this loop (same fallback as `_emit_for`).
                _written_global_bufs = []
                _token_init_mlir = []
                break

        loop_result_types = (tile_i32,) + tuple([_tok_type_mlir] * len(_token_init_mlir))
        loop_op = ct_gen.LoopOp(
            resultValues=loop_result_types,
            initValues=[lb] + _token_init_mlir,
            loc=loc,
        )
        body_arg_types = [tile_i32] + [_tok_type_mlir] * len(_token_init_mlir)
        body_block = ir.Block.create_at_start(loop_op.region, body_arg_types)
        counter_arg = body_block.arguments[0]
        counter_tile = ct.Tile(counter_arg, tile_i32)

        # Bind the LAST_OP token iter-arg (first of each pair) to
        # ctx._token_map for every written buffer, same convention
        # `_emit_for` uses, so body ops referencing the buffer chain off the
        # loop's per-iteration ordering token instead of a stale pre-loop
        # token or a fresh unordered one.
        _tok_iter_args = list(body_block.arguments[1:])
        for _bidx, _wbuf in enumerate(_written_global_bufs):
            _last_op_iter_arg = _tok_iter_args[_bidx * 2] if _bidx * 2 < len(_tok_iter_args) else None
            if _last_op_iter_arg is not None:
                ctx._token_map[_wbuf] = _as_token(ctx, _last_op_iter_arg)

        # Bind the loop var (self.body.params[0], set up identically to the
        # ForOp path in _lower_for) to the LoopOp's iter-arg block argument,
        # so body statements referencing the loop var (e.g. index math
        # derived from `w`) resolve correctly.  self.body.params may be
        # empty (e.g. a hand-built Loop with no declared body param) --
        # that's fine, it only means the body never references the counter.
        loop_var_value = self.body.params[0] if self.body.params else None
        if loop_var_value is not None:
            ctx.bind(loop_var_value, counter_arg)

        # Expose to Break: how many operands to forward, and the actual
        # values to forward as those operands.  The counter half is keyed on
        # the RAW counter tile (not an abstract Value looked up via
        # ctx.bind/ctx.lookup) so it does not depend on self.body.params
        # being populated -- the loop genuinely carries this iter-arg
        # regardless of whether the body binds it to a name.  The token half
        # (`_loop_break_forward_token_bufs`) is instead a list of BUFFER
        # KEYS, not literal values: `Break.emit_mlir` resolves
        # `ctx._token_map[buf]` live at the break site, because (unlike the
        # counter, which is loop-invariant within an iteration) a buffer's
        # current token may have been updated by a write earlier in the SAME
        # iteration, before the break fires. Both the synthesized "wave
        # exhausted" guard below and any mid-body `T.loop_break()` (lowered
        # generically by tile_ops.py's `_lower_break`, with no knowledge of
        # this loop) read this ctx state.
        saved_iter_arg_count = getattr(ctx, "_loop_iter_arg_count", 0)
        saved_forward_tiles = getattr(ctx, "_loop_break_forward_tiles", None)
        saved_forward_token_bufs = getattr(ctx, "_loop_break_forward_token_bufs", None)
        ctx._loop_iter_arg_count = 1 + len(_token_init_mlir)
        ctx._loop_break_forward_tiles = (counter_tile,)
        ctx._loop_break_forward_token_bufs = tuple(_written_global_bufs)

        # Snapshot ctx._tile_map / ctx._token_map before walking the
        # body so any in-region write can be undone afterward. SHARED/
        # REGISTER tile buffers still carry NO iter-arg (see the class
        # docstring) -- the loop-carried-tile guard above has already proven
        # the body carries no genuine loop-carried tile -- so every
        # `_tile_map` entry the body mutates is, by construction, dead
        # scratch. GLOBAL buffer tokens for the WRITTEN buffers above are
        # excluded from this
        # restore (they are bound to the LoopOp's token results instead,
        # below); any OTHER `_token_map` entry the body mutates (e.g. a
        # read-only buffer's conservative-fallback token) is still dead
        # scratch once the loop exits. Without restoring, a body write
        # leaves ctx._tile_map[buf] / _token_map[buf] pointing at an SSA
        # value defined INSIDE the LoopOp's region; any later op that
        # references it (e.g. the SAME fragment reused for an unrelated
        # computation after the loop) would violate MLIR's dominance rules.
        # Restore the exact pre-loop snapshot: entries the body mutated
        # revert to their pre-loop (dominating) value; entries the body
        # newly introduced (e.g. a tile allocated inside the loop body) are
        # removed entirely since they are inherently region-scoped and
        # cannot be legally referenced afterward.
        _pre_loop_tile_snapshot = dict(ctx._tile_map)
        _pre_loop_token_snapshot = dict(ctx._token_map)

        with ir.InsertionPoint(body_block):
            # Synthesize the "wave exhausted" guard: a plain `for` loop's
            # bounds check (lb <= counter < ub) is normally hardware/ForOp
            # machinery; LoopOp has no such native bound, so build it
            # explicitly: `if counter >= stop: break(counter)`.
            cond = ct.cmpi(
                ct.ComparisonPredicates.GREATER_THAN_OR_EQUAL,
                counter_tile,
                ub,
                ct.Signedness.SIGNED,
                loc=loc,
            )
            guard_if = ct_gen.IfOp(results_=(), condition=cond, loc=loc)
            guard_if.thenRegion.blocks.append()
            with ir.InsertionPoint(guard_if.thenRegion.blocks[0]):
                Break().emit_mlir(ctx)
            # No else branch is needed when the IfOp produces zero results.

            _walk_block(self.body, ctx)
            if not _block_ends_with_terminator(self.body):
                # Auto-terminate: continue, yielding the incremented counter
                # plus the current (possibly body-updated) token for each
                # written GLOBAL buffer -- mirrors `_emit_for`'s auto-
                # terminate token carry-out.
                next_counter = ct.add(counter_tile, step, loc=loc)
                token_carry: list[Any] = []
                for _bidx, _wbuf in enumerate(_written_global_bufs):
                    _last_op_init = _token_init_mlir[_bidx * 2] if _bidx * 2 < len(_token_init_mlir) else None
                    _last_store_init = _token_init_mlir[_bidx * 2 + 1] if _bidx * 2 + 1 < len(_token_init_mlir) else None
                    cur_tok = ctx._token_map.get(_wbuf)
                    token_carry.append(cur_tok if cur_tok is not None else _last_op_init)
                    token_carry.append(cur_tok if cur_tok is not None else _last_store_init)
                ct.loop_continue([next_counter] + token_carry, loc=loc)

        ctx._loop_iter_arg_count = saved_iter_arg_count
        ctx._loop_break_forward_tiles = saved_forward_tiles
        ctx._loop_break_forward_token_bufs = saved_forward_token_bufs

        # Restore the pre-loop snapshot (see above) -- undo any
        # in-region ctx._tile_map / ctx._token_map writes the body made,
        # except the written-GLOBAL-buffer token entries, which are bound to
        # the LoopOp's actual token results just below instead.
        _written_global_buf_ids = {id(b) for b in _written_global_bufs}
        for _k in list(ctx._tile_map.keys()):
            if _k in _pre_loop_tile_snapshot:
                ctx._tile_map[_k] = _pre_loop_tile_snapshot[_k]
            else:
                del ctx._tile_map[_k]
        for _k in list(ctx._token_map.keys()):
            if id(_k) in _written_global_buf_ids:
                continue
            if _k in _pre_loop_token_snapshot:
                ctx._token_map[_k] = _pre_loop_token_snapshot[_k]
            else:
                del ctx._token_map[_k]

        # Bind the LoopOp's LAST_OP token results back to ctx._token_map for
        # every written GLOBAL buffer -- mirrors `_emit_for`'s post-loop
        # token-result binding -- so ops AFTER the loop see an SSA value
        # that dominates them (the LoopOp's result), carrying the ordering
        # established by whichever iteration last wrote the buffer,
        # regardless of which iteration the break fired on.
        # loop_op.results[0] is the final counter value -- nothing in the
        # persistent-loop pattern reads it post-loop (the loop var is dead
        # once the loop exits) -- so token results start at index 1.
        _loop_results = list(loop_op.results)
        for _bidx, _wbuf in enumerate(_written_global_bufs):
            _last_op_idx = 1 + _bidx * 2
            if _last_op_idx < len(_loop_results):
                ctx._token_map[_wbuf] = _as_token(ctx, _loop_results[_last_op_idx])
        _redirect_op_token_map(ctx, self.body, _written_global_bufs)

    def _emit_while(self, ctx: Any, init_list, init_mlir, init_types) -> None:
        """Thread mutable tiles and memory ordering through every while edge."""
        ct, ir, loc = ctx.ct, ctx.ir, ctx.loc
        carry_ids = getattr(self, "_carry_tile_value_ids", None)
        tile_keys = [key for key in ctx._tile_map if carry_ids is None or id(key) in carry_ids]
        token_keys = _collect_loop_token_bufs(self.body)
        for key in token_keys:
            if key not in ctx._token_map:
                ctx._token_map[key] = ct.make_token(loc=loc)
        saved_tiles = dict(ctx._tile_map)
        saved_tokens = dict(ctx._token_map)
        tile_init = [_as_tile(ctx, ctx.get_tile(key)) for key in tile_keys]
        # Match the LAST_OP/LAST_STORE pair used by break-capable counted loops.
        token_init = [ctx._token_map[key] for key in token_keys for _ in range(2)]
        all_init = list(init_mlir) + tile_init + token_init
        types = [value.value.type if hasattr(value, "value") else value.type for value in all_init]
        loop_op = ctx.ct_gen.LoopOp(resultValues=tuple(types), initValues=all_init, loc=loc)
        body = ir.Block.create_at_start(loop_op.region, types)
        arguments = list(body.arguments)
        n_init, n_tiles = len(init_list), len(tile_keys)
        for key, arg in zip(init_list, arguments[:n_init]):
            ctx.bind(key, arg)
        for key, arg in zip(tile_keys, arguments[n_init : n_init + n_tiles]):
            ctx.set_tile(key, arg)
        for index, key in enumerate(token_keys):
            ctx._token_map[key] = _as_token(ctx, arguments[n_init + n_tiles + index * 2])

        saved_count = getattr(ctx, "_loop_iter_arg_count", 0)
        saved_forward = getattr(ctx, "_loop_break_forward_tiles", None)
        saved_token_keys = getattr(ctx, "_loop_break_forward_token_bufs", None)
        ctx._loop_iter_arg_count = len(all_init)
        # Values are resolved at each break/continue, after any body updates.
        ctx._loop_break_forward_tiles = tuple(init_list + tile_keys)
        ctx._loop_break_forward_token_bufs = tuple(token_keys)
        with ir.InsertionPoint(body):
            _walk_block(self.body, ctx)
            if not _block_ends_with_terminator(self.body):
                ctx.ct_gen.ContinueOp(operands_=_loop_forward_operands(ctx), loc=loc)
        ctx._loop_iter_arg_count = saved_count
        ctx._loop_break_forward_tiles = saved_forward
        ctx._loop_break_forward_token_bufs = saved_token_keys
        ctx._tile_map = saved_tiles
        ctx._token_map = saved_tokens
        results = list(loop_op.results)
        for key, result in zip(init_list, results[:n_init]):
            ctx.bind(key, result)
        for key, result in zip(tile_keys, results[n_init : n_init + n_tiles]):
            ctx.set_tile(key, result)
        for index, key in enumerate(token_keys):
            ctx._token_map[key] = _as_token(ctx, results[n_init + n_tiles + index * 2])
        _redirect_op_token_map(ctx, self.body, token_keys)


@dataclasses.dataclass(eq=False)
class IfElse(TileOp, opcode="if_else", effect=Effect.NONE):
    """Conditional branch with then/else bodies."""

    cond: Any = operand()
    then_block: Block = nested_block()
    else_block: Block = nested_block(default=None)

    def emit_mlir(self, ctx: Any) -> None:
        """Emit a branch that carries live tiles and memory tokens as results."""
        ct = ctx.ct
        ct_gen = ctx.ct_gen
        ir = ctx.ir
        loc = ctx.loc

        # Resolve the condition MLIR value.
        cond_raw = ctx.lookup(self.cond)
        cond = _as_tile(ctx, cond_raw)

        # Loads update source-buffer tokens too. Export every global buffer
        # touched by either branch so branch-local tokens cannot escape their
        # region or become inputs to the mutually exclusive sibling branch.
        def _collect_token_buffers(block_obj: Any) -> list:
            if block_obj is None:
                return []
            result = []
            for op in block_obj.ops:
                if isinstance(op, Loop):
                    # Only the loop's explicit token results escape its body.
                    # Read-only scratch loads may have no token carry at all.
                    result.extend(_collect_loop_token_bufs(op.body))
                    continue
                for buf, effect in op.buffer_effects():
                    if effect != Effect.NONE and buf.type.space == MemSpace.GLOBAL:
                        result.append(buf)
                for attr in ("then_block", "else_block", "body"):
                    nested = getattr(op, attr, None)
                    if nested is not None:
                        result.extend(_collect_token_buffers(nested))
            return result

        token_keys = list(dict.fromkeys(_collect_token_buffers(self.then_block) + _collect_token_buffers(self.else_block)))
        for buf in token_keys:
            if buf not in ctx._token_map:
                ctx._token_map[buf] = ct.make_token(loc=loc)

        # Snapshot all tile-map entries.  We carry ALL of them as IfOp results
        # (conservative but correct — the optimizer can eliminate unused ones).
        tile_keys = list(ctx._tile_map.keys())
        tile_init_mlir = [_as_tile(ctx, ctx._tile_map[k]) for k in tile_keys]
        n_tiles = len(tile_keys)

        token_init_mlir = [ctx._token_map[k] for k in token_keys]
        n_tokens = len(token_keys)

        # Determine the MLIR token type (from an existing token, if any).
        tok_mlir_type = None
        if token_init_mlir:
            try:
                tok_mlir_type = token_init_mlir[0].type
            except (AttributeError, IndexError):
                tok_mlir_type = None

        # Build result types: tile types first, then token types.
        tile_result_types = tuple(m.tile_type for m in tile_init_mlir)
        token_result_types: tuple = ()
        if tok_mlir_type is not None and n_tokens > 0:
            token_result_types = (tok_mlir_type,) * n_tokens
        result_types = tile_result_types + token_result_types

        if_op = ct_gen.IfOp(results_=result_types, condition=cond, loc=loc)

        pre_if_tokens = dict(ctx._token_map)
        pre_if_op_tokens = dict(ctx._op_token)
        branch_op_tokens = {}

        def _run_branch(block_obj, passthrough: bool) -> None:
            """Walk *block_obj* inside the current InsertionPoint,
            then yield all tile-map values followed by token-map values
            (updated by the walk or kept from pre-if values if passthrough)."""
            # Restore tile_map and token_map to pre-if values at the start of
            # each branch so both branches start from the same state.
            for k, v in zip(tile_keys, tile_init_mlir):
                ctx._tile_map[k] = v
            ctx._token_map = dict(pre_if_tokens)
            ctx._op_token = dict(pre_if_op_tokens)

            if not passthrough and block_obj is not None:
                _walk_block(block_obj, ctx)
                branch_op_tokens.update({k: v for k, v in ctx._op_token.items() if k not in pre_if_op_tokens})

            # If the walked block ended with a terminator (Break/Continue),
            # the MLIR block already has a terminator op — skip YieldOp insertion.
            # This can happen when an if-branch inside a while loop contains a break
            # or continue op.
            if not passthrough and block_obj is not None and _block_ends_with_terminator(block_obj):
                return

            # Build yield values: tiles + tokens.
            yield_tile_vals = [_as_tile(ctx, ctx._tile_map[k]) for k in tile_keys]
            yield_tok_vals = [ctx._token_map.get(k, token_init_mlir[i]) for i, k in enumerate(token_keys)]
            all_operands = [m.value for m in yield_tile_vals] + yield_tok_vals
            ct_gen.YieldOp(operands_=all_operands, loc=loc)

        # Then region
        if_op.thenRegion.blocks.append()
        with ir.InsertionPoint(if_op.thenRegion.blocks[0]):
            _run_branch(self.then_block, passthrough=False)

        # Else region — passthrough when no else_block.
        if_op.elseRegion.blocks.append()
        with ir.InsertionPoint(if_op.elseRegion.blocks[0]):
            _run_branch(self.else_block, passthrough=(self.else_block is None))

        # Restore tile_map and token_map to pre-if values before binding results
        # so the bind below is the sole update.
        for k, v in zip(tile_keys, tile_init_mlir):
            ctx._tile_map[k] = v
        ctx._token_map = pre_if_tokens
        ctx._op_token = pre_if_op_tokens | branch_op_tokens

        # Bind IfOp tile results (first n_tiles results) back to tile_map.
        for k, result in zip(tile_keys, list(if_op.results)[:n_tiles]):
            ctx._tile_map[k] = ct.Tile(result, result.type)

        # Bind IfOp token results (next n_tokens results) back to token_map.
        # Wrap each raw ir.Value as ct.Token so downstream store_view_tko etc.
        # receive the correct Python Token type.
        if n_tokens > 0 and tok_mlir_type is not None:
            _Token = type(token_init_mlir[0]) if token_init_mlir else None
            if_tok_results = list(if_op.results)[n_tiles : n_tiles + n_tokens]
            if_tok_wrapped: list = []
            for result in if_tok_results:
                try:
                    wrapped = _Token(result) if _Token is not None else result
                except (TypeError, ValueError):
                    wrapped = result
                if_tok_wrapped.append(wrapped)

            for k, wrapped in zip(token_keys, if_tok_wrapped):
                ctx._token_map[k] = wrapped

            _redirect_op_token_map(ctx, self.then_block, token_keys)
            if self.else_block is not None:
                _redirect_op_token_map(ctx, self.else_block, token_keys)

        return None  # side-effect only


def _loop_forward_operands(ctx):
    operands = []
    for value in getattr(ctx, "_loop_break_forward_tiles", None) or ():
        if isinstance(value, Value):
            value = ctx.get_tile(value) if ctx.is_tile_buffer(value) else ctx.lookup(value)
        operands.append(_as_tile(ctx, value).value)
    for key in getattr(ctx, "_loop_break_forward_token_bufs", None) or ():
        token = ctx._token_map[key]
        raw = token.value if hasattr(token, "value") else token
        operands.extend((raw, raw))
    return operands


@dataclasses.dataclass(eq=False)
class Break(TileOp, opcode="break", terminator=True, effect=Effect.NONE):
    """Break out of the enclosing loop."""

    def emit_mlir(self, ctx: Any) -> None:
        """Forward the enclosing loop's current state through an early exit.

        Counted loops supply their induction tile; while loops supply mutable
        Value keys resolved at this exit. Both supply live global token keys.
        Reject unconfigured state rather than emitting an invalid empty break.
        """
        n_carry = getattr(ctx, "_loop_iter_arg_count", 0)
        if n_carry:
            forward_tiles = getattr(ctx, "_loop_break_forward_tiles", None)
            forward_token_bufs = getattr(ctx, "_loop_break_forward_token_bufs", None) or ()
            n_forward = (len(forward_tiles) if forward_tiles is not None else 0) + 2 * len(forward_token_bufs)
            if forward_tiles is None or n_forward != n_carry:
                raise _UnsupportedTileIRNode(
                    "Break inside a loop that carries iter-args "
                    f"({n_carry}) is not supported: BreakOp would "
                    "yield 0 operands but the loop expects that many. Thread the "
                    "loop iter-args through Break or remove the carried values."
                )
            operands = _loop_forward_operands(ctx)
            ctx.ct_gen.BreakOp(operands_=operands, loc=ctx.loc)
            return None
        ctx.ct_gen.BreakOp(operands_=[], loc=ctx.loc)
        return None


@dataclasses.dataclass(eq=False)
class Continue(TileOp, opcode="continue", terminator=True, effect=Effect.NONE):
    """Continue to the next iteration of the enclosing loop."""

    def emit_mlir(self, ctx: Any) -> None:
        """Forward mutable while-loop state; reject unsupported counted carries."""
        forward = getattr(ctx, "_loop_break_forward_tiles", None)
        if forward is not None and all(isinstance(value, Value) for value in forward):
            operands = _loop_forward_operands(ctx)
            if len(operands) == getattr(ctx, "_loop_iter_arg_count", 0):
                ctx.ct_gen.ContinueOp(operands_=operands, loc=ctx.loc)
                return None
        if getattr(ctx, "_loop_iter_arg_count", 0):
            raise _UnsupportedTileIRNode(
                "Continue inside a loop that carries iter-args "
                f"({ctx._loop_iter_arg_count}) is not supported: ContinueOp "
                "would yield 0 operands but the loop expects that many. Thread "
                "the loop iter-args through Continue or remove the carried values."
            )
        ctx.ct_gen.ContinueOp(operands_=[], loc=ctx.loc)
        return None


@dataclasses.dataclass(eq=False)
class GridSync(TileOp, opcode="grid_sync", effect=Effect.NONE):
    """Grid-level synchronisation barrier (tvm_storage_sync / cooperative groups).

    In the CUDA Tile IR dialect there is no dedicated grid-sync op — the old
    backend treated ``tvm_storage_sync`` as a scheduling hint (no-op) because
    TKO token ordering provides the actual sequencing.  This implementation
    follows the same convention: GridSync emits nothing into the MLIR module.
    """

    def emit_mlir(self, ctx: Any) -> None:
        """GridSync is a no-op in MLIR emission (handled by TKO token ordering)."""
        return None

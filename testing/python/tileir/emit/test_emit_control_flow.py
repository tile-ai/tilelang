"""Tests for control-flow emit_mlir (Loop / IfElse / Break / Continue / GridSync).

Test structure
--------------
1. test_for_loop_emits_scf_for
   A Loop with is_for=True (stop != None) emits a "cuda_tile.for" op.
   The body contains a Fill op (data-movement, already implemented).

2. test_for_loop_body_fill_runs
   A for-loop whose body fills a buffer; verifies "constant" appears in MLIR
   (Fill lowering fires inside the loop body).

3. test_for_loop_with_store_in_body
   A for-loop body that does a Fill then a Store; verifies store TKO inside
   the loop body.

4. test_while_loop_emits_loop_op
   A Loop with is_for=False (stop=None) emits a "cuda_tile.loop" op.

5. test_if_then_else_emits_if_op
   An IfElse with a cond, then_block, and else_block emits "if" with
   then/else regions.

6. test_if_then_only_emits_if_op
   An IfElse with only a then_block (else_block=None) emits an "if" op.

7. test_break_emits_break
   A Break inside a for-loop body emits "cuda_tile.break" (or "break").

8. test_continue_emits_continue
   A Continue inside a for-loop body emits "cuda_tile.continue" (or "continue").

9. test_grid_sync_is_noop
   GridSync emits without error (no-op in MLIR emission).
"""

from __future__ import annotations

import pytest
from tilelang.tileir.checks import has_cuda_tile_ir_bindings

from tilelang.tileir.ir.types import TileType, MemSpace, dtype
from tilelang.tileir.ir.value import Block, Value
from tilelang.tileir.ir.ops import (
    Loop,
    IfElse,
    Break,
    Continue,
    GridSync,
    Fill,
    Store,
    Copy,
)

# ---------------------------------------------------------------------------
# Skip guard — needs cuda_tile MLIR bindings
# ---------------------------------------------------------------------------

_HAS_CUDA_TILE = has_cuda_tile_ir_bindings()

skip_no_cuda_tile = pytest.mark.skipif(
    not _HAS_CUDA_TILE,
    reason="cuda_tile MLIR bindings unavailable",
)

# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------

FP16 = dtype("float16")
I32 = dtype("int32")
BOOL = dtype("bool")


def _buf_type(shape=(64, 64)) -> TileType:
    return TileType(dtype=FP16, shape=tuple(shape), space=MemSpace.GLOBAL, layout=None)


def _reg_type(shape=(32, 32)) -> TileType:
    return TileType(dtype=FP16, shape=tuple(shape), space=MemSpace.REGISTER, layout=None)


def _scalar_type(dt=I32) -> TileType:
    return TileType(dtype=dt, shape=(), space=MemSpace.REGISTER, layout=None)


class _Ctr:
    def __init__(self):
        self._n = 0

    def next(self):
        value = self._n
        self._n += 1
        return value


def _entry_args_for(root: Block) -> list[tuple[str, TileType]]:
    return [(p.name or f"p{i}", p.type) for i, p in enumerate(root.params)]


# ---------------------------------------------------------------------------
# 1. For-loop emits cuda_tile.for
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_for_loop_emits_scf_for():
    """A Loop with is_for=True must emit a cuda_tile.for op."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    buf_ty = _buf_type()
    buf_val = Value(0, buf_ty, name="A")
    root = Block(params=[buf_val])

    # Build loop bounds as REGISTER i32 values (block params)
    lb_ty = _scalar_type(I32)
    ub_ty = _scalar_type(I32)
    lb_val = Value(1, lb_ty, name="lb")
    ub_val = Value(2, ub_ty, name="ub")
    root.params.extend([lb_val, ub_val])

    # Empty body block
    body = Block()
    loop_op = Loop(start=lb_val, stop=ub_val, step=None, init=None, body=body)
    loop_op.results = ()
    root.append(loop_op)

    entry_args = _entry_args_for(root)
    module = emit_module(root, kernel_name="for_kernel", entry_args=entry_args)
    text = str(module)

    # The for-loop renders either in generic form ('"cuda_tile.for"') or, when
    # the module is structurally valid (bounds/IV types agree), in the pretty
    # form ('for %loopIdx in (...) : tile<i32>').  Accept both.
    assert "cuda_tile.for" in text or '"cuda_tile.for"' in text or "for %loopIdx" in text, f"Expected a cuda_tile.for in:\n{text}"


# ---------------------------------------------------------------------------
# 2. For-loop body with Fill
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_for_loop_body_fill_runs():
    """A for-loop whose body fills a buffer — verifies Fill fires inside loop."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    buf_ty = _buf_type()
    buf_val = Value(0, buf_ty, name="A")
    lb_ty = _scalar_type(I32)
    ub_ty = _scalar_type(I32)
    lb_val = Value(1, lb_ty, name="lb")
    ub_val = Value(2, ub_ty, name="ub")

    root = Block(params=[buf_val, lb_val, ub_val])

    body = Block()
    fill_op = Fill(dst=buf_val, value=0.0, tile_shape=(32, 32))
    fill_op.results = ()
    body.append(fill_op)

    loop_op = Loop(start=lb_val, stop=ub_val, step=None, init=None, body=body)
    loop_op.results = ()
    root.append(loop_op)

    module = emit_module(root, kernel_name="for_fill_kernel", entry_args=_entry_args_for(root))
    text = str(module)

    assert "constant" in text, f"Expected 'constant' (Fill) inside for-loop body:\n{text}"
    assert "cuda_tile.for" in text or '"cuda_tile.for"' in text or "for %loopIdx" in text, f"Expected cuda_tile.for wrapper:\n{text}"


# ---------------------------------------------------------------------------
# 3. For-loop with Store in body
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_for_loop_with_store_in_body():
    """A for-loop body that stores a tile — verifies store TKO appears."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    ctr = _Ctr()
    buf_ty = _buf_type()
    buf_val = Value(ctr.next(), buf_ty, name="A")
    tile_ty = _reg_type()
    tile_val = Value(ctr.next(), tile_ty, name="tile")
    lb_ty = _scalar_type(I32)
    ub_ty = _scalar_type(I32)
    lb_val = Value(ctr.next(), lb_ty, name="lb")
    ub_val = Value(ctr.next(), ub_ty, name="ub")

    root = Block(params=[buf_val, tile_val, lb_val, ub_val])

    body = Block()
    store_op = Store(dst=buf_val, val=tile_val, tile_shape=(32, 32), indices=(0, 0))
    store_op.results = ()
    body.append(store_op)

    loop_op = Loop(start=lb_val, stop=ub_val, step=None, init=None, body=body)
    loop_op.results = ()
    root.append(loop_op)

    entry_args = _entry_args_for(root)
    module = emit_module(root, kernel_name="for_store_kernel", entry_args=entry_args)
    text = str(module)

    has_store = "store_view_tko" in text or "store_ptr_tko" in text
    assert has_store, f"Expected store TKO inside for-loop body:\n{text}"


# ---------------------------------------------------------------------------
# 4. While-loop emits cuda_tile.loop
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_while_loop_emits_loop_op():
    """A Loop with is_for=False (stop=None) emits a cuda_tile.loop op."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root = Block(params=[])

    body = Block()
    # Body immediately breaks
    brk_op = Break()
    brk_op.results = ()
    body.append(brk_op)

    loop_op = Loop(start=None, stop=None, step=None, init=None, body=body)
    loop_op.results = ()
    root.append(loop_op)

    module = emit_module(root, kernel_name="while_kernel", entry_args=[])
    text = str(module)

    # Pretty-printed as "loop {"; generic form as '"cuda_tile.loop"'
    assert "loop {" in text or "cuda_tile.loop" in text or '"cuda_tile.loop"' in text, f"Expected cuda_tile.loop for while-loop:\n{text}"


# ---------------------------------------------------------------------------
# 5. IfElse with then and else emits if op
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_if_then_else_emits_if_op():
    """An IfElse with then/else blocks emits a cuda_tile if op."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    buf_ty = _buf_type()
    buf_val = Value(0, buf_ty, name="A")
    bool_ty = _scalar_type(BOOL)
    cond_val = Value(1, bool_ty, name="cond")

    root = Block(params=[buf_val, cond_val])

    then_block = Block()
    fill_then = Fill(dst=buf_val, value=1.0, tile_shape=(32, 32))
    fill_then.results = ()
    then_block.append(fill_then)

    else_block = Block()
    fill_else = Fill(dst=buf_val, value=0.0, tile_shape=(32, 32))
    fill_else.results = ()
    else_block.append(fill_else)

    if_op = IfElse(cond=cond_val, then_block=then_block, else_block=else_block)
    if_op.results = ()
    root.append(if_op)

    entry_args = _entry_args_for(root)
    module = emit_module(root, kernel_name="if_else_kernel", entry_args=entry_args)
    text = str(module)

    assert "if " in text or '"cuda_tile.if"' in text, f"Expected if op in:\n{text}"
    # Should have two branches with constants
    assert text.count("constant") >= 2, f"Expected at least 2 constants (one per branch) in:\n{text}"


# ---------------------------------------------------------------------------
# 6. IfElse with then-only
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_if_then_only_emits_if_op():
    """An IfElse with no else_block emits an if op (else_block=None)."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    buf_ty = _buf_type()
    buf_val = Value(0, buf_ty, name="A")
    bool_ty = _scalar_type(BOOL)
    cond_val = Value(1, bool_ty, name="cond")

    root = Block(params=[buf_val, cond_val])

    then_block = Block()
    fill_then = Fill(dst=buf_val, value=2.0, tile_shape=(32, 32))
    fill_then.results = ()
    then_block.append(fill_then)

    if_op = IfElse(cond=cond_val, then_block=then_block, else_block=None)
    if_op.results = ()
    root.append(if_op)

    entry_args = _entry_args_for(root)
    module = emit_module(root, kernel_name="if_only_kernel", entry_args=entry_args)
    text = str(module)

    assert "if " in text or '"cuda_tile.if"' in text, f"Expected if op in:\n{text}"


# ---------------------------------------------------------------------------
# 7. Break inside for-loop body
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_break_emits_break():
    """A Break inside a for-loop body emits a break op."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    lb_ty = _scalar_type(I32)
    ub_ty = _scalar_type(I32)
    lb_val = Value(0, lb_ty, name="lb")
    ub_val = Value(1, ub_ty, name="ub")

    root = Block(params=[lb_val, ub_val])

    body = Block()
    brk_op = Break()
    brk_op.results = ()
    body.append(brk_op)

    loop_op = Loop(start=lb_val, stop=ub_val, step=None, init=None, body=body)
    loop_op.results = ()
    root.append(loop_op)

    module = emit_module(root, kernel_name="break_kernel", entry_args=_entry_args_for(root))
    text = str(module)

    # The break op terminator inside the for body
    assert "break" in text, f"Expected 'break' in:\n{text}"


# ---------------------------------------------------------------------------
# 8. Continue inside for-loop body
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_continue_emits_continue():
    """A Continue inside a for-loop body emits a continue op."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    lb_ty = _scalar_type(I32)
    ub_ty = _scalar_type(I32)
    lb_val = Value(0, lb_ty, name="lb")
    ub_val = Value(1, ub_ty, name="ub")

    root = Block(params=[lb_val, ub_val])

    body = Block()
    cont_op = Continue()
    cont_op.results = ()
    body.append(cont_op)

    loop_op = Loop(start=lb_val, stop=ub_val, step=None, init=None, body=body)
    loop_op.results = ()
    root.append(loop_op)

    module = emit_module(root, kernel_name="continue_kernel", entry_args=_entry_args_for(root))
    text = str(module)

    assert "continue" in text, f"Expected 'continue' in:\n{text}"


# ---------------------------------------------------------------------------
# 9. GridSync is a no-op
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_grid_sync_is_noop():
    """GridSync emits without error (no-op in MLIR: no special grid sync op in dialect)."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root = Block(params=[])

    gs_op = GridSync()
    gs_op.results = ()
    root.append(gs_op)

    # Must not raise
    module = emit_module(root, kernel_name="grid_sync_kernel", entry_args=[])
    assert module is not None


# ---------------------------------------------------------------------------
# 10. Nested: IfElse inside a for-loop
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_if_inside_for_loop():
    """An IfElse nested inside a for-loop body both emit correctly."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    buf_ty = _buf_type()
    buf_val = Value(0, buf_ty, name="A")
    bool_ty = _scalar_type(BOOL)
    cond_val = Value(1, bool_ty, name="cond")
    lb_ty = _scalar_type(I32)
    ub_ty = _scalar_type(I32)
    lb_val = Value(2, lb_ty, name="lb")
    ub_val = Value(3, ub_ty, name="ub")

    root = Block(params=[buf_val, cond_val, lb_val, ub_val])

    then_block = Block()
    fill_then = Fill(dst=buf_val, value=1.0, tile_shape=(32, 32))
    fill_then.results = ()
    then_block.append(fill_then)

    if_op = IfElse(cond=cond_val, then_block=then_block, else_block=None)
    if_op.results = ()

    body = Block()
    body.append(if_op)

    loop_op = Loop(start=lb_val, stop=ub_val, step=None, init=None, body=body)
    loop_op.results = ()
    root.append(loop_op)

    module = emit_module(root, kernel_name="nested_kernel", entry_args=_entry_args_for(root))
    text = str(module)

    assert "cuda_tile.for" in text or '"cuda_tile.for"' in text or "for %loopIdx" in text, f"Expected for op in:\n{text}"
    assert "if " in text or '"cuda_tile.if"' in text, f"Expected if op inside loop:\n{text}"


# ---------------------------------------------------------------------------
# 11. `_loop_break_forward_tiles` isolation across nesting.
#
# `_emit_for_with_break` (the LoopOp-based lowering of a `for` loop whose
# body directly contains a `break`, e.g. T.Persistent's outer wave loop)
# sets `ctx._loop_break_forward_tiles` to a 1-tuple holding ITS OWN counter
# tile, for the duration of walking its body, so a `Break` reached inside
# that body can forward the counter as the LoopOp's sole iter-arg operand.
#
# `_emit_for` and `_emit_while` (the OTHER two loop-emission paths) already
# save/restore `ctx._loop_iter_arg_count` around their own body walk (so a
# nested loop's Break sees the INNER loop's carry count, not the outer's) --
# but, before this fix, did NOT do the same for `_loop_break_forward_tiles`.
# A while-loop (or another for-loop) with exactly ONE genuine iter-arg,
# nested inside an outer for-with-break loop, would therefore inherit the
# OUTER's stale forward-tiles tuple: since the inner loop also sets
# `ctx._loop_iter_arg_count = 1` (matching by coincidence — one true
# iter-arg), `Break.emit_mlir`'s `len(forward_tiles) != n_carry` guard would
# NOT fire, and the inner `break` would silently forward the OUTER loop's
# counter as its own iter-arg operand instead of hitting the loud rejection
# for "Break inside a loop that carries iter-args (1) is not supported"
# (there is no legitimate way to forward the INNER loop's real iter-arg,
# since Break carries no operand information from user code).
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_break_forward_tiles_isolated_across_nested_loops():
    """An inner while-loop with ONE iter-arg and a `break` in its body,
    nested inside an outer for-with-break loop (the LoopOp path
    `_emit_for_with_break` builds for `T.Persistent`), must hit Break's loud
    iter-arg-mismatch rejection -- not silently forward the OUTER loop's
    counter as if it were the inner loop's carried value.
    """
    from tilelang.tileir.lowering.mlir_emit import emit_module
    from tilelang.tileir.errors import _UnsupportedTileIRNode

    lb_ty = _scalar_type(I32)
    ub_ty = _scalar_type(I32)
    lb_val = Value(0, lb_ty, name="lb")
    ub_val = Value(1, ub_ty, name="ub")
    iter_val = Value(2, _scalar_type(I32), name="iter0")

    root = Block(params=[lb_val, ub_val, iter_val])

    # Inner while-loop: ONE genuine iter-arg (iter_val), `break` directly in
    # its body (no operands -- Break never carries user-level iter-arg
    # values; only `_emit_for_with_break`'s synthesized counter-forwarding
    # is special-cased).
    inner_body = Block()
    inner_brk = Break()
    inner_brk.results = ()
    inner_body.append(inner_brk)
    inner_loop = Loop(start=None, stop=None, step=None, init=iter_val, body=inner_body)
    inner_loop.results = ()

    # Outer for-loop: body directly contains the inner while-loop AND a bare
    # `break` (a `Loop` boundary does not count towards the OUTER's "direct
    # break" scan, so the trailing bare `Break` is what makes
    # `_loop_body_has_direct_break` dispatch the outer loop to
    # `_emit_for_with_break` -- mirroring how a real `T.Persistent` wave loop
    # always has a genuine top-level break, guard-wrapped or not).
    outer_body = Block()
    outer_body.append(inner_loop)
    outer_brk = Break()
    outer_brk.results = ()
    outer_body.append(outer_brk)

    outer_loop = Loop(start=lb_val, stop=ub_val, step=None, init=None, body=outer_body)
    outer_loop.results = ()
    root.append(outer_loop)

    entry_args = _entry_args_for(root)
    with pytest.raises(_UnsupportedTileIRNode, match="iter-args"):
        emit_module(root, kernel_name="nested_break_isolation_kernel", entry_args=entry_args)


# ---------------------------------------------------------------------------
# 12. `_emit_for_with_break` must not leak an in-region SSA value past the
# LoopOp's region via ctx._tile_map.
#
# `_emit_for_with_break` threads NO tile iter-args at all (only the
# counter), so a body write to a REGISTER/SHARED tile must not mutate
# `ctx._tile_map[buf]` to point at the in-region SSA value with no
# snapshot/restore -- otherwise a LATER op (anywhere else in the kernel)
# that reads the SAME buffer would reference a value defined inside a
# region that has already closed, an MLIR dominance violation.
#
# This test builds the Loop directly (bypassing the pass pipeline) with
# `_carry_tile_value_ids` explicitly set to the EMPTY set -- simulating
# "loop_carry_pass ran and found nothing carried" -- so it isolates the
# tile-map snapshot/restore bookkeeping from the separate loud-rejection
# guard for loop-carried tiles (which fires first, and would otherwise
# reject any body write that is ALSO read after the loop, exactly the shape
# this test needs to exercise).
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_break_loop_does_not_leak_tile_map_entry_past_region():
    """A REGISTER tile written inside a for-with-break loop's body, then
    read again by an op AFTER the loop, must not produce an MLIR dominance
    violation. Pre-fix, `module.operation.verify()` raised exactly:
    ``error: unknown: operand #0 does not dominate this use`` (the operand
    being the in-region Fill result leaked into `ctx._tile_map`). Post-fix,
    the module verifies, and the post-loop read resolves to the PRE-loop
    (dominating) tile value -- confirming `_tile_map` was actually restored,
    not merely left un-crashing by accident.
    """
    from tilelang.tileir.lowering.mlir_emit import emit_module

    out_val = Value(0, _buf_type((32,)), name="Out")
    frag_val = Value(1, _reg_type((32,)), name="frag")
    lb_val = Value(2, _scalar_type(I32), name="lb")
    ub_val = Value(3, _scalar_type(I32), name="ub")

    root = Block(params=[out_val, lb_val, ub_val])
    root.alloc_buffers = [frag_val]

    # Pre-loop: establish frag's pre-loop (dominating) tile_map entry.
    pre_fill = Fill(dst=frag_val, value=1.0, tile_shape=(32,))
    pre_fill.results = ()
    root.append(pre_fill)

    # Loop body: write frag (scratch -- no read-before-write) then break.
    body = Block()
    body_fill = Fill(dst=frag_val, value=2.0, tile_shape=(32,))
    body_fill.results = ()
    body.append(body_fill)
    brk = Break()
    brk.results = ()
    body.append(brk)

    loop_op = Loop(start=lb_val, stop=ub_val, step=None, init=None, body=body)
    loop_op.results = ()
    # Simulate "loop_carry_pass ran and found nothing carried" so this test
    # isolates the snapshot/restore bookkeeping from the loud-rejection guard
    # (see the section docstring above).
    loop_op._carry_tile_value_ids = set()
    root.append(loop_op)

    # Post-loop: read frag again (Copy REGISTER -> GLOBAL).
    post_copy = Copy(src=frag_val, dst=out_val, tile_shape=(32,))
    post_copy.results = ()
    root.append(post_copy)

    entry_args = _entry_args_for(root)
    module = emit_module(root, kernel_name="i3_no_leak_kernel", entry_args=entry_args)

    # Must verify cleanly -- no cross-region dominance violation.
    assert module.operation.verify() is True

    # The post-loop store must reference the PRE-loop fill value (1.0), not
    # the in-region body fill (2.0) -- confirms restoration, not merely luck.
    text = str(module)
    assert "1.000000e+00" in text
    store_line = next(line for line in text.splitlines() if "store_view_tko" in line)
    assert "cst_1" in store_line, f"expected the pre-loop tile in the post-loop store:\n{store_line}"
    assert "2.000000e+00" not in store_line, f"post-loop store must not reference the in-region body value:\n{store_line}"

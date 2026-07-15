"""Loop-carry liveness analysis.

Determines, for each loop op, the minimal set of kernel-local (SHARED/REGISTER)
tile buffers that genuinely need to be threaded as ``ForOp`` / ``LoopOp``
iter-args.  A tile must be carried only when it is a true loop-carried value:

  * read-before-write inside the loop body (an accumulator / live-in whose
    previous-iteration value matters), OR
  * written inside the body AND read somewhere outside the body (live-out — the
    post-loop read would otherwise reference an SSA value defined inside the
    loop region, an MLIR domination error).

Write-first scratch buffers (overwritten fresh each iteration and never read
after the loop) are not carried. Carrying them is correct but pins their
registers live across the whole loop, crushing occupancy on memory-bound
kernels (e.g. RMSNorm / softmax persistent loops carrying their staging tiles).

The result is stored on each loop op as ``op._carry_tile_value_ids`` — a set of
``id(buffer_value)``.  ``Loop.emit_mlir`` consults it to filter the tiles it
threads as iter-args; when the attribute is absent (this pass did not run, e.g.
standalone emit tests) it falls back to carrying every ``_tile_map`` entry.

Missing a true live-out here is a *loud* failure (the cuda_tile verifier rejects
the cross-region SSA reference), never a silent miscompile — so the analysis errs
toward dropping only buffers it can prove are scratch.
"""

from __future__ import annotations

from typing import Any

from tilelang.tileir.ir.ops import Effect, TileOp
from tilelang.tileir.ir.value import Block
from tilelang.tileir.passes.base import PassContext, walk_block

__all__ = ["loop_carry_pass"]

_LOOP_OPCODES = frozenset({"loop"})


def _buffer_reads_writes(op: Any) -> tuple[list[Any], list[Any]]:
    """Return ``(reads, writes)`` from the schema's per-buffer effects."""
    if not hasattr(op, "buffer_effects"):
        return [], []
    reads: list[Any] = []
    writes: list[Any] = []
    for val, effect in op.buffer_effects():
        if effect in (Effect.READ, Effect.READWRITE):
            reads.append(val)
        if effect in (Effect.WRITE, Effect.READWRITE):
            writes.append(val)
    return reads, writes


def loop_carry_pass(root: Block, ctx: PassContext) -> None:
    """Annotate every loop op with ``_carry_tile_value_ids`` (see module doc)."""
    loops: list[Any] = []

    def _collect_loop(op: Any) -> None:
        if isinstance(op, TileOp) and getattr(op, "_opcode", None) in _LOOP_OPCODES:
            loops.append(op)

    walk_block(root, _collect_loop)

    for loop in loops:
        body_blocks = [b for b in loop.nested_blocks() if isinstance(b, Block)]

        # Body op identities (so we can identify reads OUTSIDE the body).
        body_op_ids: set[int] = set()
        for blk in body_blocks:
            walk_block(blk, lambda op, s=body_op_ids: s.add(id(op)))

        # Read-before-write + written, in body program order (pre-order ≈ order).
        seen_written: set[int] = set()
        read_before_write: set[int] = set()
        written_in_body: set[int] = set()

        def _visit_body(op: Any, _sw=seen_written, _rbw=read_before_write, _w=written_in_body) -> None:
            reads, writes = _buffer_reads_writes(op)
            for r in reads:
                if id(r) not in _sw:
                    _rbw.add(id(r))
            for w in writes:
                _w.add(id(w))
                _sw.add(id(w))

        for blk in body_blocks:
            walk_block(blk, _visit_body)

        # Reads anywhere OUTSIDE this loop body (live-out / cross-loop reuse).
        reads_outside: set[int] = set()

        def _visit_outside(op: Any, _ids=body_op_ids, _ro=reads_outside) -> None:
            if id(op) in _ids:
                return
            reads, _ = _buffer_reads_writes(op)
            for r in reads:
                _ro.add(id(r))

        walk_block(root, _visit_outside)

        carry: set[int] = set(read_before_write)
        for wid in written_in_body:
            if wid in reads_outside:
                carry.add(wid)
        loop._carry_tile_value_ids = carry

    ctx.results["loop_carry"] = True

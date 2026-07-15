"""TileIR token-order pass.

Computes the LAST_OP / LAST_STORE token-dependency plan for the memory ops
in a Block.

Overview
--------
Memory ops in CUDA Tile IR carry *tokens* that sequence their execution.  Two
ops are ordered iff a token dependency links them; ops with no shared token
may execute in parallel (overlap).  A naive per-buffer chain that threads
one token through every op on the same buffer forbids load-after-load
parallelism (unnecessary) and is therefore incorrect for performance parity.

This pass computes, for each memory op in the Block, the *exact* set of
prior memory ops it must wait on:

  - **STORE → LAST_OP(alias)**  (WAW + WAR):
        A store on alias A depends on the most recent op (read or write) on A.
        This covers both write-after-write (WAW: prior store) and
        write-after-read (WAR: prior load that a subsequent store would
        clobber).

  - **LOAD → LAST_STORE(alias)**  (RAW):
        A load on alias A depends only on the most recent *write* to A (not on
        prior loads).  Load-after-load carries no ordering constraint — the two
        loads may overlap freely.

  - **Disjoint aliases → no edge**:
        Ops on buffers whose alias-set bits do not overlap are independent and
        get no edge regardless of effect.

Algorithm
---------
Walk the block in program order.  Maintain two per-alias dicts:

  ``last_op[alias_set]``     — the most recent op (read OR write) on that alias.
  ``last_store[alias_set]``  — the most recent WRITE on that alias.

For each memory op:

  1. Resolve the alias set(s) for its ``buffer_operands()``.
  2. Compute deps:
       - If effect is WRITE (or READWRITE): deps = {last_op[alias]} if present.
       - If effect is READ:                 deps = {last_store[alias]} if present.
  3. Update:
       - Always update last_op[alias] = this op.
       - If effect is WRITE (or READWRITE): also update last_store[alias] = this op.

Note on multi-buffer ops (Copy / Gemm)
--------------------------------------
Ops like ``Copy`` have two buffer_operands (src, dst) and no SSA result.
The alias set must be queried on the *operand Values* (not on any result
Value, which doesn't exist) — i.e. on the op's ``buffer_operands()`` Values.

For multi-buffer ops the per-buffer effect declared by each
``buffer_operand(effect=...)`` determines whether that buffer is read,
written, or both.

Output
------
The pass stores a ``TokenPlan`` object at ``ctx.results["token_order"]``.
``TokenPlan.deps_for(op)`` returns a ``frozenset`` of prior ops that *op*
must wait on.  An empty frozenset means "no ordering constraint" (the op
may start as soon as a fresh root token is available).

Emit-seam contract
------------------
The plan is stored on ``ctx.results["token_order"]`` and forwarded to the
emit path as ``emit_ctx.token_plan`` so that ``_ensure_token`` consults it:

  * **When token_plan is present on EmitContext:**
    ``_ensure_token(ctx, buf_val)`` for a READ op returns the token of
    LAST_STORE (not LAST_OP), enabling load-load overlap.
    ``_ensure_token(ctx, buf_val)`` for a WRITE op returns the token of
    LAST_OP.

  * **When token_plan is absent (fallback):**
    A conservative per-buffer chain is used, serializing all ops on the
    same buffer (including load-after-load).
"""

from __future__ import annotations

from typing import Any

from tilelang.tileir.ir.value import Block
from tilelang.tileir.ir.ops import Effect
from tilelang.tileir.passes.base import walk_block
from tilelang.tileir.passes.dataflow import DataflowResult, ALIAS_UNIVERSE, ALIAS_EMPTY

__all__ = [
    "TokenPlan",
    "token_order_pass",
]


# TokenPlan — the pass output


class TokenPlan:
    """Mapping from each memory op to the prior ops it must wait on.

    Attributes
    ----------
    _deps : dict[int, frozenset]
        Maps ``id(op)`` → ``frozenset`` of prior op objects that *op*
        depends on.  An empty frozenset means no ordering constraint.

    The plan is indexed by Python ``id(op)`` (object identity) to avoid
    requiring ops to be hashable.  Callers retrieve deps via
    ``deps_for(op)``.
    """

    def __init__(self) -> None:
        self._deps: dict[int, frozenset] = {}
        self._id_to_op: dict[int, Any] = {}

    def record(self, op: Any, deps: frozenset) -> None:
        """Record *deps* as the dependency set for *op*."""
        key = id(op)
        self._deps[key] = deps
        self._id_to_op[key] = op

    def deps_for(self, op: Any) -> frozenset:
        """Return the frozenset of prior ops that *op* must wait on.

        Returns an empty frozenset if *op* is not in the plan (it has no
        ordering constraints — e.g. it was not seen as a memory op).
        """
        return self._deps.get(id(op), frozenset())

    def __repr__(self) -> str:
        return f"TokenPlan({len(self._deps)} ops)"


# _overlaps — alias-set overlap check


def _overlaps(a: int, b: int) -> bool:
    """Return True if alias sets *a* and *b* overlap.

    ALIAS_UNIVERSE (-1 / all bits set) overlaps everything.
    """
    if a == ALIAS_UNIVERSE or b == ALIAS_UNIVERSE:
        return True
    return bool(a & b)


# token_order_pass — the Pass entry point


def token_order_pass(root: Block, ctx: Any) -> None:
    """Compute token-ordering dependencies for all memory ops in *root*.

    Reads ``ctx.results['dataflow']`` (a ``DataflowResult``) and walks the
    Block to compute, for each memory op, which prior ops it must wait on.
    The result is stored at ``ctx.results['token_order']`` as a
    ``TokenPlan``.

    Parameters
    ----------
    root :
        The root ``Block`` to analyse (produced by ``IRBuilder`` or
        ``lower_kernel``).
    ctx :
        A ``PassContext`` whose ``results['dataflow']`` has been populated by
        ``dataflow_pass``.  ``token_order_pass`` MUST run after
        ``dataflow_pass`` in the pipeline.

    Raises
    ------
    KeyError
        If ``ctx.results['dataflow']`` is missing (i.e. dataflow was not run).
    """
    # This raises KeyError if dataflow wasn't run — intentional.
    dataflow: DataflowResult = ctx.results["dataflow"]

    plan = TokenPlan()

    # Per-alias tracking.
    # Keys are alias_set int values.
    # Values are op objects (Python references, not ids).
    last_op: dict[int, Any] = {}  # last op (read or write) on each alias
    last_store: dict[int, Any] = {}  # last write on each alias

    def _process_op(op: Any) -> None:
        """Examine one op and record its dependencies in the plan.

        Uses the schema's per-buffer effects to apply READ-only deps (RAW) to
        input buffers and WRITE deps (WAW+WAR) only to output buffers.  This
        prevents spurious WAR edges when two ops both read the same input.
        """
        effect = getattr(op, "memory_effect", Effect.NONE)
        if effect == Effect.NONE:
            return

        per_buf = list(op.buffer_effects()) if hasattr(op, "buffer_effects") else []
        if not per_buf:
            return

        # Accumulate deps for this op.
        deps: set[Any] = set()

        for v, buffer_effect in per_buf:
            alias = dataflow[v.id].alias_set
            if alias == ALIAS_EMPTY:
                continue

            if buffer_effect in (Effect.WRITE, Effect.READWRITE):
                # WAW + WAR: depend on LAST_OP (any prior op on overlapping alias).
                for tracked_alias, tracked_op in last_op.items():
                    if _overlaps(alias, tracked_alias):
                        deps.add(tracked_op)

            if buffer_effect in (Effect.READ, Effect.READWRITE):
                # RAW: depend only on LAST_STORE (most recent write to this alias).
                # Load-after-load carries no ordering constraint (LAST_STORE only).
                for tracked_alias, tracked_store in last_store.items():
                    if _overlaps(alias, tracked_alias):
                        deps.add(tracked_store)

        plan.record(op, frozenset(deps))

        # Update LAST_OP and LAST_STORE after computing deps for this op.
        # last_op is updated for ALL buffer operands (read OR write) because
        # a subsequent WRITE on the same alias must order after this op (WAR).
        for v, buffer_effect in per_buf:
            alias = dataflow[v.id].alias_set
            if alias == ALIAS_EMPTY:
                continue
            last_op[alias] = op
            if buffer_effect in (Effect.WRITE, Effect.READWRITE):
                last_store[alias] = op

    walk_block(root, _process_op)

    ctx.results["token_order"] = plan

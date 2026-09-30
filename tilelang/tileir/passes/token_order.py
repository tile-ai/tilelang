"""Compute memory-token dependencies for TileIR operations.

Loads wait on the latest overlapping store; writes wait on the latest
overlapping read or write. Disjoint aliases and consecutive loads remain
independent. The resulting ``TokenPlan`` is stored in the pass context for
MLIR emission.
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
        ``dataflow_pass``. ``token_order_pass`` must run after
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

        # A later write must wait on either a read or a write of the alias.
        for v, buffer_effect in per_buf:
            alias = dataflow[v.id].alias_set
            if alias == ALIAS_EMPTY:
                continue
            last_op[alias] = op
            if buffer_effect in (Effect.WRITE, Effect.READWRITE):
                last_store[alias] = op

    walk_block(root, _process_op)

    ctx.results["token_order"] = plan

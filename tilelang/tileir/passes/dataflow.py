"""TileIR fixpoint dataflow analysis.

Computes, per SSA ``Value.id``, a ``DataPredicate`` that tracks:

  - ``alias_set``             — bitmask indicating which memory alias groups
                                the value touches.
  - ``div_by``                — address divisibility (div_by=16 means the
                                address is guaranteed to be divisible by 16).
  - ``may_alias_internally``  — conservative flag: True if the value may alias
                                itself across different access paths.

The analysis is **flow-insensitive** but iterative to a fixpoint: it scans
the entire Block (incl. nested control-flow) repeatedly until no predicate
changes (dirty flag goes False).  This handles loop-carried aliases.

Key divergence from cuTile
--------------------------
cuTile keys predicates by ``Var.name`` (a string); our IR uses identity-based
``Value`` objects with integer ``Value.id`` keys.  The ``_Tracker`` here is
keyed by ``int`` (``Value.id``).

Alias seeding
-------------
Buffer ``Value``s declared in ``Block.params`` (the kernel entry parameters)
are seeded with distinct, non-overlapping alias-set bits — one fresh bit per
buffer param.  Each buffer is also seeded with the ``div_by`` constraint
supplied via ``param_constraints`` (a dict mapping ``Value.id`` → ``{"div_by":
int}``).  Non-buffer params (or any param not in ``param_constraints``) are
treated as ALIAS_UNIVERSE / div_by=1 (conservative).

Divisibility propagation (mirrors cuTile ``PointerOffset``)
-----------------------------------------------------------
For ``_Offset``-style ops (any op where ``buffer_operands()`` returns exactly
the base and plain ``operands()`` returns the offset), the result's div_by is
``gcd(base.div_by, offset.div_by)``.  The alias set flows from the base
unchanged — the offset does not introduce new aliases.

For ``_Assign``-style ops (buffer_operands() → a single source, no integer
arithmetic), the result inherits the source's full predicate.

For all other ops, the result is set to ``ALWAYS_TRUE_PREDICATE``
(alias_set=ALIAS_UNIVERSE, div_by=1, may_alias_internally=True) —
conservative but safe.

Fixpoint termination
--------------------
Predicates are updated via ``_Tracker.update`` which merges (OR alias_set,
GCD div_by, OR may_alias_internally).  Because alias_set only grows
(OR only sets bits) and div_by can only decrease (GCD ≤ min), the lattice
is finite and bounded: alias_set has at most N bits (N = number of params)
and div_by ≥ 1 (gcd(a,b) ≥ 1 for all positive a,b, except when one is 0
which is treated as the identity — gcd(a,0)=a).  Therefore the iteration
must terminate in at most N+log2(max_div_by) passes.
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass
from math import gcd
from typing import Any

from tilelang.tileir.ir.value import Block, Value

__all__ = [
    "DataPredicate",
    "DataflowResult",
    "dataflow_analysis",
    "dataflow_pass",
    "ALIAS_EMPTY",
    "ALIAS_UNIVERSE",
]

# Alias set constants

ALIAS_UNIVERSE: int = -1  # all bits set (signed -1 = 0xFFFF...F in two's complement)
ALIAS_EMPTY: int = 0  # no bits set

AliasSet = int


# DataPredicate


@dataclass(frozen=True)
class DataPredicate:
    """Per-value dataflow predicate.

    Attributes
    ----------
    alias_set : int
        Bitmask of alias groups this value touches.  Distinct bits represent
        distinct, non-aliasing memory regions.  ALIAS_UNIVERSE (-1) means the
        value may touch any memory region (conservative).
    div_by : int
        Address divisibility: the value's address (or the address it
        represents) is guaranteed divisible by this integer.  div_by=1 is
        the bottom / least-informative value.  div_by=0 is treated as a
        special marker for "exactly zero" (gcd(a,0)=a, so it acts as the
        neutral element for GCD-based merging).
    may_alias_internally : bool
        Conservative flag: True when the value may alias itself across
        different paths (e.g. an array element aliased via different indices).
    """

    alias_set: AliasSet
    div_by: int
    may_alias_internally: bool

    def unify(self, other: DataPredicate) -> DataPredicate:
        """Merge two predicates at a join point (widen / take conservative bound).

        - alias_set: OR (union of potential aliases).
        - div_by: GCD (weakest common divisibility guarantee).
        - may_alias_internally: OR (if either path may alias, the merge does).
        """
        return DataPredicate(
            alias_set=self.alias_set | other.alias_set,
            div_by=gcd(self.div_by, other.div_by),
            may_alias_internally=self.may_alias_internally | other.may_alias_internally,
        )

    def replace(self, **kv: Any) -> DataPredicate:
        """Return a copy with the given fields overridden."""
        return dataclasses.replace(self, **kv)


# Maximally conservative predicate (alias everything, div_by=1).
ALWAYS_TRUE_PREDICATE = DataPredicate(
    alias_set=ALIAS_UNIVERSE,
    div_by=1,
    may_alias_internally=True,
)


# DataflowResult


@dataclass
class DataflowResult:
    """Mapping from ``Value.id`` (int) to ``DataPredicate``.

    Attributes
    ----------
    predicates : dict[int, DataPredicate]
        Key: ``Value.id``.  Value: the predicate computed for that SSA value.
    """

    predicates: dict[int, DataPredicate]

    def __getitem__(self, value_id: int) -> DataPredicate:
        """Return the predicate for *value_id*, or ALWAYS_TRUE if not tracked."""
        return self.predicates.get(value_id, ALWAYS_TRUE_PREDICATE)


# _Tracker — keyed by Value.id (int)


class _Tracker:
    """Mutable state: Value.id → DataPredicate, with monotone updates."""

    def __init__(self) -> None:
        self.dirty: bool = False
        self._predicates: dict[int, DataPredicate] = {}

    def get(self, value_id: int) -> DataPredicate:
        """Return the current predicate for *value_id* (or ALWAYS_TRUE if unknown)."""
        return self._predicates.get(value_id, ALWAYS_TRUE_PREDICATE)

    def update(self, value_id: int, pred: DataPredicate) -> None:
        """Merge *pred* into the stored predicate for *value_id*.

        If the merged predicate differs from the old one, sets ``dirty=True``.
        """
        old = self._predicates.get(value_id)
        if old is None:
            new = pred
        else:
            new = old.unify(pred)
            if new == old:
                return
        self.dirty = True
        self._predicates[value_id] = new

    def propagate(self, src_id: int, dst_id: int) -> None:
        """Copy the predicate from *src_id* to *dst_id* (via update/merge)."""
        self.update(dst_id, self.get(src_id))

    def finalize(self) -> dict[int, DataPredicate]:
        """Return the final predicate mapping."""
        return dict(self._predicates)

    def reset_dirty(self) -> None:
        self.dirty = False


# _AliasSetMapper — assign fresh bit-per-buffer


class _AliasSetMapper:
    """Assigns a fresh power-of-two alias bit to each distinct buffer param."""

    def __init__(self) -> None:
        self._bit_seq = (1 << i for i in itertools.count())

    def next_bit(self) -> int:
        """Return the next unused alias bit."""
        return next(self._bit_seq)


# _analyze_block — one scan of a Block (plus nested control flow)


def _analyze_block(
    block: Block,
    tracker: _Tracker,
    innermost_loop: Any | None,
) -> None:
    """Scan all ops in *block*, propagating predicates through each op.

    Recursion handles nested control-flow (``Loop``, ``IfElse``).

    Alias propagation rules (mirroring cuTile's ``_analyze_aliases_in_block``):

    1. ``_Assign``-like (single buffer_operand, no integer offset): result
       inherits the buffer operand's predicate in full.
    2. ``_Offset``-like (one buffer_operand base + one plain operand offset):
       result inherits alias_set from base; div_by = gcd(base.div_by, offset.div_by).
    3. ``Loop``: each init-value flows into the corresponding body-block param
       (loop-carried data).  Then recurse into the body Block.
    4. ``IfElse``: recurse into both then- and else-blocks.
    5. Any other op with results: results are set to ALWAYS_TRUE (conservative).

    Note on our IR vs. cuTile
    -------------------------
    In our IR an op carries:
      - ``buffer_operands()`` — memory reference Values.
      - ``operands()``        — plain SSA Values (non-memory).
      - ``results``           — a tuple of SSA result Values.
      - ``nested_blocks()``   — nested Block bodies (for Loop/IfElse).
    """
    from tilelang.tileir.ir.ops import Loop, IfElse

    for op in block.ops:
        buf_ops = list(op.buffer_operands()) if hasattr(op, "buffer_operands") else []
        # op.operands() is the inherited TileOp method on every op (the
        # Elementwise field is named `inputs`, so it does not shadow this
        # method).  Elementwise declares a single tuple-valued operand
        # field (`inputs`), so operands() yields a 1-tuple wrapping that tuple;
        # flatten one level so the alias heuristic sees the individual Values.
        _raw_ops = list(op.operands()) if hasattr(op, "operands") else []
        plain_ops = []
        for _o in _raw_ops:
            if isinstance(_o, (tuple, list)):
                plain_ops.extend(_o)
            else:
                plain_ops.append(_o)
        results = list(op.results) if hasattr(op, "results") else []
        nested = list(op.nested_blocks()) if hasattr(op, "nested_blocks") else []

        if isinstance(op, Loop):
            # Flow loop init values into body-block params.
            init_list = op.init if op.init is not None else []
            if not isinstance(init_list, (list, tuple)):
                init_list = [init_list]

            # body.params corresponds to loop-carried iter-args. Continue and
            # Break carried values are conservatively mapped to ALIAS_UNIVERSE.
            body_params = op.body.params if op.body is not None else []
            for init_val, body_param in zip(init_list, body_params):
                if isinstance(init_val, Value) and isinstance(body_param, Value):
                    tracker.propagate(init_val.id, body_param.id)

            # For for-loops, init also flows into result values (0-iteration path).
            if op.is_for and results:
                for init_val, result_val in zip(init_list, results):
                    if isinstance(init_val, Value) and isinstance(result_val, Value):
                        tracker.propagate(init_val.id, result_val.id)

            # Recurse into the body.
            if op.body is not None:
                _analyze_block(op.body, tracker, op)

        elif isinstance(op, IfElse):
            if op.then_block is not None:
                _analyze_block(op.then_block, tracker, innermost_loop)
            if op.else_block is not None:
                _analyze_block(op.else_block, tracker, innermost_loop)

        elif len(buf_ops) == 1 and len(plain_ops) == 0 and len(results) > 0:
            # Assign-like: single buffer operand, zero plain operands → result is a
            # derived pointer and inherits the buffer's full alias/div predicate.
            # Assumption: this branch fires ONLY for pointer-derivation ops (e.g. a
            # view or cast that produces a new buffer reference).  Store/Load ops
            # carry Effect != NONE and produce no aliasing pointer result, so they
            # are handled by the caller's memory-effect path before reaching here.
            src_id = buf_ops[0].id
            for r in results:
                tracker.propagate(src_id, r.id)

        elif len(buf_ops) == 1 and len(plain_ops) == 1 and len(results) > 0:
            # Offset-like: base buffer + scalar offset → result inherits alias,
            # div_by = gcd(base_div, offset_div).
            base_val = buf_ops[0]
            offset_val = plain_ops[0]
            base_pred = tracker.get(base_val.id)
            offset_pred = tracker.get(offset_val.id)
            new_div = gcd(base_pred.div_by, offset_pred.div_by)
            result_pred = base_pred.replace(div_by=new_div)
            for r in results:
                tracker.update(r.id, result_pred)

        else:
            # Conservative: all results get ALWAYS_TRUE.
            # But first recurse into any nested blocks that aren't Loop/IfElse.
            for nb in nested:
                if nb is not None:
                    _analyze_block(nb, tracker, innermost_loop)
            for r in results:
                tracker.update(r.id, ALWAYS_TRUE_PREDICATE)


# dataflow_analysis — public entry point


def dataflow_analysis(
    root: Block,
    param_constraints: dict[int, dict[str, Any]] | None = None,
) -> DataflowResult:
    """Run fixpoint dataflow analysis on *root*, returning a ``DataflowResult``.

    Parameters
    ----------
    root :
        The root ``Block`` to analyse.  Its ``params`` list provides the
        kernel-entry Values (buffer and scalar arguments).
    param_constraints :
        Optional mapping ``Value.id → {"div_by": int}``.  Buffer params listed
        here are seeded with the given divisibility and a fresh alias-set bit.
        Params NOT listed here get ALWAYS_TRUE (conservative).

    Returns
    -------
    DataflowResult
        Keyed by ``Value.id``.
    """
    if param_constraints is None:
        param_constraints = {}

    tracker = _Tracker()
    alias_mapper = _AliasSetMapper()

    # Seed kernel-entry params (block.params)
    # Buffer params listed in param_constraints receive a fresh distinct alias
    # bit plus the specified div_by.  Other block params get ALWAYS_TRUE.
    seeded_ids: set[int] = set()
    for param in root.params:
        constraint = param_constraints.get(param.id)
        if constraint is not None:
            alias_bit = alias_mapper.next_bit()
            raw_div = constraint.get("div_by", 1)
            pred = DataPredicate(
                alias_set=alias_bit,
                div_by=raw_div if raw_div is not None else 1,
                may_alias_internally=False,
            )
        else:
            pred = ALWAYS_TRUE_PREDICATE
        tracker.update(param.id, pred)
        seeded_ids.add(param.id)

    # Seed any additional Values supplied via param_constraints
    # Callers may supply constraints for non-block-param Values (e.g. integer
    # offset constants with a known divisibility that are not entry params, or
    # alloc_shared / alloc_fragment buffers).
    #
    # If the constraint dict contains ``"alias_distinct": True``, the value
    # receives a fresh distinct alias bit (like a block param).  This is used
    # by pipeline.py to give alloc_shared / alloc_fragment buffers their own
    # alias bit so the token_order pass can identify load-load independence.
    #
    # Otherwise (plain div_by constraints), alias_set=ALIAS_UNIVERSE is used
    # (conservative — they don't restrict aliasing with respect to other values).
    for vid, constraint in param_constraints.items():
        if vid in seeded_ids:
            continue  # already seeded as a block param
        raw_div = constraint.get("div_by", 1)
        if constraint.get("alias_distinct", False):
            # Give this a fresh distinct alias bit (mirrors block-param seeding).
            alias_bit = alias_mapper.next_bit()
            pred = DataPredicate(
                alias_set=alias_bit,
                div_by=raw_div if raw_div is not None else 1,
                may_alias_internally=False,
            )
        else:
            pred = DataPredicate(
                alias_set=ALIAS_UNIVERSE,
                div_by=raw_div if raw_div is not None else 1,
                may_alias_internally=True,
            )
        tracker.update(vid, pred)
        seeded_ids.add(vid)

    # Also seed block-param Values from inner blocks that aren't in root.params
    # (e.g. loop body params) — they start as ALWAYS_TRUE until propagated.

    # First pass
    _analyze_block(root, tracker, innermost_loop=None)

    # Iterate to fixpoint
    # The lattice is finite (alias_set bits only grow via OR; div_by only
    # shrinks via GCD; may_alias_internally is a single bit).  Termination
    # is guaranteed.
    MAX_ITERS = 256  # safety ceiling; should never be hit in practice
    for _ in range(MAX_ITERS):
        if not tracker.dirty:
            break
        tracker.reset_dirty()
        _analyze_block(root, tracker, innermost_loop=None)

    return DataflowResult(predicates=tracker.finalize())


# dataflow_pass — Pass-shaped wrapper


def dataflow_pass(block: Block, ctx: Any) -> None:
    """Run ``dataflow_analysis`` and stash the result on ``ctx.results['dataflow']``.

    Reads ``ctx.results.get('param_constraints', {})`` to obtain divisibility
    seeds for kernel-param buffers.  Writes ``ctx.results['dataflow']`` with
    the resulting ``DataflowResult``.

    Parameters
    ----------
    block :
        The root ``Block`` to analyse.
    ctx :
        A ``PassContext`` (or any object with a ``results`` dict).  The
        result is stored under the key ``"dataflow"``.
    """
    param_constraints = ctx.results.get("param_constraints", {})
    result = dataflow_analysis(block, param_constraints=param_constraints)
    ctx.results["dataflow"] = result

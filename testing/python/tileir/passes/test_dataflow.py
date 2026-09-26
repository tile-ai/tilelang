"""Tests for fixpoint dataflow analysis.

Each test exercises a property stated in the task brief:
  1. distinct kernel-param buffers get DISTINCT alias sets;
  2. two Values derived from the SAME buffer share an alias set;
  3. divisibility propagation through pointer offset arithmetic;
  4. a value defined inside a Loop reaches fixpoint (no infinite iteration).
"""

from __future__ import annotations

import dataclasses
from math import gcd
from typing import Any


from tilelang.tileir.ir.types import TileType, MemSpace, dtype
from tilelang.tileir.ir.value import Block, Value
from tilelang.tileir.ir.ops import (
    TileOp,
    Effect,
    operand,
    buffer_operand,
    Loop,
)


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from tilelang.tileir.passes.dataflow import (
    DataPredicate,
    DataflowResult,
    dataflow_analysis,
    dataflow_pass,
    ALIAS_EMPTY,
    ALIAS_UNIVERSE,
)
from tilelang.tileir.passes.base import PassContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _global_type(dt="float16", shape=(128, 128)) -> TileType:
    return TileType(dtype=dtype(dt), shape=shape, space=MemSpace.GLOBAL, layout=None)


def _reg_type(dt="float16", shape=(16, 16)) -> TileType:
    return TileType(dtype=dtype(dt), shape=shape, space=MemSpace.REGISTER, layout=None)


def _int32_scalar() -> TileType:
    return TileType(dtype=dtype("int32"), shape=(), space=MemSpace.REGISTER, layout=None)


def _make_buffer_value(name: str, id_: int = 0, dt="float16") -> Value:
    """Create a Value that acts as a kernel-param buffer."""
    v = Value(id=id_, type=_global_type(dt), name=name)
    return v


# ---------------------------------------------------------------------------
# Helper op: PointerOffset (simple SSA node for arithmetic propagation tests)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(eq=False)
class _Offset(TileOp, opcode="ptr_offset_test", effect=Effect.NONE):
    """Synthetic op: result = base + offset (pointer arithmetic, for testing)."""

    base: Any = buffer_operand(effect=Effect.NONE)
    offset: Any = operand()


@dataclasses.dataclass(eq=False)
class _Assign(TileOp, opcode="assign_test", effect=Effect.NONE):
    """Synthetic assign op: result = src (plain value propagation, for testing)."""

    src: Any = buffer_operand(effect=Effect.NONE)


# ---------------------------------------------------------------------------
# 1. Distinct kernel-param buffers → distinct alias sets
# ---------------------------------------------------------------------------


class TestDistinctAliasSets:
    def test_two_params_distinct_alias_sets(self):
        """Two distinct buffer params should carry non-overlapping alias sets."""
        root = Block()
        buf_a = _make_buffer_value("buf_a", id_=0)
        buf_b = _make_buffer_value("buf_b", id_=1)
        root.params = [buf_a, buf_b]

        result = dataflow_analysis(
            root,
            param_constraints={
                buf_a.id: {"div_by": 16},
                buf_b.id: {"div_by": 16},
            },
        )

        pred_a = result[buf_a.id]
        pred_b = result[buf_b.id]

        # Alias sets must be distinct (non-zero) and non-overlapping.
        assert pred_a.alias_set != ALIAS_EMPTY
        assert pred_b.alias_set != ALIAS_EMPTY
        assert pred_a.alias_set != ALIAS_UNIVERSE
        assert pred_b.alias_set != ALIAS_UNIVERSE
        assert (pred_a.alias_set & pred_b.alias_set) == 0, (
            f"Expected distinct alias sets; got a={pred_a.alias_set:#010b} b={pred_b.alias_set:#010b}"
        )

    def test_three_params_all_distinct(self):
        """Three distinct buffer params should carry pairwise non-overlapping alias sets."""
        root = Block()
        bufs = [_make_buffer_value(f"buf_{i}", id_=i) for i in range(3)]
        root.params = bufs

        result = dataflow_analysis(root, param_constraints={b.id: {"div_by": 16} for b in bufs})

        preds = [result[b.id] for b in bufs]
        for i, pi in enumerate(preds):
            for j, pj in enumerate(preds):
                if i != j:
                    assert (pi.alias_set & pj.alias_set) == 0, f"buf_{i} and buf_{j} share alias bits"


# ---------------------------------------------------------------------------
# 2. Values derived from the same buffer share alias set
# ---------------------------------------------------------------------------


class TestDerivedValueAlias:
    def test_assign_inherits_alias_set(self):
        """A value derived (assigned) from a buffer param inherits its alias set."""
        root = Block()
        buf_a = _make_buffer_value("buf_a", id_=0)
        root.params = [buf_a]

        # Synthesise: derived = assign(buf_a)
        derived = Value(id=10, type=_global_type(), name="derived")
        op = _Assign(src=buf_a)
        op.results = (derived,)
        root.append(op)

        result = dataflow_analysis(
            root,
            param_constraints={
                buf_a.id: {"div_by": 16},
            },
        )

        pred_a = result[buf_a.id]
        pred_derived = result[derived.id]

        # Derived value must overlap the same alias set as buf_a.
        assert (pred_derived.alias_set & pred_a.alias_set) != 0, (
            f"derived.alias_set={pred_derived.alias_set:#010b} should overlap buf_a.alias_set={pred_a.alias_set:#010b}"
        )

    def test_derived_does_not_alias_different_buf(self):
        """A value derived from buf_a must not alias buf_b."""
        root = Block()
        buf_a = _make_buffer_value("buf_a", id_=0)
        buf_b = _make_buffer_value("buf_b", id_=1)
        root.params = [buf_a, buf_b]

        derived = Value(id=10, type=_global_type(), name="derived_a")
        op = _Assign(src=buf_a)
        op.results = (derived,)
        root.append(op)

        result = dataflow_analysis(
            root,
            param_constraints={
                buf_a.id: {"div_by": 16},
                buf_b.id: {"div_by": 16},
            },
        )

        pred_b = result[buf_b.id]
        pred_derived = result[derived.id]

        assert (pred_derived.alias_set & pred_b.alias_set) == 0, "derived_a should not alias buf_b"


# ---------------------------------------------------------------------------
# 3. Divisibility propagation
# ---------------------------------------------------------------------------


class TestDivisibilityPropagation:
    def test_base_div_preserved_by_zero_offset(self):
        """A buffer with div_by=16, offset by 0, keeps div_by=16."""
        root = Block()
        buf = _make_buffer_value("buf", id_=0)
        root.params = [buf]

        # offset value seeded with div_by=0 (zero)
        offset = Value(id=5, type=_int32_scalar(), name="offset_zero")
        result_val = Value(id=6, type=_global_type(), name="offset_ptr")
        op = _Offset(base=buf, offset=offset)
        op.results = (result_val,)
        root.append(op)

        result = dataflow_analysis(
            root,
            param_constraints={
                buf.id: {"div_by": 16},
                offset.id: {"div_by": 0},  # literal zero
            },
        )

        pred = result[result_val.id]
        # gcd(16, 0) == 16 in math.gcd (gcd(a,0)=a)
        assert pred.div_by == 16

    def test_base_div16_plus_multiple_of_16_keeps_16(self):
        """base div_by=16, offset div_by=32 (multiple of 16): result is gcd(16,32)=16."""
        root = Block()
        buf = _make_buffer_value("buf", id_=0)
        root.params = [buf]

        offset = Value(id=5, type=_int32_scalar(), name="offset_32")
        result_val = Value(id=6, type=_global_type(), name="result_ptr")
        op = _Offset(base=buf, offset=offset)
        op.results = (result_val,)
        root.append(op)

        result = dataflow_analysis(
            root,
            param_constraints={
                buf.id: {"div_by": 16},
                offset.id: {"div_by": 32},
            },
        )

        pred = result[result_val.id]
        assert pred.div_by == gcd(16, 32), f"expected 16 got {pred.div_by}"

    def test_base_div16_plus_offset4_drops_to_4(self):
        """base div_by=16, offset div_by=4: result is gcd(16,4)=4."""
        root = Block()
        buf = _make_buffer_value("buf", id_=0)
        root.params = [buf]

        offset = Value(id=5, type=_int32_scalar(), name="offset_4")
        result_val = Value(id=6, type=_global_type(), name="result_ptr")
        op = _Offset(base=buf, offset=offset)
        op.results = (result_val,)
        root.append(op)

        result = dataflow_analysis(
            root,
            param_constraints={
                buf.id: {"div_by": 16},
                offset.id: {"div_by": 4},
            },
        )

        pred = result[result_val.id]
        assert pred.div_by == gcd(16, 4), f"expected 4 got {pred.div_by}"

    def test_alias_propagates_through_offset(self):
        """Alias set is preserved after pointer offset (not reset to universe)."""
        root = Block()
        buf = _make_buffer_value("buf", id_=0)
        other = _make_buffer_value("other", id_=1)
        root.params = [buf, other]

        offset = Value(id=5, type=_int32_scalar(), name="offset")
        result_val = Value(id=6, type=_global_type(), name="result_ptr")
        op = _Offset(base=buf, offset=offset)
        op.results = (result_val,)
        root.append(op)

        result = dataflow_analysis(
            root,
            param_constraints={
                buf.id: {"div_by": 16},
                other.id: {"div_by": 16},
                offset.id: {"div_by": 16},
            },
        )

        pred_buf = result[buf.id]
        pred_other = result[other.id]
        pred_result = result[result_val.id]

        # result_ptr should alias buf but not other
        assert (pred_result.alias_set & pred_buf.alias_set) != 0
        assert (pred_result.alias_set & pred_other.alias_set) == 0


# ---------------------------------------------------------------------------
# 4. Loop fixpoint
# ---------------------------------------------------------------------------


class TestLoopFixpoint:
    def test_loop_carried_alias_stabilises(self):
        """A loop-carried buffer reference stabilises within a finite number of passes.

        Build:
            root Block with params = [buf_a, buf_b]
            a Loop with body that assigns buf_a into body_var
                inside body: _Assign(src=body_var) → inner_val

        The analysis must terminate (not loop forever) and the inner_val
        must end up with buf_a's alias set.
        """
        root = Block()
        buf_a = _make_buffer_value("buf_a", id_=0)
        buf_b = _make_buffer_value("buf_b", id_=1)
        root.params = [buf_a, buf_b]

        # body_var is a loop-carried iter-arg (simulated as a Block param)
        body_var = Value(id=10, type=_global_type(), name="body_var")

        # inner_val is derived inside the loop body
        inner_val = Value(id=11, type=_global_type(), name="inner_val")

        # Build the inner body block
        body_block = Block(params=[body_var])
        inner_op = _Assign(src=body_var)
        inner_op.results = (inner_val,)
        body_block.append(inner_op)

        # Build the loop with buf_a as init value (flows into body_var)
        loop_op = Loop(
            start=None,
            stop=None,
            step=None,
            init=[buf_a],
            body=body_block,
        )
        root.append(loop_op)

        # Run analysis — must terminate
        result = dataflow_analysis(
            root,
            param_constraints={
                buf_a.id: {"div_by": 16},
                buf_b.id: {"div_by": 16},
            },
        )

        # inner_val must inherit buf_a's alias set
        pred_a = result[buf_a.id]
        pred_inner = result[inner_val.id]

        assert (pred_inner.alias_set & pred_a.alias_set) != 0, (
            f"inner_val.alias_set={pred_inner.alias_set:#010b} should overlap buf_a.alias_set={pred_a.alias_set:#010b}"
        )

    def test_loop_does_not_alias_unrelated_buf(self):
        """Loop-carried value derived from buf_a must not alias buf_b."""
        root = Block()
        buf_a = _make_buffer_value("buf_a", id_=0)
        buf_b = _make_buffer_value("buf_b", id_=1)
        root.params = [buf_a, buf_b]

        body_var = Value(id=10, type=_global_type(), name="body_var")
        inner_val = Value(id=11, type=_global_type(), name="inner_val")

        body_block = Block(params=[body_var])
        inner_op = _Assign(src=body_var)
        inner_op.results = (inner_val,)
        body_block.append(inner_op)

        loop_op = Loop(
            start=None,
            stop=None,
            step=None,
            init=[buf_a],
            body=body_block,
        )
        root.append(loop_op)

        result = dataflow_analysis(
            root,
            param_constraints={
                buf_a.id: {"div_by": 16},
                buf_b.id: {"div_by": 16},
            },
        )

        pred_b = result[buf_b.id]
        pred_inner = result[inner_val.id]

        assert (pred_inner.alias_set & pred_b.alias_set) == 0, "inner_val should not alias buf_b"


# ---------------------------------------------------------------------------
# 5. PassContext integration
# ---------------------------------------------------------------------------


class TestDataflowPass:
    def test_dataflow_pass_stashes_result(self):
        """dataflow_pass() must stash the DataflowResult on ctx.results['dataflow']."""
        root = Block()
        buf_a = _make_buffer_value("buf_a", id_=0)
        root.params = [buf_a]

        ctx = PassContext()
        ctx.results["param_constraints"] = {buf_a.id: {"div_by": 16}}
        dataflow_pass(root, ctx)

        assert "dataflow" in ctx.results
        assert isinstance(ctx.results["dataflow"], DataflowResult)

    def test_dataflow_result_accessible_after_pass(self):
        """The stashed DataflowResult should contain buf_a's predicate."""
        root = Block()
        buf_a = _make_buffer_value("buf_a", id_=0)
        root.params = [buf_a]

        ctx = PassContext()
        ctx.results["param_constraints"] = {buf_a.id: {"div_by": 16}}
        dataflow_pass(root, ctx)

        df: DataflowResult = ctx.results["dataflow"]
        pred = df[buf_a.id]
        assert isinstance(pred, DataPredicate)
        assert pred.div_by == 16


# ---------------------------------------------------------------------------
# 6. DataPredicate.unify contract
# ---------------------------------------------------------------------------


class TestDataPredicateUnify:
    def test_unify_alias_sets_or_together(self):
        a = DataPredicate(alias_set=0b01, div_by=16, may_alias_internally=False)
        b = DataPredicate(alias_set=0b10, div_by=16, may_alias_internally=False)
        u = a.unify(b)
        assert u.alias_set == 0b11

    def test_unify_div_by_takes_gcd(self):
        a = DataPredicate(alias_set=1, div_by=16, may_alias_internally=False)
        b = DataPredicate(alias_set=1, div_by=4, may_alias_internally=False)
        u = a.unify(b)
        assert u.div_by == gcd(16, 4)

    def test_unify_may_alias_internally_ors(self):
        a = DataPredicate(alias_set=1, div_by=1, may_alias_internally=False)
        b = DataPredicate(alias_set=1, div_by=1, may_alias_internally=True)
        u = a.unify(b)
        assert u.may_alias_internally is True

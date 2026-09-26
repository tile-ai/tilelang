"""Tests for the token_order pass.

The pass walks a TileIR Block, uses the DataflowResult (alias sets) from
the dataflow analysis, and produces a TokenPlan: a mapping from each memory op (by
identity) to the set of prior memory ops it must wait on.

Ordering rules (LAST_OP / LAST_STORE):
  - STORE depends on LAST_OP of its alias (WAW + WAR): every prior op
    on the same alias must complete before the store.
  - LOAD depends on LAST_STORE of its alias (RAW): only the most recent
    WRITE to the same alias must complete; earlier LOADs are NOT in the
    dependency set.
  - DISJOINT aliases: no edge between stores (or loads) on different buffers.
  - LOAD after LOAD on the same alias: no edge (load-load overlap is safe).

The plan is a dict[int, frozenset[int]] mapping id(op) → frozenset(id(dep)).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from tilelang.tileir.ir.types import TileType, MemSpace, dtype
from tilelang.tileir.ir.value import Block, Value
from tilelang.tileir.ir.ops import (
    TileOp,
    Effect,
    buffer_operand,
)
from tilelang.tileir.passes.base import PassContext
from tilelang.tileir.passes.dataflow import dataflow_pass
from tilelang.tileir.passes.token_order import token_order_pass, TokenPlan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _global_type(dt: str = "float16", shape: tuple = (128, 128)) -> TileType:
    return TileType(dtype=dtype(dt), shape=shape, space=MemSpace.GLOBAL, layout=None)


def _reg_type(dt: str = "float16", shape: tuple = (16, 16)) -> TileType:
    return TileType(dtype=dtype(dt), shape=shape, space=MemSpace.REGISTER, layout=None)


def _make_buf(name: str, id_: int, dt: str = "float16") -> Value:
    return Value(id=id_, type=_global_type(dt), name=name)


# Minimal synthetic ops for testing — we only need READ/WRITE effects
# and a single buffer_operand so alias resolution works.


@dataclasses.dataclass(eq=False)
class _FakeLoad(TileOp, opcode="fake_load_tok", effect=Effect.READ):
    """Synthetic READ op with a single buffer_operand (for alias tests)."""

    src: Any = buffer_operand(effect=Effect.READ)


@dataclasses.dataclass(eq=False)
class _FakeStore(TileOp, opcode="fake_store_tok", effect=Effect.WRITE):
    """Synthetic WRITE op with a single buffer_operand (for alias tests)."""

    dst: Any = buffer_operand(effect=Effect.WRITE)


def _run_passes(root: Block, buf_values: list[Value]) -> tuple[PassContext, TokenPlan]:
    """Run dataflow then token_order on root; return (ctx, plan)."""
    ctx = PassContext()
    ctx.results["param_constraints"] = {v.id: {"div_by": 16} for v in buf_values}
    dataflow_pass(root, ctx)
    token_order_pass(root, ctx)
    plan: TokenPlan = ctx.results["token_order"]
    return ctx, plan


# ---------------------------------------------------------------------------
# Test 1: WAW — two stores to the SAME alias are ordered
# ---------------------------------------------------------------------------


class TestWAW:
    def test_second_store_depends_on_first(self):
        """WAW: store2 to same alias must wait for store1 to complete."""
        root = Block()
        buf = _make_buf("buf", id_=0)
        root.params = [buf]

        store1 = _FakeStore(dst=buf)
        store2 = _FakeStore(dst=buf)
        root.append(store1)
        root.append(store2)

        _, plan = _run_passes(root, [buf])

        # store1 has no predecessor (it's first)
        deps1 = plan.deps_for(store1)
        assert store2 not in deps1, "store1 should NOT depend on store2"

        # store2 must depend on store1 (WAW)
        deps2 = plan.deps_for(store2)
        assert store1 in deps2, f"store2 should depend on store1 (WAW); got deps={deps2}"


# ---------------------------------------------------------------------------
# Test 2: RAW — load after store to same alias gets an edge
# ---------------------------------------------------------------------------


class TestRAW:
    def test_load_depends_on_prior_store(self):
        """RAW: a load must wait for the most recent store to the same alias."""
        root = Block()
        buf = _make_buf("buf", id_=0)
        root.params = [buf]

        store = _FakeStore(dst=buf)
        load = _FakeLoad(src=buf)
        root.append(store)
        root.append(load)

        _, plan = _run_passes(root, [buf])

        deps = plan.deps_for(load)
        assert store in deps, f"load should depend on prior store (RAW); got deps={deps}"

    def test_store_has_no_predecessor_when_first(self):
        """A store that is the first op on its alias has no predecessor."""
        root = Block()
        buf = _make_buf("buf", id_=0)
        root.params = [buf]

        store = _FakeStore(dst=buf)
        root.append(store)

        _, plan = _run_passes(root, [buf])

        deps = plan.deps_for(store)
        assert len(deps) == 0, f"first store should have no deps; got {deps}"


# ---------------------------------------------------------------------------
# Test 3: DISJOINT — stores to different aliases get NO edge
# ---------------------------------------------------------------------------


class TestDisjoint:
    def test_stores_to_disjoint_aliases_have_no_edge(self):
        """Stores to buffers with distinct alias sets are independent."""
        root = Block()
        buf_a = _make_buf("buf_a", id_=0)
        buf_b = _make_buf("buf_b", id_=1)
        root.params = [buf_a, buf_b]

        store_a = _FakeStore(dst=buf_a)
        store_b = _FakeStore(dst=buf_b)
        root.append(store_a)
        root.append(store_b)

        _, plan = _run_passes(root, [buf_a, buf_b])

        deps_a = plan.deps_for(store_a)
        deps_b = plan.deps_for(store_b)

        assert store_b not in deps_a, "store_a should NOT depend on store_b"
        assert store_a not in deps_b, "store_b should NOT depend on store_a"


# ---------------------------------------------------------------------------
# Test 4: LAL — load after load on same alias gets NO edge (overlap allowed)
# ---------------------------------------------------------------------------


class TestLoadAfterLoad:
    def test_load_after_load_no_edge(self):
        """Load-load on the same alias: no ordering edge (overlap is safe)."""
        root = Block()
        buf = _make_buf("buf", id_=0)
        root.params = [buf]

        load1 = _FakeLoad(src=buf)
        load2 = _FakeLoad(src=buf)
        root.append(load1)
        root.append(load2)

        _, plan = _run_passes(root, [buf])

        deps2 = plan.deps_for(load2)
        assert load1 not in deps2, f"load2 should NOT depend on load1 (load-load is safe); got deps={deps2}"

    def test_load_after_load_both_have_no_deps_without_prior_store(self):
        """Two loads with no prior store: both have empty dep sets."""
        root = Block()
        buf = _make_buf("buf", id_=0)
        root.params = [buf]

        load1 = _FakeLoad(src=buf)
        load2 = _FakeLoad(src=buf)
        root.append(load1)
        root.append(load2)

        _, plan = _run_passes(root, [buf])

        assert len(plan.deps_for(load1)) == 0
        assert len(plan.deps_for(load2)) == 0

    def test_load_after_store_then_load(self):
        """store → load1 → load2: load1 depends on store (RAW), load2 does NOT depend on load1."""
        root = Block()
        buf = _make_buf("buf", id_=0)
        root.params = [buf]

        store = _FakeStore(dst=buf)
        load1 = _FakeLoad(src=buf)
        load2 = _FakeLoad(src=buf)
        root.append(store)
        root.append(load1)
        root.append(load2)

        _, plan = _run_passes(root, [buf])

        # load1 depends on store (RAW)
        deps1 = plan.deps_for(load1)
        assert store in deps1, "load1 should depend on store (RAW)"

        # load2 also depends on store (RAW), NOT on load1
        deps2 = plan.deps_for(load2)
        assert store in deps2, "load2 should depend on store (RAW)"
        assert load1 not in deps2, "load2 should NOT depend on load1 (load-load overlap)"

    def test_arbitrary_buffer_names_use_explicit_schema_effects(self):
        @dataclasses.dataclass(eq=False)
        class _FakeTransfer(TileOp, opcode="fake_transfer_tok", effect=Effect.READWRITE):
            arbitrary_input_name: Any = buffer_operand(effect=Effect.READ)
            arbitrary_output_name: Any = buffer_operand(effect=Effect.WRITE)

        root = Block()
        shared_input = _make_buf("shared_input", id_=0)
        output_a = _make_buf("output_a", id_=1)
        output_b = _make_buf("output_b", id_=2)
        root.params = [shared_input, output_a, output_b]

        transfer_a = _FakeTransfer(
            arbitrary_input_name=shared_input,
            arbitrary_output_name=output_a,
        )
        transfer_b = _FakeTransfer(
            arbitrary_input_name=shared_input,
            arbitrary_output_name=output_b,
        )
        root.append(transfer_a)
        root.append(transfer_b)

        _, plan = _run_passes(root, [shared_input, output_a, output_b])

        assert transfer_a not in plan.deps_for(transfer_b)


# ---------------------------------------------------------------------------
# Test 5: WAR — load then store to the SAME alias: store depends on the load
# ---------------------------------------------------------------------------


class TestWAR:
    def test_store_depends_on_prior_load(self):
        """WAR: a store must wait for a prior load on the same alias to finish."""
        root = Block()
        buf = _make_buf("buf", id_=0)
        root.params = [buf]

        load = _FakeLoad(src=buf)
        store = _FakeStore(dst=buf)
        root.append(load)
        root.append(store)

        _, plan = _run_passes(root, [buf])

        deps = plan.deps_for(store)
        assert load in deps, f"store should depend on prior load (WAR); got deps={deps}"

    def test_load_has_no_dep_before_store(self):
        """WAR: the load that precedes the store has no predecessor of its own."""
        root = Block()
        buf = _make_buf("buf", id_=0)
        root.params = [buf]

        load = _FakeLoad(src=buf)
        store = _FakeStore(dst=buf)
        root.append(load)
        root.append(store)

        _, plan = _run_passes(root, [buf])

        deps_load = plan.deps_for(load)
        assert len(deps_load) == 0, f"load (first op on alias) should have no deps; got {deps_load}"


# ---------------------------------------------------------------------------
# Test 6: TokenPlan stashed on ctx
# ---------------------------------------------------------------------------


class TestTokenPlanStash:
    def test_token_order_pass_stashes_plan(self):
        """token_order_pass must stash a TokenPlan under ctx.results['token_order']."""
        root = Block()
        buf = _make_buf("buf", id_=0)
        root.params = [buf]

        ctx = PassContext()
        ctx.results["param_constraints"] = {buf.id: {"div_by": 16}}
        dataflow_pass(root, ctx)
        token_order_pass(root, ctx)

        assert "token_order" in ctx.results
        assert isinstance(ctx.results["token_order"], TokenPlan)

    def test_token_order_requires_dataflow(self):
        """token_order_pass raises if dataflow result is missing from ctx."""
        root = Block()
        buf = _make_buf("buf", id_=0)
        root.params = [buf]

        ctx = PassContext()  # no dataflow result
        with pytest.raises((KeyError, ValueError, RuntimeError)):
            token_order_pass(root, ctx)

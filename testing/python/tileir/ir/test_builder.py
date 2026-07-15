"""Tests for IRBuilder."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tilelang.tileir.ir.types import TileType, MemSpace, dtype
from tilelang.tileir.ir.value import Value, Block
from tilelang.tileir.ir.ops import TileOp, Effect, operand, buffer_operand, attribute
from tilelang.tileir.ir.builder import IRBuilder


# ---------------------------------------------------------------------------
# Minimal fake ops defined IN the test (independent of the concrete op catalog)
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class _FakeStore(TileOp, opcode="fake_store", effect=Effect.WRITE):
    """A write op with one buffer operand and one value operand."""

    dst: object = buffer_operand(effect=Effect.WRITE)
    val: object = operand()


@dataclass(eq=False)
class _FakePure(TileOp, opcode="fake_pure", effect=Effect.NONE):
    """A pure op that produces one result; no operands required."""

    scale: float = attribute(default=1.0)


@dataclass(eq=False)
class _FakeBadVerify(TileOp, opcode="fake_bad_verify", effect=Effect.NONE):
    """An op whose verify() always raises — used to assert create() propagates."""

    scale: float = attribute(default=1.0)

    def verify(self) -> None:
        raise ValueError("_FakeBadVerify.verify() intentionally raised")


# ---------------------------------------------------------------------------
# Helper: a TileType we can use for result allocation
# ---------------------------------------------------------------------------


def _fp16_tile() -> TileType:
    return TileType(
        dtype=dtype("float16"),
        shape=(16, 16),
        space=MemSpace.REGISTER,
        layout=None,
    )


# ---------------------------------------------------------------------------
# IRBuilder construction
# ---------------------------------------------------------------------------


def test_builder_has_module_block():
    b = IRBuilder()
    mb = b.module_block()
    assert isinstance(mb, Block)


def test_builder_current_block_is_module_block():
    b = IRBuilder()
    assert b.block is b.module_block()


# ---------------------------------------------------------------------------
# create() — basic append
# ---------------------------------------------------------------------------


def test_create_appends_op_to_current_block():
    b = IRBuilder()
    op = _FakeStore(dst="D", val="V")
    b.create(op)
    assert b.block.ops == [op]


def test_create_returns_the_op():
    b = IRBuilder()
    op = _FakeStore(dst="D", val="V")
    returned = b.create(op)
    assert returned is op


def test_create_no_results_by_default():
    b = IRBuilder()
    op = _FakeStore(dst="D", val="V")
    b.create(op)
    assert op.results == ()


# ---------------------------------------------------------------------------
# create() — result allocation
# ---------------------------------------------------------------------------


def test_create_allocates_results():
    b = IRBuilder()
    ty = _fp16_tile()
    op = _FakePure()
    b.create(op, result_types=(ty,))
    assert len(op.results) == 1
    v = op.results[0]
    assert isinstance(v, Value)
    assert v.type is ty


def test_create_allocates_multiple_results():
    b = IRBuilder()
    ty = _fp16_tile()
    op = _FakePure()
    b.create(op, result_types=(ty, ty))
    assert len(op.results) == 2
    assert all(isinstance(v, Value) for v in op.results)
    assert all(v.type is ty for v in op.results)


def test_create_result_ids_are_unique():
    b = IRBuilder()
    ty = _fp16_tile()
    op1 = _FakePure()
    op2 = _FakePure()
    b.create(op1, result_types=(ty,))
    b.create(op2, result_types=(ty,))
    # IDs must be strictly increasing across builder calls.
    assert op1.results[0].id < op2.results[0].id


def test_create_result_ids_unique_within_op():
    b = IRBuilder()
    ty = _fp16_tile()
    op = _FakePure()
    b.create(op, result_types=(ty, ty))
    id0, id1 = op.results[0].id, op.results[1].id
    assert id0 != id1


# ---------------------------------------------------------------------------
# create() — verify() is called
# ---------------------------------------------------------------------------


def test_create_calls_verify_and_propagates_error():
    b = IRBuilder()
    op = _FakeBadVerify()
    with pytest.raises(ValueError, match="intentionally raised"):
        b.create(op)
    # Op must NOT have been appended if verify raises.
    assert b.block.ops == []


# ---------------------------------------------------------------------------
# block_scope() — context manager
# ---------------------------------------------------------------------------


def test_block_scope_yields_fresh_block():
    b = IRBuilder()
    original = b.block
    with b.block_scope() as inner:
        assert isinstance(inner, Block)
        assert inner is not original
        assert b.block is inner


def test_block_scope_restores_on_exit():
    b = IRBuilder()
    original = b.block
    with b.block_scope():
        pass
    assert b.block is original


def test_block_scope_ops_go_into_inner_block():
    b = IRBuilder()
    original = b.block
    with b.block_scope() as inner:
        op = _FakeStore(dst="D", val="V")
        b.create(op)
        assert inner.ops == [op]
    # Nothing appended to the outer block.
    assert original.ops == []


def test_block_scope_with_params():
    b = IRBuilder()
    ty = _fp16_tile()
    v = Value(id=0, type=ty)
    with b.block_scope(params=[v]) as inner:
        assert inner.params == [v]


def test_block_scope_restores_on_exception():
    b = IRBuilder()
    original = b.block
    try:
        with b.block_scope():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert b.block is original


# ---------------------------------------------------------------------------
# IRBuilder from ir package __init__
# ---------------------------------------------------------------------------


def test_irbuilder_importable_from_ir():
    from tilelang.tileir.ir import IRBuilder as IB  # noqa: F401

    assert IB is IRBuilder

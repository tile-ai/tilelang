"""Tests for TileOp schema base machinery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from tilelang.tileir.ir.ops import (
    TileOp,
    Effect,
    Reduce,
    operand,
    buffer_operand,
    attribute,
    nested_block,
)
from tilelang.tileir.ir.value import Block


# ---------------------------------------------------------------------------
# Minimal concrete ops used across tests
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class _FakeStore(TileOp, opcode="fake_store", effect=Effect.WRITE):
    dst: object = buffer_operand(effect=Effect.WRITE)
    val: object = operand()
    flag: bool = attribute(default=False)


@dataclass(eq=False)
class _FakeLoad(TileOp, opcode="fake_load", effect=Effect.READ):
    src: object = buffer_operand(effect=Effect.READ)
    idx: object = operand(default=None)


@dataclass(eq=False)
class _FakePure(TileOp, opcode="fake_pure", effect=Effect.NONE):
    a: object = operand()
    b: object = operand(default=None)
    scale: float = attribute(default=1.0)


@dataclass(eq=False)
class _FakeLoop(TileOp, opcode="fake_loop", effect=Effect.NONE):
    body: object = nested_block()


@dataclass(eq=False)
class _FakeTerm(TileOp, opcode="fake_term", terminator=True, effect=Effect.NONE):
    retval: object = operand(default=None)


@dataclass(eq=False)
class _FakeWithClassVar(TileOp, opcode="fake_cv", effect=Effect.NONE):
    # A subclass may declare its own ClassVar metadata; it must be skipped
    # by __init_subclass__ (not mistaken for a marker field) and excluded
    # from the generated __init__.
    tag: ClassVar[str] = "constant"
    x: object = operand()


# ---------------------------------------------------------------------------
# Effect enum
# ---------------------------------------------------------------------------


def test_effect_enum_members():
    # Membership, not truthiness — robust if Effect ever becomes an IntEnum
    # whose NONE member is 0 (which is falsy under `assert`).
    assert Effect.NONE in Effect
    assert Effect.READ in Effect
    assert Effect.WRITE in Effect
    assert Effect.READWRITE in Effect
    # All four and only four members
    assert set(e.name for e in Effect) == {"NONE", "READ", "WRITE", "READWRITE"}


# ---------------------------------------------------------------------------
# __init_subclass__ class-level introspection
# ---------------------------------------------------------------------------


def test_init_subclass_categorizes_fields():
    assert _FakeStore._opcode == "fake_store"
    assert _FakeStore.memory_effect == Effect.WRITE
    assert _FakeStore._buffer_operand_names == ["dst"]
    assert _FakeStore._operand_names == ["val"]
    assert _FakeStore._attr_names == ["flag"]
    assert _FakeStore._block_names == []


def test_opcode_stored():
    assert _FakeLoad._opcode == "fake_load"
    assert _FakePure._opcode == "fake_pure"


def test_terminator_flag():
    assert _FakeTerm._terminator is True
    assert _FakeStore._terminator is False
    assert _FakePure._terminator is False


def test_memory_effect_classification():
    assert _FakeStore.memory_effect == Effect.WRITE
    assert _FakeLoad.memory_effect == Effect.READ
    assert _FakePure.memory_effect == Effect.NONE


def test_multiple_operands_order():
    # _FakePure: two operands a, b in declaration order; no buffer_operands
    assert _FakePure._operand_names == ["a", "b"]
    assert _FakePure._buffer_operand_names == []
    assert _FakePure._attr_names == ["scale"]


def test_nested_block_categorized():
    assert _FakeLoop._block_names == ["body"]
    assert _FakeLoop._operand_names == []
    assert _FakeLoop._buffer_operand_names == []
    assert _FakeLoop._attr_names == []


# ---------------------------------------------------------------------------
# Instance traversal helpers
# ---------------------------------------------------------------------------


def test_instance_traversal_store():
    op = _FakeStore(dst="D", val="V", flag=True)
    assert op.buffer_operands() == ("D",)
    assert op.operands() == ("V",)


def test_buffer_effects_come_from_explicit_schema_metadata():
    @dataclass(eq=False)
    class _FakeTransfer(TileOp, opcode="fake_transfer", effect=Effect.READWRITE):
        arbitrary_input_name: object = buffer_operand(effect=Effect.READ)
        arbitrary_output_name: object = buffer_operand(effect=Effect.WRITE)
        arbitrary_update_name: object = buffer_operand(effect=Effect.READWRITE)

    op = _FakeTransfer(
        arbitrary_input_name="input",
        arbitrary_output_name="output",
        arbitrary_update_name="update",
    )

    assert op.buffer_effects() == (
        ("input", Effect.READ),
        ("output", Effect.WRITE),
        ("update", Effect.READWRITE),
    )


def test_reduce_destination_is_readwrite_for_clear_false_accumulation():
    op = Reduce(src="input", dst="accumulator", op="sum", axis=0, clear=False)

    assert op.buffer_effects() == (
        ("input", Effect.READ),
        ("accumulator", Effect.READWRITE),
    )


def test_buffer_operand_requires_an_explicit_effect():
    with pytest.raises(TypeError, match="missing.*effect"):
        buffer_operand()


def test_instance_traversal_pure():
    op = _FakePure(a="X", b="Y", scale=2.0)
    assert op.operands() == ("X", "Y")
    assert op.buffer_operands() == ()


def test_instance_traversal_loop():
    blk = Block()
    op = _FakeLoop(body=blk)
    assert op.nested_blocks() == (blk,)
    assert op.operands() == ()
    assert op.buffer_operands() == ()


def test_default_operand_none():
    op = _FakeLoad(src="S")
    assert op.buffer_operands() == ("S",)
    assert op.operands() == (None,)


# ---------------------------------------------------------------------------
# results and loc fields
# ---------------------------------------------------------------------------


def test_results_and_loc_defaults():
    op = _FakePure(a="A", b="B")
    assert op.results == ()
    assert op.loc is None


def test_results_settable():
    op = _FakePure(a="A", b="B", results=("r1", "r2"))
    assert op.results == ("r1", "r2")


# ---------------------------------------------------------------------------
# verify() and emit_mlir() defaults
# ---------------------------------------------------------------------------


def test_verify_default_noop():
    op = _FakePure(a="A", b="B")
    op.verify()  # must not raise


def test_emit_mlir_default_raises():
    op = _FakePure(a="A", b="B")
    with pytest.raises(NotImplementedError):
        op.emit_mlir(ctx=None)


# ---------------------------------------------------------------------------
# markers compose correctly with @dataclass
# ---------------------------------------------------------------------------


def test_dataclass_init_works():
    """@dataclass(eq=False) must coexist with the markers."""
    op = _FakeStore(dst="D", val="V", flag=True)
    assert op.dst == "D"
    assert op.val == "V"
    assert op.flag is True


def test_default_attribute():
    op = _FakeStore(dst="D", val="V")
    assert op.flag is False


def test_identity_equality():
    """eq=False: two separate instances are never equal."""
    op1 = _FakePure(a="A", b="B")
    op2 = _FakePure(a="A", b="B")
    assert op1 is not op2
    assert op1 != op2


# ---------------------------------------------------------------------------
# ClassVar on a subclass is skipped (not treated as a marker field)
# ---------------------------------------------------------------------------


def test_subclass_classvar_skipped():
    # The ClassVar field must NOT appear in any name list...
    assert _FakeWithClassVar._operand_names == ["x"]
    assert _FakeWithClassVar._attr_names == []
    assert _FakeWithClassVar._buffer_operand_names == []
    assert _FakeWithClassVar._block_names == []
    # ...and must NOT be a constructor parameter (it stays a class constant).
    op = _FakeWithClassVar(x="V")
    assert op.operands() == ("V",)
    assert op.tag == "constant"

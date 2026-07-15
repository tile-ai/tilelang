"""Tests for tilelang.tileir.ir.value.

Tests cover:
  - Value identity equality (two distinct Values never compare equal)
  - fresh_value requires explicit counter (no module global)
  - Block.append preserves op order
  - Region wraps a single Block
  - Value.def_op and .name attributes
"""

from tilelang.tileir.ir.types import TileType, MemSpace, dtype
from tilelang.tileir.ir.value import Block, Region, fresh_value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_type(shape=(128, 64)) -> TileType:
    return TileType(dtype("float16"), shape, MemSpace.REGISTER, None)


class Counter:
    """Minimal counter object matching the expected fresh_value interface."""

    def __init__(self):
        self.value = 0

    def next(self) -> int:
        n = self.value
        self.value += 1
        return n


# ---------------------------------------------------------------------------
# Value tests
# ---------------------------------------------------------------------------


def test_value_has_required_attributes():
    counter = Counter()
    ty = make_type()
    v = fresh_value(counter, ty, name="x")
    assert isinstance(v.id, int)
    assert v.type is ty
    assert v.name == "x"
    assert v.def_op is None


def test_value_name_defaults_to_none():
    counter = Counter()
    ty = make_type()
    v = fresh_value(counter, ty)
    assert v.name is None


def test_value_ids_are_monotonically_increasing():
    counter = Counter()
    ty = make_type()
    ids = [fresh_value(counter, ty).id for _ in range(5)]
    assert ids == sorted(ids)
    assert len(set(ids)) == 5  # all distinct


def test_value_identity_equality():
    """Two distinct Values are never equal even with identical type/name."""
    counter = Counter()
    ty = make_type()
    a = fresh_value(counter, ty, name="x")
    b = fresh_value(counter, ty, name="x")
    assert a is not b
    assert a != b


def test_separate_counters_are_independent():
    """Each counter object starts from 0 independently."""
    c1 = Counter()
    c2 = Counter()
    ty = make_type()
    v1 = fresh_value(c1, ty)
    v2 = fresh_value(c2, ty)
    # Both start at 0; ids may coincide, but Values are distinct objects
    assert v1 is not v2
    assert v1.id == 0
    assert v2.id == 0


def test_value_def_op_can_be_set():
    """def_op is settable after construction (used by op constructors)."""
    counter = Counter()
    ty = make_type()
    v = fresh_value(counter, ty)
    sentinel = object()
    v.def_op = sentinel
    assert v.def_op is sentinel


# ---------------------------------------------------------------------------
# Block tests
# ---------------------------------------------------------------------------


class StubOp:
    """Minimal stand-in for a TileOp — used to test Block.append ordering."""

    def __init__(self, tag: str):
        self.tag = tag


def test_block_starts_empty():
    block = Block()
    assert block.params == []
    assert block.ops == []


def test_block_append_preserves_order():
    block = Block()
    ops = [StubOp(f"op{i}") for i in range(4)]
    for op in ops:
        block.append(op)
    assert block.ops == ops
    assert [o.tag for o in block.ops] == ["op0", "op1", "op2", "op3"]


def test_block_params_are_values():
    counter = Counter()
    ty = make_type()
    params = [fresh_value(counter, ty, name=f"p{i}") for i in range(3)]
    block = Block(params=params)
    # list == compares elementwise; Value has identity equality, so this
    # confirms Block holds the *same* Value objects, not just equal content.
    assert block.params == params
    for i, p in enumerate(block.params):
        assert p.name == f"p{i}"


def test_block_params_default_empty():
    block = Block()
    assert block.params == []


def test_block_ops_list_is_mutable():
    block = Block()
    op = StubOp("x")
    block.append(op)
    assert len(block.ops) == 1
    block.ops.clear()
    assert len(block.ops) == 0


# ---------------------------------------------------------------------------
# Region tests
# ---------------------------------------------------------------------------


def test_region_wraps_single_block():
    block = Block()
    region = Region(block)
    assert region.block is block


def test_region_block_is_accessible():
    counter = Counter()
    ty = make_type()
    block = Block(params=[fresh_value(counter, ty)])
    block.append(StubOp("load"))
    region = Region(block)
    assert len(region.block.params) == 1
    assert len(region.block.ops) == 1

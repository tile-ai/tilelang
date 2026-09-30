"""Focused tests for per-buffer effects in loop-carry analysis."""

from __future__ import annotations

import dataclasses
from typing import Any

from tilelang.tileir.ir.ops import Copy, Effect, IfElse, Loop, TileOp, buffer_operand
from tilelang.tileir.ir.types import MemSpace, TileType, dtype
from tilelang.tileir.ir.value import Block, Value
from tilelang.tileir.passes.base import PassContext
from tilelang.tileir.passes.loop_carry import loop_carry_pass


def _make_reg(name: str, id_: int) -> Value:
    ty = TileType(
        dtype=dtype("float16"),
        shape=(16, 16),
        space=MemSpace.REGISTER,
        layout=None,
    )
    return Value(id=id_, type=ty, name=name)


def test_arbitrary_buffer_names_use_explicit_schema_effects():
    @dataclasses.dataclass(eq=False)
    class _FakeTransfer(TileOp, opcode="fake_transfer_carry", effect=Effect.READWRITE):
        arbitrary_input_name: Any = buffer_operand(effect=Effect.READ)
        arbitrary_output_name: Any = buffer_operand(effect=Effect.WRITE)

    source = _make_reg("source", id_=0)
    scratch = _make_reg("scratch", id_=1)
    body = Block()
    body.append(
        _FakeTransfer(
            arbitrary_input_name=source,
            arbitrary_output_name=scratch,
        )
    )
    loop = Loop(stop=None, body=body)
    root = Block()
    root.append(loop)

    loop_carry_pass(root, PassContext())

    assert id(source) in loop._carry_tile_value_ids
    assert id(scratch) not in loop._carry_tile_value_ids


def test_alias_writes_carry_one_backing_even_when_conditional():
    source, backing, alias, other_alias, output, scratch, untouched = (
        _make_reg(name, i) for i, name in enumerate(("source", "backing", "alias", "other_alias", "output", "scratch", "untouched"))
    )
    branch = Block()
    branch.append(Copy(src=source, dst=alias))
    body = Block()
    body.append(IfElse(cond=None, then_block=branch))
    body.append(Copy(src=backing, dst=output))
    body.append(Copy(src=source, dst=scratch))
    loop = Loop(stop=None, body=body)
    root = Block()
    root.buffer_aliases = {alias: backing, other_alias: backing, _make_reg("unused_view", 7): untouched}
    root.append(loop)

    loop_carry_pass(root, PassContext())

    assert id(backing) in loop._carry_tile_value_ids
    assert not ({id(alias), id(other_alias), id(scratch), id(untouched)} & loop._carry_tile_value_ids)


def test_backing_write_and_alias_live_out_share_storage():
    source, backing, alias, output = (_make_reg(name, i) for i, name in enumerate(("source", "backing", "alias", "output")))
    body = Block()
    body.append(Copy(src=source, dst=backing))
    loop = Loop(stop=None, body=body)
    root = Block()
    root.buffer_aliases = {alias: backing}
    root.append(loop)
    root.append(Copy(src=alias, dst=output))

    loop_carry_pass(root, PassContext())

    assert id(backing) in loop._carry_tile_value_ids
    assert id(alias) not in loop._carry_tile_value_ids

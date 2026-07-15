"""Focused tests for per-buffer effects in loop-carry analysis."""

from __future__ import annotations

import dataclasses
from typing import Any

from tilelang.tileir.ir.ops import Effect, Loop, TileOp, buffer_operand
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

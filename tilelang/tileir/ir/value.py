"""SSA values and block containers for TileIR."""

from __future__ import annotations

from typing import Any, Protocol

from tilelang.tileir.ir.types import TileType

__all__ = ["Value", "Block", "Region", "fresh_value"]


# Counter protocol


class IdCounter(Protocol):
    """Structural type for the id source consumed by ``fresh_value``."""

    def next(self) -> int: ...


# Value


class Value:
    """An identity-based SSA value produced by one defining operation."""

    __slots__ = ("id", "type", "name", "def_op")

    def __init__(self, id: int, type: TileType, name: str | None = None) -> None:
        self.id: int = id
        self.type: TileType = type
        self.name: str | None = name
        self.def_op: Any = None

    def __repr__(self) -> str:
        tag = f"[{self.name}]" if self.name else ""
        return f"Value(id={self.id}{tag}, type={self.type})"


# Block


class Block:
    """An ordered operation sequence with optional block parameters."""

    def __init__(self, params: list[Value] | None = None) -> None:
        self.params: list[Value] = list(params) if params is not None else []
        self.alloc_buffers: list[Value] = []
        self.ops: list[Any] = []
        # Block/thread index values keyed by axis name.
        self.index_values: dict[str, Value] = {}
        # Thread-axis extents keyed by axis name ("tx", "ty", "tz").
        self.thread_extents: dict[str, int] = {}
        # Block-axis extents keyed by axis name ("bx", "by", "bz").
        self.block_extents: dict[str, int] = {}
        # Dynamic shape placeholders mapped to their buffer and dimension.
        self.shape_bindings: dict[Value, tuple[Value, int]] = {}
        # Reshape-view aliases mapped to their base buffers.
        self.buffer_aliases: dict[Value, Value] = {}
        # Addressable scratch buffers demoted from tile-inexpressible storage.
        self.alloca_buffers: dict[Value, tuple[tuple, str]] = {}

    def append(self, op: Any) -> None:
        """Append *op* to the end of this block's op sequence."""
        self.ops.append(op)

    def __repr__(self) -> str:
        return f"Block(params={self.params!r}, ops=[{len(self.ops)} op(s)])"


# Region


class Region:
    """A single-block control-flow region."""

    __slots__ = ("block",)

    def __init__(self, block: Block) -> None:
        self.block: Block = block

    def __repr__(self) -> str:
        return f"Region({self.block!r})"


# fresh_value


def fresh_value(counter: IdCounter, type: TileType, name: str | None = None) -> Value:
    """Stamp a new ``Value`` with the next id from *counter*."""
    return Value(counter.next(), type, name)

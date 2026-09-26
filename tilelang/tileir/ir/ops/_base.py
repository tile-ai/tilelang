"""Schema and field markers for typed TileIR operations."""

from __future__ import annotations

import dataclasses
import enum
import typing
from typing import Any, ClassVar

from tilelang.tileir.ir.value import Block, Value


# Effect enum


class Effect(enum.Enum):
    """Coarse memory-effect classification for a ``TileOp`` subclass."""

    NONE = "none"
    READ = "read"
    WRITE = "write"
    READWRITE = "readwrite"


# Field-kind enum + metadata key


class _FieldKind(enum.IntEnum):
    OPERAND = 0
    BUFFER_OPERAND = 1
    ATTRIBUTE = 2
    NESTED_BLOCK = 3


_KIND: str = "tileir_field_kind"
_BUFFER_EFFECT: str = "tileir_buffer_effect"


def _is_classvar(annotation: Any) -> bool:
    """Return whether *annotation* denotes ``typing.ClassVar``."""
    if isinstance(annotation, str):
        return annotation == "ClassVar" or annotation.startswith(("ClassVar[", "ClassVar "))
    return typing.get_origin(annotation) is ClassVar


# Field markers


def operand(*, default: Any = dataclasses.MISSING) -> dataclasses.Field:
    """Mark a field as a plain SSA-value operand."""
    return dataclasses.field(
        default=default,
        metadata={_KIND: _FieldKind.OPERAND},
        kw_only=True,
    )


def buffer_operand(
    *,
    effect: Effect,
    default: Any = dataclasses.MISSING,
) -> dataclasses.Field:
    """Mark a buffer operand and its per-buffer memory effect."""
    if not isinstance(effect, Effect):
        raise TypeError(f"buffer_operand effect must be an Effect, got {effect!r}")
    return dataclasses.field(
        default=default,
        metadata={
            _KIND: _FieldKind.BUFFER_OPERAND,
            _BUFFER_EFFECT: effect,
        },
        kw_only=True,
    )


def attribute(*, default: Any = dataclasses.MISSING) -> dataclasses.Field:
    """Mark a field as a compile-time constant attribute."""
    return dataclasses.field(
        default=default,
        metadata={_KIND: _FieldKind.ATTRIBUTE},
        kw_only=True,
    )


def nested_block(*, default: Any = dataclasses.MISSING) -> dataclasses.Field:
    """Mark a field as a nested ``Block`` body."""
    return dataclasses.field(
        default=default,
        metadata={_KIND: _FieldKind.NESTED_BLOCK},
        kw_only=True,
    )


# TileOp base


@dataclasses.dataclass(eq=False)
class TileOp:
    """Abstract base class for all typed TileIR operations.

    Subclass with ``class MyOp(TileOp, opcode="my_op", ...)`` and decorate
    with ``@dataclass(eq=False)``.  Each annotated field on the subclass
    must use one of the marker functions (``operand``, ``buffer_operand``,
    ``attribute``, ``nested_block``).

    Concrete ops must inherit directly from ``TileOp`` because marker fields
    are collected from the subclass's own annotations. Every non-``ClassVar``
    field must use one of the marker functions above.
    """

    results: tuple[Value, ...] = dataclasses.field(default=(), kw_only=True)
    loc: Any = dataclasses.field(default=None, kw_only=True)

    # These are set by __init_subclass__ on every concrete subclass.
    # ClassVar annotations are excluded from @dataclass __init__ generation,
    # so they are never surfaced as constructor parameters.
    _opcode: ClassVar[str]
    _terminator: ClassVar[bool]
    memory_effect: ClassVar[Effect]
    _operand_names: ClassVar[list[str]]
    _buffer_operand_names: ClassVar[list[str]]
    _buffer_operand_effects: ClassVar[list[Effect]]
    _attr_names: ClassVar[list[str]]
    _block_names: ClassVar[list[str]]

    def __init_subclass__(
        cls,
        *,
        opcode: str,
        terminator: bool = False,
        effect: Effect = Effect.NONE,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)

        cls._opcode = opcode
        cls._terminator = terminator
        cls.memory_effect = effect

        operand_names: list[str] = []
        buffer_operand_names: list[str] = []
        buffer_operand_effects: list[Effect] = []
        attr_names: list[str] = []
        block_names: list[str] = []

        # Iterate only the annotations declared directly on *this* subclass,
        # not inherited ones from TileOp (results, loc are already handled).
        for field_name, annotation in cls.__annotations__.items():
            # ClassVar attributes are class-level metadata, not marker fields;
            # @dataclass also excludes them from __init__, so skip them here.
            if _is_classvar(annotation):
                continue
            f = getattr(cls, field_name, None)
            kind = f.metadata.get(_KIND) if isinstance(f, dataclasses.Field) else None
            if kind == _FieldKind.OPERAND:
                operand_names.append(field_name)
            elif kind == _FieldKind.BUFFER_OPERAND:
                buffer_operand_names.append(field_name)
                buffer_operand_effects.append(f.metadata[_BUFFER_EFFECT])
            elif kind == _FieldKind.ATTRIBUTE:
                attr_names.append(field_name)
            elif kind == _FieldKind.NESTED_BLOCK:
                block_names.append(field_name)
            else:
                raise TypeError(
                    f"Field '{field_name}' on {cls.__qualname__} must use one of "
                    "operand(), buffer_operand(), attribute(), or nested_block()."
                )

        cls._operand_names = operand_names
        cls._buffer_operand_names = buffer_operand_names
        cls._buffer_operand_effects = buffer_operand_effects
        cls._attr_names = attr_names
        cls._block_names = block_names

    # Instance traversal helpers

    def operands(self) -> tuple[Any, ...]:
        """Return plain SSA operand values in declaration order."""
        return tuple(getattr(self, n) for n in self._operand_names)

    def buffer_operands(self) -> tuple[Any, ...]:
        """Return memory-reference operand values in declaration order."""
        return tuple(getattr(self, n) for n in self._buffer_operand_names)

    def buffer_effects(self) -> tuple[tuple[Any, Effect], ...]:
        """Return ``(buffer value, effect)`` pairs in declaration order."""
        return tuple((getattr(self, name), effect) for name, effect in zip(self._buffer_operand_names, self._buffer_operand_effects))

    def nested_blocks(self) -> tuple[Block, ...]:
        """Return nested ``Block`` bodies in declaration order."""
        return tuple(getattr(self, n) for n in self._block_names)

    # Verification and MLIR emission (overridden by concrete ops)

    def verify(self) -> None:
        """Validate the op's invariants.

        Default implementation is a no-op.  Concrete ops override to add
        their own checks.
        """

    def emit_mlir(self, ctx: Any) -> Any:
        """Lower this op to MLIR.

        Default raises ``NotImplementedError``.  Concrete ops override this.
        """
        raise NotImplementedError(f"{type(self).__qualname__}.emit_mlir() is not implemented")


# Op catalog
#
# Every concrete op is ``@dataclass(eq=False)`` and inherits directly from
# ``TileOp``.  Each op overrides ``emit_mlir`` to lower itself to MLIR.
#
# Field naming conventions
# - ``operand()``        — plain SSA value (loop bounds, condition, …)
# - ``buffer_operand(effect=...)`` — memory reference with an explicit role
# - ``attribute()``      — compile-time constant (opcode string, flags, …)
# - ``nested_block()``   — a nested ``Block`` body


# Control flow

"""TileOp schema base + field markers for TileIR.

Provides:
  - ``Effect``          — enum describing an op's memory side-effects.
  - ``operand``         — field marker for a plain SSA-value operand.
  - ``buffer_operand``  — field marker for an operand that IS a memory ref.
  - ``attribute``       — field marker for a compile-time constant attribute.
  - ``nested_block``    — field marker for a nested Block body.
  - ``TileOp``          — abstract base class for all typed TileIR operations.

Design notes
------------
The field-marker / ``__init_subclass__`` pattern is mirrored from
cuTile's ``cuda/tile/_ir/ir.py`` (lines ~549-609).

``__init_subclass__`` runs at *class-creation time*, before the
``@dataclass(eq=False)`` decorator has a chance to process the
subclass.  At that moment the class attributes for annotated fields are
still the raw ``dataclasses.Field`` objects returned by the marker
functions.  We read them via ``getattr(cls, field_name, None)`` and
inspect ``f.metadata``.

``buffer_operand`` is an addition beyond cuTile: it marks operands that
are memory references (buffers / tensors) and records each operand's
memory effect.  Analysis passes use that metadata without re-examining
opcode strings or field names.  ``memory_effect`` remains the coarse,
op-level classification.

The concrete op dataclasses are **not** defined here — only the machinery
they depend on.
"""

from __future__ import annotations

import dataclasses
import enum
import typing
from typing import Any, ClassVar

from tilelang.tileir.ir.value import Block, Value


# Effect enum


class Effect(enum.Enum):
    """Coarse memory-effect classification for a ``TileOp`` subclass.

    Together with the set of ``buffer_operand`` fields this is the single
    source of truth for an op's memory semantics.  Analysis passes MUST
    read these instead of pattern-matching on opcode strings.
    """

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
    """Return True if *annotation* denotes ``typing.ClassVar``.

    Because this module uses ``from __future__ import annotations``, the
    values in ``cls.__annotations__`` are *strings*, so a string-prefix
    check covers the common case.  The ``get_origin`` check handles a
    subclass that opts out of stringized annotations.
    """
    if isinstance(annotation, str):
        return annotation == "ClassVar" or annotation.startswith(("ClassVar[", "ClassVar "))
    return typing.get_origin(annotation) is ClassVar


# Field markers


def operand(*, default: Any = dataclasses.MISSING) -> dataclasses.Field:
    """Mark a field as a plain SSA-value operand.

    Parameters
    ----------
    default :
        Optional default value.  Omit to make the field required.
    """
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
    """Mark a field as a memory-reference operand (buffer / tensor).

    Like ``operand`` but semantically denotes that the value IS a memory
    reference.  ``effect`` is the per-buffer memory role consumed by analysis
    passes.  It is required so field names never become implicit semantics.
    """
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
    """Mark a field as a compile-time constant attribute.

    Parameters
    ----------
    default :
        Optional default value.  Omit to make the field required.
    """
    return dataclasses.field(
        default=default,
        metadata={_KIND: _FieldKind.ATTRIBUTE},
        kw_only=True,
    )


def nested_block(*, default: Any = dataclasses.MISSING) -> dataclasses.Field:
    """Mark a field as a nested ``Block`` body.

    Parameters
    ----------
    default :
        Optional default value.  Omit to make the field required.
    """
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

    Subclassing constraints
    -----------------------
    * **Single-level inheritance only.**  ``__init_subclass__`` reads
      ``cls.__annotations__``, which holds only the annotations declared
      directly on the subclass.  An intermediate base's marker fields are
      NOT re-collected, so a two-level hierarchy would silently drop the
      parent's fields from ``operands()`` / ``buffer_operands()`` / etc.
      Concrete ops must inherit directly from ``TileOp``.
    * Every *marker* field must use one of the four marker functions.  A
      ``ClassVar``-annotated attribute on a subclass is permitted (it is
      skipped, not treated as a marker field); any other bare annotation
      raises ``TypeError`` at class-creation time.

    Class attributes (populated by ``__init_subclass__``)
    -----------------------------------------------------
    _opcode : str
    _terminator : bool
    memory_effect : Effect
    _operand_names : list[str]
    _buffer_operand_names : list[str]
    _buffer_operand_effects : list[Effect]
    _attr_names : list[str]
    _block_names : list[str]

    Instance fields (defined on the base dataclass)
    -----------------------------------------------
    results : tuple[Value, ...]
        SSA values produced by this op.  Defaults to ``()``.
    loc : Any | None
        Source location tag.  Defaults to ``None``.
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

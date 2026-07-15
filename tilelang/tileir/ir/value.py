"""SSA values and block containers for TileIR.

Provides:
  - ``Value``       — SSA definition: id, type, optional name, optional def_op.
  - ``Block``       — ordered list of ops with entry params (list[Value]).
  - ``Region``      — single-block region wrapper for control-flow bodies.
  - ``fresh_value`` — factory that stamps a monotonically increasing id from
                      an explicit caller-supplied counter; no module global.

Design notes
------------
* **Value identity equality** — two distinct ``Value`` objects are never equal,
  even when they carry the same type and name.  ``__eq__`` and ``__hash__``
  are intentionally left at the ``object`` defaults (identity-based).

* **No global counter** — ``fresh_value`` requires a counter object with a
  ``next() -> int`` method.  The IRBuilder owns and passes that counter;
  nothing here allocates a module-level id sequence.

* **Loose op typing** — ``Block.ops`` holds ``Any`` items rather than
  ``TileOp`` to avoid coupling this module to the concrete op classes; the
  list annotation can be tightened without changing the runtime behaviour.
"""

from __future__ import annotations

from typing import Any, Protocol

from tilelang.tileir.ir.types import TileType

__all__ = ["Value", "Block", "Region", "fresh_value"]


# Counter protocol


class IdCounter(Protocol):
    """Structural type for the id source ``fresh_value`` consumes.

    Any object exposing ``next() -> int`` satisfies it.  The IRBuilder
    supplies a concrete counter; this module never owns one, so the
    contract is expressed structurally rather than by a base class.
    """

    def next(self) -> int: ...


# Value


class Value:
    """An SSA value produced by a single defining operation.

    Attributes
    ----------
    id : int
        Monotonically increasing integer assigned by ``fresh_value``.
    type : TileType
        The TileIR type of this value.
    name : str | None
        Optional human-readable tag (e.g. the variable name from source).
    def_op : Any | None
        Back-pointer to the ``TileOp`` that defines this value.  Starts as
        ``None``; set by the op constructor after the ``Value`` is created.

    Equality is **identity-based** (inherited from ``object``).  Two ``Value``
    instances with the same id/type are still distinct SSA definitions.
    """

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
    """A sequence of ops with an optional list of entry-block parameters.

    Attributes
    ----------
    params : list[Value]
        Block arguments (entry values for region bodies such as loop vars).
        For the root kernel block these are the GLOBAL buffer entry params
        (position-aligned with ``entry_args`` in ``emit_module``).
    alloc_buffers : list[Value]
        Kernel-local SHARED / REGISTER (non-GLOBAL) buffer Values produced by
        ``alloc_shared`` / ``alloc_fragment``.  Populated by ``lower_kernel``
        from ``LoweringScope._buffers`` for non-GLOBAL entries so that
        ``emit_module`` can materialise them as zero-constant tiles before
        walking the ops.  Empty for nested blocks (loop bodies, if-branches).
    ops : list[Any]
        Ordered sequence of ``TileOp`` instances appended via ``append``.
    """

    def __init__(self, params: list[Value] | None = None) -> None:
        self.params: list[Value] = list(params) if params is not None else []
        self.alloc_buffers: list[Value] = []
        self.ops: list[Any] = []
        # Block/thread index Value objects, keyed by axis name.
        # Populated by sem_to_ir._lower_thread_extent and consumed by
        # emit_module to bind MLIR get_tile_block_id components.
        # Keys: "bx", "by", "bz" for blockIdx; "tx", "ty", "tz" for threadIdx.
        self.index_values: dict[str, Value] = {}
        # Thread-axis extents (int), keyed by axis name ("tx", "ty", "tz").
        # Populated by sem_to_ir._lower_thread_extent alongside index_values.
        # Used by emit_module to emit ct.iota(extent, Int32) for threadIdx.* —
        # this gives each thread its real lane index instead of constant-0,
        # which is required by per-thread SIMT scatter kernels (dequant_gemm).
        self.thread_extents: dict[str, int] = {}
        # Block-axis extents (int), keyed by axis name ("bx", "by", "bz").
        # Populated by sem_to_ir._lower_thread_extent alongside index_values.
        # Used by _lower_threadblock_swizzle_pattern to build the swizzle arithmetic.
        self.block_extents: dict[str, int] = {}
        # Dynamic shape symbol placeholders: placeholder Value → (buffer Value,
        # dim index). A GLOBAL param dim declared as a symbolic name (e.g.
        # ``max_selected_blocks``) already has its runtime value in the entry
        # ABI; emit_module binds each placeholder to the buffer's shape tile.
        # Populated by sem_to_ir.lower_kernel.
        self.shape_bindings: dict[Value, tuple[Value, int]] = {}
        # Reshape-view aliases (T.reshape / T.view): alias buffer Value → base
        # buffer Value. EmitContext.get_tile/set_tile redirect through the base
        # tile with reshapes so both names see one consistent value.
        # Populated by sem_to_ir.lower_kernel.
        self.buffer_aliases: dict[Value, Value] = {}
        # SIMT-demoted scratch buffers: buffer Value → (shape, dtype_name).
        # SHARED buffers accessed in tile-inexpressible ways (data-dependent
        # scatter/atomic indices) become addressable ``alloca global`` memory;
        # emit_module materializes each as an alloca + _BufferInfo so the
        # pointer gather/scatter/atomic machinery applies unchanged.
        # Populated by sem_to_ir.lower_kernel.
        self.alloca_buffers: dict[Value, tuple[tuple, str]] = {}

    def append(self, op: Any) -> None:
        """Append *op* to the end of this block's op sequence."""
        self.ops.append(op)

    def __repr__(self) -> str:
        return f"Block(params={self.params!r}, ops=[{len(self.ops)} op(s)])"


# Region


class Region:
    """A single-block region, used as the body of control-flow ops.

    TileIR uses only single-block regions (no branching within a region body).
    Multi-region control flow is represented by nesting ``Region`` objects.

    Attributes
    ----------
    block : Block
        The one and only block in this region.
    """

    __slots__ = ("block",)

    def __init__(self, block: Block) -> None:
        self.block: Block = block

    def __repr__(self) -> str:
        return f"Region({self.block!r})"


# fresh_value


def fresh_value(counter: IdCounter, type: TileType, name: str | None = None) -> Value:
    """Stamp a new ``Value`` with the next id from *counter*.

    Parameters
    ----------
    counter : IdCounter
        Any object that exposes a ``next() -> int`` method.  The IRBuilder
        supplies this; nothing in this module owns it.
    type : TileType
        The TileIR type to attach to the new value.
    name : str | None
        Optional human-readable label (e.g. the source variable name).

    Returns
    -------
    Value
        A freshly stamped SSA value with identity-based equality.
    """
    return Value(counter.next(), type, name)

"""Parametrized tests for TileIR op catalog.

Gate criteria:
  - Every op in the catalog has a unique ``_opcode``.
  - ``memory_effect`` matches the catalog intent.
  - Memory ops expose the expected ``buffer_operand`` names.
  - The total catalog count matches exactly (no op missing, no extra).
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Import catalog ops (the test will FAIL with ImportError until they exist)
# ---------------------------------------------------------------------------
from tilelang.tileir.ir.ops import (
    Effect,
    # Control flow
    Loop,
    IfElse,
    Break,
    Continue,
    GridSync,
    # Data movement
    Copy,
    TmaCopy,
    Load,
    Store,
    Fill,
    PartitionView,
    # Compute
    Gemm,
    Tcgen05Gemm,
    Reduce,
    Cumsum,
    ThreadAllreduce,
    Elementwise,
    Cast,
    Select,
    RepeatInterleave,
    # Atomic
    AtomicRMW,
    AtomicCAS,
    # Misc
    Barrier,
    DeviceAssert,
    DebugPrint,
    DecodeI4,
    DecodeI2,
    Dp4a,
)

# ---------------------------------------------------------------------------
# Complete catalog (op_class, expected_effect, expected_buffer_operand_names)
# ---------------------------------------------------------------------------
CATALOG = [
    # --- Control flow ---
    (Loop, Effect.NONE, []),
    (IfElse, Effect.NONE, []),
    (Break, Effect.NONE, []),
    (Continue, Effect.NONE, []),
    (GridSync, Effect.NONE, []),
    # --- Data movement ---
    (Copy, Effect.READWRITE, ["src", "dst"]),
    (TmaCopy, Effect.READWRITE, ["src", "dst"]),
    (Load, Effect.READ, ["src"]),
    (Store, Effect.WRITE, ["dst"]),
    (Fill, Effect.WRITE, ["dst"]),
    (PartitionView, Effect.READ, ["src"]),
    # --- Compute ---
    (Gemm, Effect.READWRITE, ["lhs", "rhs", "acc"]),
    (Tcgen05Gemm, Effect.READWRITE, ["lhs", "rhs", "acc"]),
    (Reduce, Effect.READWRITE, ["src", "dst"]),
    (Cumsum, Effect.READWRITE, ["src", "dst"]),
    (ThreadAllreduce, Effect.NONE, []),
    (Elementwise, Effect.NONE, []),
    (Cast, Effect.NONE, []),
    (Select, Effect.NONE, []),
    (RepeatInterleave, Effect.NONE, []),
    # --- Atomic ---
    (AtomicRMW, Effect.READWRITE, ["dst"]),
    (AtomicCAS, Effect.READWRITE, ["dst"]),
    # --- Misc ---
    (Barrier, Effect.NONE, []),
    (DeviceAssert, Effect.NONE, []),
    (DebugPrint, Effect.NONE, []),
    (DecodeI4, Effect.READWRITE, ["src", "dst"]),
    (DecodeI2, Effect.READWRITE, ["src", "dst"]),
    (Dp4a, Effect.READWRITE, ["lhs", "rhs", "acc"]),
]

CATALOG_COUNT = len(CATALOG)
_EXPECTED_COUNT = 28  # update this comment if catalog grows


# ---------------------------------------------------------------------------
# Helper: gather names for pytest parametrize
# ---------------------------------------------------------------------------
def _ids():
    return [cls.__name__ for cls, _, _ in CATALOG]


# ---------------------------------------------------------------------------
# 1. Unique opcode per class
# ---------------------------------------------------------------------------


def test_all_opcodes_unique():
    """Every op must have a distinct _opcode string."""
    opcodes = [cls._opcode for cls, _, _ in CATALOG]
    assert len(opcodes) == len(set(opcodes)), "Duplicate opcodes detected: " + str([op for op in set(opcodes) if opcodes.count(op) > 1])


# ---------------------------------------------------------------------------
# 2. Catalog completeness
# ---------------------------------------------------------------------------


def test_catalog_count():
    """Exact number of ops in the catalog matches expectation."""
    assert CATALOG_COUNT == _EXPECTED_COUNT, (
        f"Expected {_EXPECTED_COUNT} ops, got {CATALOG_COUNT}. Update CATALOG and _EXPECTED_COUNT if you added/removed ops."
    )


# ---------------------------------------------------------------------------
# 3. Per-op parametrized checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_class,expected_effect,expected_bufops", CATALOG, ids=_ids())
def test_catalog_metadata(op_class, expected_effect, expected_bufops):
    assert isinstance(op_class._opcode, str), f"{op_class.__name__}._opcode must be str"
    assert op_class._opcode, f"{op_class.__name__}._opcode must not be empty"
    assert op_class.memory_effect == expected_effect, (
        f"{op_class.__name__}.memory_effect: expected {expected_effect}, got {op_class.memory_effect}"
    )
    assert op_class._buffer_operand_names == expected_bufops, (
        f"{op_class.__name__}._buffer_operand_names: expected {expected_bufops}, got {op_class._buffer_operand_names}"
    )


# ---------------------------------------------------------------------------
# 4. Specific structural checks (fields, attributes, nested blocks)
# ---------------------------------------------------------------------------


def test_loop_has_body_block():
    assert "body" in Loop._block_names, "Loop must have a 'body' nested_block"


def test_loop_has_start_stop_step():
    for field in ("start", "stop", "step"):
        assert field in Loop._operand_names, f"Loop must have '{field}' operand"


def test_loop_is_for_property():
    """Loop.is_for must be a property (not a dataclass field)."""
    assert isinstance(Loop.__dict__.get("is_for"), property), "Loop.is_for must be a @property"


def test_ifelse_has_cond_operand():
    assert "cond" in IfElse._operand_names, "IfElse must have 'cond' operand"


def test_ifelse_has_then_else_blocks():
    assert "then_block" in IfElse._block_names, "IfElse must have 'then_block'"
    assert "else_block" in IfElse._block_names, "IfElse must have 'else_block'"


def test_elementwise_has_fn_attribute():
    assert "fn" in Elementwise._attr_names, "Elementwise must have 'fn' attribute"


def test_atomicrmw_has_kind_attribute():
    assert "kind" in AtomicRMW._attr_names, "AtomicRMW must have 'kind' attribute"


def test_gemm_has_trans_attributes():
    assert "trans_a" in Gemm._attr_names, "Gemm must have 'trans_a' attribute"
    assert "trans_b" in Gemm._attr_names, "Gemm must have 'trans_b' attribute"


def test_reduce_has_op_and_axis_attributes():
    assert "op" in Reduce._attr_names, "Reduce must have 'op' attribute"
    assert "axis" in Reduce._attr_names, "Reduce must have 'axis' attribute"


def _minimal_kwargs(op_class) -> dict:
    """Build the minimum constructor kwargs to instantiate *op_class*.

    Required marker fields (operands, buffer operands, nested blocks, and
    attributes with no default) get ``None``; fields with a default are left
    out so their defaults apply.
    """
    import dataclasses

    kwargs: dict = {}
    for name in op_class._operand_names:
        kwargs[name] = None
    for name in op_class._buffer_operand_names:
        kwargs[name] = None
    for name in op_class._block_names:
        kwargs[name] = None
    for name in op_class._attr_names:
        for f in dataclasses.fields(op_class):
            if f.name == name and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                kwargs[name] = None
                break
    return kwargs


# ---------------------------------------------------------------------------
# 5. emit_mlir status — implemented vs. not-yet-implemented
# ---------------------------------------------------------------------------
# Ops with a real emit_mlir.  Import-order independent
# because the methods live directly on the class (not via monkey-patching).
_IMPLEMENTED_EMIT = {
    # Data movement
    Load,
    Store,
    Copy,
    TmaCopy,
    Fill,
    PartitionView,
    # Control flow
    Loop,
    IfElse,
    Break,
    Continue,
    GridSync,
    # Gemm
    Gemm,
    Tcgen05Gemm,
    # Reduce / cumsum (real emit) + thread_allreduce (raises _UnsupportedTileIRNode)
    Reduce,
    Cumsum,
    ThreadAllreduce,
    # Elementwise / cast / select
    Elementwise,
    Cast,
    Select,
    RepeatInterleave,
    # Atomic rmw (unified) + cas (implemented via ct.atomic_cas_tko)
    AtomicRMW,
    AtomicCAS,
    # Misc (barrier no-op, device assert, debug print, decoders, dp4a)
    Barrier,
    DeviceAssert,
    DebugPrint,
    DecodeI4,
    DecodeI2,
    Dp4a,
}

_NOT_YET_EMIT = [(cls, eff, buf) for cls, eff, buf in CATALOG if cls not in _IMPLEMENTED_EMIT]
_NOT_YET_IDS = [cls.__name__ for cls, _, _ in _NOT_YET_EMIT]

_IMPLEMENTED_EMIT_LIST = [(cls, eff, buf) for cls, eff, buf in CATALOG if cls in _IMPLEMENTED_EMIT]
_IMPLEMENTED_IDS = [cls.__name__ for cls, _, _ in _IMPLEMENTED_EMIT_LIST]


@pytest.mark.parametrize("op_class,expected_effect,expected_bufops", _NOT_YET_EMIT, ids=_NOT_YET_IDS)
def test_emit_mlir_raises_not_implemented(op_class, expected_effect, expected_bufops):
    """Ops that have not yet implemented emit_mlir must raise NotImplementedError."""
    op = op_class(**_minimal_kwargs(op_class))
    with pytest.raises(NotImplementedError):
        op.emit_mlir(ctx=None)


@pytest.mark.parametrize("op_class,expected_effect,expected_bufops", _IMPLEMENTED_EMIT_LIST, ids=_IMPLEMENTED_IDS)
def test_emit_mlir_is_overridden(op_class, expected_effect, expected_bufops):
    """Ops in _IMPLEMENTED_EMIT must define emit_mlir directly on the class (not inherited)."""
    assert "emit_mlir" in op_class.__dict__, (
        f"{op_class.__name__} is listed in _IMPLEMENTED_EMIT but does not define "
        "emit_mlir as a real class method (it would fall back to the base stub)."
    )

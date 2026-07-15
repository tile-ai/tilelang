"""Tests for EmitContext + module/entry scaffold.

Test structure
--------------
1. test_emit_module_empty_block  — entry function with one param, no ops;
   verifies str(module) is non-empty and contains the kernel symbol.
2. test_emit_module_entry_arg_bound  — same setup; verifies the entry-arg
   Value is mapped in the EmitContext value_map after emit_module returns.
3. test_emit_module_body_walk  — adds a trivial no-result op (overrides
   emit_mlir to do nothing); verifies the walk calls emit_mlir exactly once.
4. test_emit_module_result_binding  — adds a trivial result-producing op;
   verifies the walk binds the returned mlir value to op.results[0].
5. test_emit_context_lookup_missing  — verifies lookup() raises KeyError for
   an unbound Value.
6. test_emit_context_bind_and_lookup  — verifies bind/lookup round-trip.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import pytest
from tilelang.tileir.checks import has_cuda_tile_ir_bindings

from tilelang.tileir.ir.types import TileType, MemSpace, dtype
from tilelang.tileir.ir.value import Block, Value
from tilelang.tileir.ir.ops import TileOp, Effect, attribute


# ---------------------------------------------------------------------------
# Skip guard — needs cuda_tile MLIR bindings
# ---------------------------------------------------------------------------

_HAS_CUDA_TILE = has_cuda_tile_ir_bindings()

skip_no_cuda_tile = pytest.mark.skipif(
    not _HAS_CUDA_TILE,
    reason="cuda_tile MLIR bindings unavailable",
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _f32_scalar_type() -> TileType:
    return TileType(dtype=dtype("float32"), shape=(), space=MemSpace.REGISTER, layout=None)


def _make_simple_root_block() -> tuple[Block, Value]:
    """Return (root_block, param_value) — one scalar f32 param, no ops."""
    ty = _f32_scalar_type()
    param = Value(0, ty, name="x")
    block = Block(params=[param])
    return block, param


# ---------------------------------------------------------------------------
# Minimal trivial ops used only in these tests
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class _NoResultOp(TileOp, opcode="_test_no_result", effect=Effect.NONE):
    """Trivial op: emit_mlir is overridden to record the call and return None."""

    _call_log: list = attribute(default=dataclasses.field(default_factory=list))

    def emit_mlir(self, ctx: Any) -> None:
        self._call_log.append("called")
        return None  # no results


@dataclass(eq=False)
class _ResultOp(TileOp, opcode="_test_result_op", effect=Effect.NONE):
    """Trivial op: emit_mlir returns a sentinel mlir value object."""

    _sentinel: Any = attribute(default=None)

    def emit_mlir(self, ctx: Any) -> Any:
        return self._sentinel


@dataclass(eq=False)
class _SeqResultOp(TileOp, opcode="_test_seq_result_op", effect=Effect.NONE):
    """Trivial op: emit_mlir returns a tuple/list of mlir values."""

    _seq: Any = attribute(default=None)

    def emit_mlir(self, ctx: Any) -> Any:
        return self._seq


# ---------------------------------------------------------------------------
# EmitContext unit tests (no MLIR context needed)
# ---------------------------------------------------------------------------


def test_emit_context_lookup_missing():
    """lookup() raises KeyError for a value not yet in the map."""
    from tilelang.tileir.lowering.mlir_emit import EmitContext

    ctx = EmitContext.__new__(EmitContext)
    ctx.value_map = {}
    ctx.ct = None
    ctx.ct_gen = None
    ctx.ir = None
    ctx.loc = None
    ctx._root_token = None

    v = Value(99, _f32_scalar_type())
    with pytest.raises(KeyError):
        ctx.lookup(v)


def test_emit_context_bind_and_lookup():
    """bind() + lookup() round-trip returns the stored mlir value."""
    from tilelang.tileir.lowering.mlir_emit import EmitContext

    ctx = EmitContext.__new__(EmitContext)
    ctx.value_map = {}
    ctx.ct = None
    ctx.ct_gen = None
    ctx.ir = None
    ctx.loc = None
    ctx._root_token = None

    v = Value(1, _f32_scalar_type())
    sentinel = object()
    ctx.bind(v, sentinel)
    assert ctx.lookup(v) is sentinel


def test_emit_context_token_for_returns_root():
    """token_for() returns the root_token for any op."""
    from tilelang.tileir.lowering.mlir_emit import EmitContext

    ctx = EmitContext.__new__(EmitContext)
    ctx.value_map = {}
    ctx.ct = None
    ctx.ct_gen = None
    ctx.ir = None
    ctx.loc = None
    root_token = object()
    ctx._root_token = root_token

    dummy_op = _NoResultOp(_call_log=[])
    assert ctx.token_for(dummy_op) is root_token


# ---------------------------------------------------------------------------
# emit_module integration tests (require MLIR)
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_emit_module_empty_block():
    """emit_module returns a non-empty module containing the kernel symbol."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, _param = _make_simple_root_block()
    ty = _f32_scalar_type()
    module = emit_module(root, kernel_name="my_kernel", entry_args=[("x", ty)])

    text = str(module)
    assert text, "module str() should be non-empty"
    assert "my_kernel" in text, f"kernel symbol not found in:\n{text}"


@skip_no_cuda_tile
def test_emit_module_entry_arg_bound():
    """The entry-arg Value (root.params[0]) is bound in the context after emission."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, param = _make_simple_root_block()
    ty = _f32_scalar_type()
    # emit_module returns (module, ctx) so we can inspect the value map
    module, ctx = emit_module(root, kernel_name="argbind_kernel", entry_args=[("x", ty)], return_ctx=True)

    # The param Value should be in the context's value_map
    assert param in ctx.value_map, "entry-arg Value not bound in EmitContext"
    # The bound value should be an MLIR BlockArgument
    mlir_val = ctx.lookup(param)
    assert mlir_val is not None


@skip_no_cuda_tile
def test_emit_module_body_walk_no_result():
    """emit_module walks root.ops and calls emit_mlir once per op."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    call_log: list[str] = []
    root, _ = _make_simple_root_block()
    op = _NoResultOp(_call_log=call_log)
    op.results = ()
    root.append(op)

    ty = _f32_scalar_type()
    emit_module(root, kernel_name="walk_kernel", entry_args=[("x", ty)])

    assert call_log == ["called"], f"emit_mlir not called once; log={call_log}"


@skip_no_cuda_tile
def test_emit_module_result_binding():
    """emit_module binds the mlir value returned by emit_mlir to op.results[0]."""
    from tilelang.tileir.lowering.mlir_emit import emit_module
    from tilelang.tileir.ir.value import Value

    sentinel_mlir = object()  # stand-in for a real mlir value
    root, _ = _make_simple_root_block()
    ty = _f32_scalar_type()
    result_val = Value(42, ty)
    op = _ResultOp(_sentinel=sentinel_mlir)
    op.results = (result_val,)
    root.append(op)

    _, ctx = emit_module(root, kernel_name="result_kernel", entry_args=[("x", ty)], return_ctx=True)

    assert result_val in ctx.value_map, "result Value not bound after body walk"
    assert ctx.lookup(result_val) is sentinel_mlir


@skip_no_cuda_tile
def test_emit_module_returned_module_is_usable():
    """The returned ModuleOp must remain printable after emit_module returns.

    The module keeps its owning ``ir.Context`` alive after emission.
    """
    import gc

    from tilelang.tileir.lowering.mlir_emit import emit_module

    root, _ = _make_simple_root_block()
    ty = _f32_scalar_type()
    module = emit_module(root, kernel_name="usable_kernel", entry_args=[("x", ty)])
    gc.collect()  # force any premature context teardown to surface

    text = str(module)
    assert "usable_kernel" in text, f"module unusable after return:\n{text}"


@skip_no_cuda_tile
def test_emit_module_sequence_result_length_mismatch_raises():
    """A sequence return whose length != len(op.results) raises ValueError."""
    from tilelang.tileir.lowering.mlir_emit import emit_module
    from tilelang.tileir.ir.value import Value

    root, _ = _make_simple_root_block()
    ty = _f32_scalar_type()
    r0 = Value(50, ty)
    r1 = Value(51, ty)
    # emit_mlir returns ONE value but the op declares TWO results.
    op = _SeqResultOp(_seq=[object()])
    op.results = (r0, r1)
    root.append(op)

    with pytest.raises(ValueError, match="must match 1-to-1"):
        emit_module(root, kernel_name="mismatch_kernel", entry_args=[("x", ty)])


@skip_no_cuda_tile
def test_emit_module_sequence_result_binding():
    """A matched-length sequence return binds each result element-wise."""
    from tilelang.tileir.lowering.mlir_emit import emit_module
    from tilelang.tileir.ir.value import Value

    root, _ = _make_simple_root_block()
    ty = _f32_scalar_type()
    r0 = Value(60, ty)
    r1 = Value(61, ty)
    s0, s1 = object(), object()
    op = _SeqResultOp(_seq=[s0, s1])
    op.results = (r0, r1)
    root.append(op)

    _, ctx = emit_module(root, kernel_name="seq_kernel", entry_args=[("x", ty)], return_ctx=True)

    assert ctx.lookup(r0) is s0
    assert ctx.lookup(r1) is s1

"""Tests for Elementwise / Cast / Select emit_mlir.

Coverage
--------
1. test_elementwise_unary_exp — unary exp emits 'exp'
2. test_elementwise_binary_add — binary add emits 'add'
3. test_elementwise_binary_mul — binary mul emits 'mul'
4. test_elementwise_pow — pow emits 'pow'
5. test_elementwise_pow_fma — FMA pattern (mul+add with fast_math=True) emits 'fma'
6. test_cast_f32_to_f16 — cast float32→float16 emits a cast/ftof mnemonic
7. test_select_emits_select — select emits 'select'
8. test_elementwise_result_bound — result Value is bound in EmitContext after emission
9. test_cast_result_bound — Cast result Value is bound after emission
10. test_select_result_bound — Select result Value is bound after emission
"""

from __future__ import annotations

from typing import Any

import pytest
from tilelang.tileir.checks import has_cuda_tile_ir_bindings
from tilelang.tileir.errors import TileIRLoweringError

from tilelang.tileir.ir.types import TileType, MemSpace, dtype
from tilelang.tileir.ir.value import Block, Value
from tilelang.tileir.ir.ops import Elementwise, Cast, Select

# ---------------------------------------------------------------------------
# Skip guard — needs cuda_tile MLIR bindings
# ---------------------------------------------------------------------------

_HAS_CUDA_TILE = has_cuda_tile_ir_bindings()

skip_no_cuda_tile = pytest.mark.skipif(
    not _HAS_CUDA_TILE,
    reason="cuda_tile MLIR bindings unavailable",
)

# ---------------------------------------------------------------------------
# Shared dtype/type helpers
# ---------------------------------------------------------------------------

FP32 = dtype("float32")
FP16 = dtype("float16")
I32 = dtype("int32")
BOOL = dtype("bool")

SHAPE = (64, 64)


def _reg_fp32(shape=SHAPE) -> TileType:
    return TileType(dtype=FP32, shape=tuple(shape), space=MemSpace.REGISTER, layout=None)


def _reg_fp16(shape=SHAPE) -> TileType:
    return TileType(dtype=FP16, shape=tuple(shape), space=MemSpace.REGISTER, layout=None)


def _reg_bool(shape=SHAPE) -> TileType:
    return TileType(dtype=BOOL, shape=tuple(shape), space=MemSpace.REGISTER, layout=None)


# ---------------------------------------------------------------------------
# Helper: simple counter for fresh_value
# ---------------------------------------------------------------------------


class _Ctr:
    def __init__(self):
        self._n = 0

    def next(self):
        v = self._n
        self._n += 1
        return v


# ---------------------------------------------------------------------------
# Helper: emit a root block with register tile params and record the module
# ---------------------------------------------------------------------------


def _emit_with_ops(params: list[Value], ops: list[Any]) -> tuple[Any, Any]:
    """Build a root block, add ops, emit, return (module_text, ctx)."""
    from tilelang.tileir.lowering.mlir_emit import emit_module

    root = Block(params=params)
    for op in ops:
        root.append(op)

    entry_args = [(p.name or f"p{i}", p.type) for i, p in enumerate(root.params)]
    module, ctx = emit_module(
        root,
        kernel_name="test_kernel",
        entry_args=entry_args,
        return_ctx=True,
    )
    return str(module), ctx


# ===========================================================================
# 1. Elementwise unary — exp
# ===========================================================================


@skip_no_cuda_tile
def test_elementwise_unary_exp():
    """Elementwise(fn='exp', inputs=(x,)) emits an 'exp' op mnemonic."""
    ctr = _Ctr()
    x_ty = _reg_fp32()
    x_val = Value(ctr.next(), x_ty, name="x")
    out_ty = _reg_fp32()
    out_val = Value(ctr.next(), out_ty, name="out")

    op = Elementwise(fn="exp", inputs=(x_val,))
    op.results = (out_val,)

    text, ctx = _emit_with_ops([x_val], [op])
    assert "exp" in text, f"Expected 'exp' mnemonic in:\n{text}"


# ===========================================================================
# 2. Elementwise binary — add
# ===========================================================================


@skip_no_cuda_tile
def test_elementwise_binary_add():
    """Elementwise(fn='add', inputs=(lhs, rhs)) emits an 'add' op mnemonic."""
    ctr = _Ctr()
    a_ty = _reg_fp32()
    b_ty = _reg_fp32()
    out_ty = _reg_fp32()
    a_val = Value(ctr.next(), a_ty, name="a")
    b_val = Value(ctr.next(), b_ty, name="b")
    out_val = Value(ctr.next(), out_ty, name="out")

    op = Elementwise(fn="add", inputs=(a_val, b_val))
    op.results = (out_val,)

    text, ctx = _emit_with_ops([a_val, b_val], [op])
    assert "add" in text, f"Expected 'add' mnemonic in:\n{text}"


# ===========================================================================
# 3. Elementwise binary — mul
# ===========================================================================


@skip_no_cuda_tile
def test_elementwise_binary_mul():
    """Elementwise(fn='mul', inputs=(lhs, rhs)) emits a 'mul' op mnemonic."""
    ctr = _Ctr()
    a_ty = _reg_fp32()
    b_ty = _reg_fp32()
    out_ty = _reg_fp32()
    a_val = Value(ctr.next(), a_ty, name="a")
    b_val = Value(ctr.next(), b_ty, name="b")
    out_val = Value(ctr.next(), out_ty, name="out")

    op = Elementwise(fn="mul", inputs=(a_val, b_val))
    op.results = (out_val,)

    text, ctx = _emit_with_ops([a_val, b_val], [op])
    assert "mul" in text, f"Expected 'mul' mnemonic in:\n{text}"


@skip_no_cuda_tile
def test_elementwise_rejects_ambiguous_same_rank_broadcast():
    """Different non-unit extents need semantic axis metadata, not guessing."""
    ctr = _Ctr()
    a_val = Value(ctr.next(), _reg_fp32((4,)), name="a")
    b_val = Value(ctr.next(), _reg_fp32((8,)), name="b")
    out_val = Value(ctr.next(), _reg_fp32((8,)), name="out")
    op = Elementwise(fn="mul", inputs=(a_val, b_val))
    op.results = (out_val,)

    with pytest.raises(TileIRLoweringError, match="cannot broadcast tile shape.*4.*8"):
        _emit_with_ops([a_val, b_val], [op])


# ===========================================================================
# 4. Elementwise pow
# ===========================================================================


@skip_no_cuda_tile
def test_elementwise_pow():
    """Elementwise(fn='pow', inputs=(base, exp)) emits a 'pow' op mnemonic."""
    ctr = _Ctr()
    a_ty = _reg_fp32()
    b_ty = _reg_fp32()
    out_ty = _reg_fp32()
    a_val = Value(ctr.next(), a_ty, name="base")
    b_val = Value(ctr.next(), b_ty, name="exp")
    out_val = Value(ctr.next(), out_ty, name="out")

    op = Elementwise(fn="pow", inputs=(a_val, b_val))
    op.results = (out_val,)

    text, ctx = _emit_with_ops([a_val, b_val], [op])
    assert "pow" in text, f"Expected 'pow' mnemonic in:\n{text}"


# ===========================================================================
# 5. FMA: mul + add with fast_math=True should emit fma
# ===========================================================================


@skip_no_cuda_tile
def test_elementwise_fma():
    """Elementwise(fn='fma', inputs=(a, b, c)) emits 'fma' op mnemonic."""
    ctr = _Ctr()
    a_val = Value(ctr.next(), _reg_fp32(), name="a")
    b_val = Value(ctr.next(), _reg_fp32(), name="b")
    c_val = Value(ctr.next(), _reg_fp32(), name="c")
    out_val = Value(ctr.next(), _reg_fp32(), name="out")

    op = Elementwise(fn="fma", inputs=(a_val, b_val, c_val))
    op.results = (out_val,)

    text, ctx = _emit_with_ops([a_val, b_val, c_val], [op])
    assert "fma" in text, f"Expected 'fma' mnemonic in:\n{text}"


# ===========================================================================
# 6. Cast: float32 → float16
# ===========================================================================


@skip_no_cuda_tile
def test_cast_f32_to_f16():
    """Cast(src, dtype='float16') emits a float-to-float cast mnemonic."""
    ctr = _Ctr()
    src_val = Value(ctr.next(), _reg_fp32(), name="src")
    out_val = Value(ctr.next(), _reg_fp16(), name="out")

    op = Cast(src=src_val, dtype="float16")
    op.results = (out_val,)

    text, ctx = _emit_with_ops([src_val], [op])
    # ftof is the cuTile float-to-float cast; or 'cast' may appear
    assert "ftof" in text or "cast" in text, f"Expected float-to-float cast mnemonic in:\n{text}"


# ===========================================================================
# 7. Select
# ===========================================================================


@skip_no_cuda_tile
def test_select_emits_select():
    """Select(cond, true_val, false_val) emits a 'select' op mnemonic."""
    ctr = _Ctr()
    cond_val = Value(ctr.next(), _reg_bool(), name="cond")
    tv_val = Value(ctr.next(), _reg_fp32(), name="tv")
    fv_val = Value(ctr.next(), _reg_fp32(), name="fv")
    out_val = Value(ctr.next(), _reg_fp32(), name="out")

    op = Select(cond=cond_val, true_val=tv_val, false_val=fv_val)
    op.results = (out_val,)

    text, ctx = _emit_with_ops([cond_val, tv_val, fv_val], [op])
    assert "select" in text, f"Expected 'select' mnemonic in:\n{text}"


# ===========================================================================
# 8–10. Result bindings
# ===========================================================================


@skip_no_cuda_tile
def test_elementwise_result_bound():
    """The Elementwise result Value is bound in EmitContext after emission."""
    ctr = _Ctr()
    x_val = Value(ctr.next(), _reg_fp32(), name="x")
    out_val = Value(ctr.next(), _reg_fp32(), name="out")

    op = Elementwise(fn="exp", inputs=(x_val,))
    op.results = (out_val,)

    _text, ctx = _emit_with_ops([x_val], [op])
    assert out_val in ctx.value_map, f"Expected out_val (id={out_val.id}) in ctx.value_map after Elementwise emission."


@skip_no_cuda_tile
def test_cast_result_bound():
    """The Cast result Value is bound in EmitContext after emission."""
    ctr = _Ctr()
    src_val = Value(ctr.next(), _reg_fp32(), name="src")
    out_val = Value(ctr.next(), _reg_fp16(), name="out")

    op = Cast(src=src_val, dtype="float16")
    op.results = (out_val,)

    _text, ctx = _emit_with_ops([src_val], [op])
    assert out_val in ctx.value_map, f"Expected out_val (id={out_val.id}) in ctx.value_map after Cast emission."


@skip_no_cuda_tile
def test_select_result_bound():
    """The Select result Value is bound in EmitContext after emission."""
    ctr = _Ctr()
    cond_val = Value(ctr.next(), _reg_bool(), name="cond")
    tv_val = Value(ctr.next(), _reg_fp32(), name="tv")
    fv_val = Value(ctr.next(), _reg_fp32(), name="fv")
    out_val = Value(ctr.next(), _reg_fp32(), name="out")

    op = Select(cond=cond_val, true_val=tv_val, false_val=fv_val)
    op.results = (out_val,)

    _text, ctx = _emit_with_ops([cond_val, tv_val, fv_val], [op])
    assert out_val in ctx.value_map, f"Expected out_val (id={out_val.id}) in ctx.value_map after Select emission."

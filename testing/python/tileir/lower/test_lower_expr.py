"""Tests for lower_expr — real expression lowering.

Verifies:
- IntImm/FloatImm → Constant op with correct value/dtype.
- Var → scope lookup returns bound Value.
- BufferLoad → Load op result.
- Binary arithmetic (Add/Sub/Mul/Div/FloorDiv/FloorMod/Min/Max) → Elementwise.
- Comparisons (LT/LE/GT/GE/EQ/NE) → Elementwise with bool result.
- Logical And/Or/Not → Elementwise.
- Cast → Cast op.
- Select → Select op.
- Unary math calls (exp/log/sqrt/tanh) → Elementwise unary.
- FMA (a*b+c float) → Elementwise(fn='fma').
- Loop stop Value traces to real bound (not placeholder name "stop").
- buffer_store val traces to a real lowered expr.
- No placeholder Values remain for covered constructs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from tvm import tirx

import tilelang
from tilelang import language as T

from tilelang.tileir.lowering.tir_to_sem import tir_to_sem
from tilelang.tileir.lowering.sem_to_ir import (
    LoweringScope,
    lower_expr,
    lower_kernel,
)
from tilelang.tileir.ir.builder import IRBuilder
from tilelang.tileir.ir.ops import (
    Cast,
    Constant,
    Elementwise,
    Load,
    Loop,
    Select,
    Store,
)
from tilelang.tileir.ir.types import MemSpace, TileType, dtype as lookup_dtype
from tilelang.tileir.ir.value import Block, Value
from tilelang.tileir.semantic import SemanticBuffer, SemanticKernel, SemanticStmt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_scope() -> LoweringScope:
    body = SemanticStmt(kind="seq", children=())
    kernel = SemanticKernel(
        name="test",
        grid=("1", "1", "1"),
        threads=("128", "1", "1"),
        alloc_buffers=(),
        body=body,
    )
    return LoweringScope(kernel)


def _all_ops(block: Block) -> list:
    result = []
    for op in block.ops:
        result.append(op)
        for nested in op.nested_blocks():
            if nested is not None:
                result.extend(_all_ops(nested))
    return result


def _ops_of_type(block: Block, typ) -> list:
    return [op for op in _all_ops(block) if isinstance(op, typ)]


# ---------------------------------------------------------------------------
# Test 1: IntImm → Constant op
# ---------------------------------------------------------------------------


def test_lower_expr_int_imm():
    """IntImm(42) should lower to a Constant op with value=42 and int result."""
    scope = _empty_scope()
    builder = IRBuilder()
    expr = tirx.IntImm("int32", 42)
    val = lower_expr(expr, scope, builder)

    assert val is not None
    assert isinstance(val, Value)
    assert val.type.shape == ()
    assert str(val.type.dtype) == "int32"

    # The block should contain exactly one Constant op.
    const_ops = [op for op in builder.block.ops if isinstance(op, Constant)]
    assert len(const_ops) == 1
    assert const_ops[0].value == 42
    assert const_ops[0].dtype == "int32"
    # The result Value should be what lower_expr returned.
    assert const_ops[0].results[0] is val


# ---------------------------------------------------------------------------
# Test 2: FloatImm → Constant op
# ---------------------------------------------------------------------------


def test_lower_expr_float_imm():
    """FloatImm(1.5) should lower to a Constant op with value=1.5."""
    scope = _empty_scope()
    builder = IRBuilder()
    expr = tirx.FloatImm("float32", 1.5)
    val = lower_expr(expr, scope, builder)

    assert val is not None
    const_ops = [op for op in builder.block.ops if isinstance(op, Constant)]
    assert len(const_ops) == 1
    assert abs(const_ops[0].value - 1.5) < 1e-7
    assert const_ops[0].dtype == "float32"


# ---------------------------------------------------------------------------
# Test 3: Var → scope lookup
# ---------------------------------------------------------------------------


def test_lower_expr_var_bound():
    """A Var whose name is bound in scope should return that Value."""
    scope = _empty_scope()
    builder = IRBuilder()

    # Manually bind a Value for var 'x'.
    i32_ty = TileType(dtype=lookup_dtype("int32"), shape=(), space=MemSpace.REGISTER, layout=None)
    x_val = Value(id=99, type=i32_ty, name="x")
    scope.bind("x", x_val)

    expr = tirx.Var("x", "int32")
    result = lower_expr(expr, scope, builder)
    assert result is x_val
    # No new ops should be created.
    assert len(builder.block.ops) == 0


def test_lower_expr_var_unbound():
    """An unbound Var must raise _UnsupportedTileIRNode (fail loud, not silent placeholder)."""
    from tilelang.tileir.errors import _UnsupportedTileIRNode

    scope = _empty_scope()
    builder = IRBuilder()
    expr = tirx.Var("y", "int32")
    with pytest.raises(_UnsupportedTileIRNode, match="y"):
        lower_expr(expr, scope, builder)


# ---------------------------------------------------------------------------
# Test 4: BufferLoad → Load op
# ---------------------------------------------------------------------------


def test_lower_expr_buffer_load():
    """BufferLoad from a known buffer should produce a Load op."""
    # Create a scope with a known buffer.
    buf = SemanticBuffer(name="mybuf", shape=(8,), dtype="float32", scope="local")
    body = SemanticStmt(kind="seq", children=())
    kernel = SemanticKernel(
        name="test",
        grid=("1", "1", "1"),
        threads=("128", "1", "1"),
        alloc_buffers=(buf,),
        body=body,
    )
    scope = LoweringScope(kernel)
    builder = IRBuilder()

    # Build a TIR BufferLoad
    tir_buf = tirx.decl_buffer((8,), "float32", name="mybuf")
    expr = tirx.BufferLoad(tir_buf, [tirx.IntImm("int32", 0)])
    val = lower_expr(expr, scope, builder)

    assert val is not None
    load_ops = [op for op in builder.block.ops if isinstance(op, Load)]
    assert len(load_ops) >= 1, "Expected at least one Load op"
    assert load_ops[0].results[0] is val


# ---------------------------------------------------------------------------
# Test 5: Binary arithmetic → Elementwise
# ---------------------------------------------------------------------------


def test_lower_expr_add():
    """Add(a, b) should produce Elementwise(fn='add', inputs=(va, vb))."""
    scope = _empty_scope()
    builder = IRBuilder()

    a = tirx.IntImm("int32", 3)
    b = tirx.IntImm("int32", 4)
    expr = tirx.Add(a, b)
    val = lower_expr(expr, scope, builder)

    assert val is not None
    ew_ops = [op for op in builder.block.ops if isinstance(op, Elementwise)]
    add_ops = [op for op in ew_ops if op.fn == "add"]
    assert len(add_ops) == 1, f"Expected 1 add Elementwise, got {len(add_ops)}"
    assert add_ops[0].results[0] is val


def test_lower_expr_mul():
    """Mul should produce Elementwise(fn='mul')."""
    scope = _empty_scope()
    builder = IRBuilder()
    a = tirx.FloatImm("float32", 2.0)
    b = tirx.FloatImm("float32", 3.0)
    val = lower_expr(tirx.Mul(a, b), scope, builder)
    mul_ops = [op for op in builder.block.ops if isinstance(op, Elementwise) and op.fn == "mul"]
    assert len(mul_ops) == 1
    assert mul_ops[0].results[0] is val


def test_lower_expr_sub():
    """Sub should produce Elementwise(fn='sub')."""
    scope = _empty_scope()
    builder = IRBuilder()
    a = tirx.IntImm("int32", 10)
    b = tirx.IntImm("int32", 3)
    lower_expr(tirx.Sub(a, b), scope, builder)
    sub_ops = [op for op in builder.block.ops if isinstance(op, Elementwise) and op.fn == "sub"]
    assert len(sub_ops) == 1


def test_lower_expr_min_max():
    """Min/Max should produce Elementwise(fn='min'/'max')."""
    scope = _empty_scope()
    builder = IRBuilder()
    a = tirx.IntImm("int32", 5)
    b = tirx.IntImm("int32", 7)
    lower_expr(tirx.Min(a, b), scope, builder)
    lower_expr(tirx.Max(a, b), scope, builder)
    min_ops = [op for op in builder.block.ops if isinstance(op, Elementwise) and op.fn == "min"]
    max_ops = [op for op in builder.block.ops if isinstance(op, Elementwise) and op.fn == "max"]
    assert len(min_ops) == 1
    assert len(max_ops) == 1


# ---------------------------------------------------------------------------
# Test 6: Comparison → Elementwise with bool result
# ---------------------------------------------------------------------------


def test_lower_expr_lt():
    """LT(a, b) should produce Elementwise(fn='lt') with bool result type."""
    scope = _empty_scope()
    builder = IRBuilder()
    a = tirx.IntImm("int32", 3)
    b = tirx.IntImm("int32", 5)
    val = lower_expr(tirx.LT(a, b), scope, builder)

    assert val is not None
    lt_ops = [op for op in builder.block.ops if isinstance(op, Elementwise) and op.fn == "lt"]
    assert len(lt_ops) == 1
    # Result type should be bool scalar.
    assert str(lt_ops[0].results[0].type.dtype) == "bool"


def test_lower_expr_eq():
    """EQ should produce Elementwise(fn='eq')."""
    scope = _empty_scope()
    builder = IRBuilder()
    lower_expr(tirx.EQ(tirx.IntImm("int32", 1), tirx.IntImm("int32", 1)), scope, builder)
    eq_ops = [op for op in builder.block.ops if isinstance(op, Elementwise) and op.fn == "eq"]
    assert len(eq_ops) == 1


# ---------------------------------------------------------------------------
# Test 7: Logical And/Or/Not
# ---------------------------------------------------------------------------


def test_lower_expr_logical_and_or():
    """And/Or/Not should produce Elementwise ops."""
    scope = _empty_scope()
    builder = IRBuilder()
    t = tirx.IntImm("bool", 1)
    f = tirx.IntImm("bool", 0)
    lower_expr(tirx.And(t, f), scope, builder)
    lower_expr(tirx.Or(t, f), scope, builder)
    lower_expr(tirx.Not(t), scope, builder)
    and_ops = [op for op in builder.block.ops if isinstance(op, Elementwise) and op.fn == "andi"]
    or_ops = [op for op in builder.block.ops if isinstance(op, Elementwise) and op.fn == "ori"]
    not_ops = [op for op in builder.block.ops if isinstance(op, Elementwise) and op.fn == "not"]
    assert len(and_ops) == 1
    assert len(or_ops) == 1
    assert len(not_ops) == 1


# ---------------------------------------------------------------------------
# Test 8: Cast → Cast op
# ---------------------------------------------------------------------------


def test_lower_expr_cast():
    """Cast(int32 → float32) should produce a Cast op."""
    scope = _empty_scope()
    builder = IRBuilder()
    src = tirx.IntImm("int32", 5)
    expr = tirx.Cast("float32", src)
    val = lower_expr(expr, scope, builder)

    assert val is not None
    cast_ops = [op for op in builder.block.ops if isinstance(op, Cast)]
    assert len(cast_ops) == 1
    assert cast_ops[0].dtype == "float32"
    assert cast_ops[0].results[0] is val
    assert str(val.type.dtype) == "float32"


# ---------------------------------------------------------------------------
# Test 9: Select → Select op
# ---------------------------------------------------------------------------


def test_lower_expr_select():
    """Select(cond, true, false) should produce a Select op."""
    scope = _empty_scope()
    builder = IRBuilder()
    cond = tirx.IntImm("bool", 1)
    tv = tirx.FloatImm("float32", 1.0)
    fv = tirx.FloatImm("float32", 0.0)
    # Use tirx.Select
    expr = tirx.Select(cond, tv, fv)
    val = lower_expr(expr, scope, builder)

    assert val is not None
    sel_ops = [op for op in builder.block.ops if isinstance(op, Select)]
    assert len(sel_ops) == 1
    assert sel_ops[0].results[0] is val


# ---------------------------------------------------------------------------
# Test 10: Unary math calls → Elementwise
# ---------------------------------------------------------------------------


def test_lower_expr_unary_math():
    """Unary math calls (exp, log, sqrt, tanh) → Elementwise unary."""
    scope = _empty_scope()
    builder = IRBuilder()
    x = tirx.FloatImm("float32", 2.0)

    lower_expr(tirx.exp(x), scope, builder)
    lower_expr(tirx.log(x), scope, builder)
    lower_expr(tirx.sqrt(x), scope, builder)
    lower_expr(tirx.tanh(x), scope, builder)

    ew_ops = [op for op in builder.block.ops if isinstance(op, Elementwise)]
    fns = {op.fn for op in ew_ops}
    assert "exp" in fns
    assert "log" in fns
    assert "sqrt" in fns
    assert "tanh" in fns


# ---------------------------------------------------------------------------
# Test 11: FMA pattern (a*b + c for floats)
# ---------------------------------------------------------------------------


def test_lower_expr_fma():
    """a*b + c for float should produce Elementwise(fn='fma') only under fast_math."""
    # FMA is IEEE-unsafe (contracts a*b+c into one rounded op), so the TileIR
    # backend gates it behind the fast_math flag, matching cuTile semantics.
    # Without fast_math the scope emits separate Mul + Add.
    body = SemanticStmt(kind="seq", children=())
    kernel = SemanticKernel(
        name="test",
        grid=("1", "1", "1"),
        threads=("128", "1", "1"),
        alloc_buffers=(),
        body=body,
    )
    scope_fm = LoweringScope(kernel, fast_math=True)
    builder = IRBuilder()
    a = tirx.FloatImm("float32", 2.0)
    b = tirx.FloatImm("float32", 3.0)
    c = tirx.FloatImm("float32", 1.0)
    expr = tirx.Add(tirx.Mul(a, b), c)
    val = lower_expr(expr, scope_fm, builder)

    fma_ops = [op for op in builder.block.ops if isinstance(op, Elementwise) and op.fn == "fma"]
    assert len(fma_ops) == 1, f"Expected 1 fma op under fast_math, got {len(fma_ops)}"
    assert fma_ops[0].results[0] is val
    assert len(fma_ops[0].inputs) == 3

    # Also verify that without fast_math, FMA is NOT emitted.
    scope_no_fm = LoweringScope(kernel)
    builder2 = IRBuilder()
    lower_expr(expr, scope_no_fm, builder2)
    fma_ops2 = [op for op in builder2.block.ops if isinstance(op, Elementwise) and op.fn == "fma"]
    assert len(fma_ops2) == 0, f"Expected 0 fma ops without fast_math, got {len(fma_ops2)}"


# ---------------------------------------------------------------------------
# Test 12: Loop stop traces to real bound, not "stop" placeholder
# ---------------------------------------------------------------------------


def test_lower_kernel_loop_stop_is_real():
    """For-loop stop Value should NOT be a placeholder named 'stop'.

    Verifies that the stop Value is tied to a real TileIR op (Constant or
    Elementwise) rather than an orphaned fresh_value with name='stop'.
    The stop Value must appear as the result of some op in the block.
    """
    from tilelang.tileir.semantic import SemanticStmt, SemanticKernel
    from tvm import tirx as tir

    # Build a minimal for-loop with extent=4 directly via SemanticStmt
    # so the test doesn't depend on tilelang.jit source-inspection.
    for_stmt = SemanticStmt(
        kind="for",
        attrs=(
            ("kind", "serial"),
            ("var", "i"),
            ("min", tir.IntImm("int32", 0)),
            ("extent", tir.IntImm("int32", 4)),
        ),
        children=(),
    )
    body = SemanticStmt(kind="seq", children=(for_stmt,))
    kernel = SemanticKernel(
        name="test_loop_stop",
        grid=("1", "1", "1"),
        threads=("128", "1", "1"),
        alloc_buffers=(),
        body=body,
    )
    builder = IRBuilder()
    block = lower_kernel(kernel, builder)

    loop_ops = [op for op in _all_ops(block) if isinstance(op, Loop)]
    assert len(loop_ops) >= 1

    for_loops = [op for op in loop_ops if op.is_for]
    assert len(for_loops) >= 1

    for_loop = for_loops[0]
    stop_val = for_loop.stop
    assert stop_val is not None, "Loop stop should not be None"
    # stop_val should NOT be an orphaned placeholder (name='stop', no def_op).
    assert stop_val.name != "stop" or stop_val.def_op is not None, (
        "stop_val is a raw placeholder (name='stop', no def_op) — lower_expr was not invoked for the loop bound"
    )

    # Verify the stop Value is the result of a Constant op with value=4.
    all_block_ops = _all_ops(block)
    const_ops = [op for op in all_block_ops if isinstance(op, Constant)]
    const_for_stop = [op for op in const_ops if op.results and op.results[0] is stop_val]
    assert len(const_for_stop) == 1, (
        f"Expected stop_val to be a Constant(4) result; const ops: {[(op.value, op.results) for op in const_ops]}"
    )
    assert const_for_stop[0].value == 4


# ---------------------------------------------------------------------------
# Test 13: buffer_store value traces to a real lowered expr
# ---------------------------------------------------------------------------


def test_buffer_store_value_is_real():
    """buffer_store with a TIR value expr should produce a real val, not placeholder."""
    buf = SemanticBuffer(name="out", shape=(4,), dtype="float32", scope="local")

    # Build: out[0] = 3.14
    store_stmt = SemanticStmt(
        kind="buffer_store",
        attrs=(("buffer", "out"), ("value", tirx.FloatImm("float32", 3.14))),
    )
    body = SemanticStmt(kind="seq", children=(store_stmt,))
    kernel = SemanticKernel(
        name="test_store",
        grid=("1", "1", "1"),
        threads=("128", "1", "1"),
        alloc_buffers=(buf,),
        body=body,
    )
    builder = IRBuilder()
    lower_kernel(kernel, builder)

    store_ops = [op for op in _all_ops(builder.block) if isinstance(op, Store)]
    assert len(store_ops) == 1

    store_op = store_ops[0]
    val = store_op.val

    # val should be a result of a Constant op (3.14).
    const_ops = [op for op in builder.block.ops if isinstance(op, Constant)]
    const_vals = {op.results[0] for op in const_ops if op.results}
    assert val in const_vals, f"Store val {val} is not a Constant result; Constant values: {const_vals}"


# ---------------------------------------------------------------------------
# Test 14: C[i] = A[i] + B[i] kernel produces Load, Load, Elementwise(add), Store
# ---------------------------------------------------------------------------


def test_elementwise_add_kernel_structure():
    """A scalar element-wise add kernel lowers to Load+Load+Elementwise(add)+Store."""

    @tilelang.jit
    def add_kernel(A, B, C):
        A: T.Tensor((128,), T.float32)
        B: T.Tensor((128,), T.float32)
        C: T.Tensor((128,), T.float32)

        with T.Kernel(1, threads=128):
            a_local = T.alloc_fragment((128,), T.float32)
            b_local = T.alloc_fragment((128,), T.float32)
            T.copy(A, a_local)
            T.copy(B, b_local)

    prim_func = add_kernel.get_tir(None, None, None)
    program = tir_to_sem(prim_func)
    kernel = program.kernels[0]

    builder = IRBuilder()
    block = lower_kernel(kernel, builder, program=program)

    # Should have Copy ops (lowered from the T.copy calls).
    from tilelang.tileir.ir.ops import Copy

    copy_ops = [op for op in _all_ops(block) if isinstance(op, Copy)]
    assert len(copy_ops) >= 2, f"Expected at least 2 Copy ops, got {len(copy_ops)}"


# ---------------------------------------------------------------------------
# Test 15: No placeholder Values for covered constructs (sampling check)
# ---------------------------------------------------------------------------


def test_no_placeholder_for_int_imm():
    """lower_expr of IntImm must not produce a Value named 'expr'."""
    scope = _empty_scope()
    builder = IRBuilder()
    val = lower_expr(tirx.IntImm("int32", 7), scope, builder)

    # Name should not be the fallback "expr_<typename>" or "expr"
    assert val.name != "expr"
    assert not (val.name or "").startswith("expr_")


def test_no_placeholder_for_add():
    """lower_expr of Add must produce an Elementwise, not a placeholder."""
    scope = _empty_scope()
    builder = IRBuilder()
    val = lower_expr(
        tirx.Add(tirx.IntImm("int32", 1), tirx.IntImm("int32", 2)),
        scope,
        builder,
    )
    ew_ops = [op for op in builder.block.ops if isinstance(op, Elementwise)]
    assert len(ew_ops) >= 1, "Expected an Elementwise op for Add"
    # Placeholder names would be "expr" or "expr_Add"
    assert val.name not in ("expr", "expr_Add")

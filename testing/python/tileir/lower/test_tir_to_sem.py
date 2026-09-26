"""Tests for lowering/tir_to_sem.py — TIR → SemanticIR front door.

Verifies:
- tir_to_sem returns a SemanticProgram with expected kernels/stmt kinds.
- tir_to_sem and extract_semantic_program agree.
- Unsupported TIR constructs propagate TileLangSemanticError unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow imports from the sibling test-utils directory (same pattern as other tileir tests).
sys.path.insert(0, str(Path(__file__).parents[1]))

import tilelang
from tilelang import language as T
from tilelang import tvm as tvm
from tvm import tirx

from tilelang.tileir.lowering.tir_to_sem import tir_to_sem
from tilelang.tileir.semantic import (
    SemanticProgram,
    TileLangSemanticError,
    extract_semantic_program,
)
from tileir_test_utils import _semantic_kinds, _semantic_tile_ops, _semantic_stmts

Range = tvm.ir.Range


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_copy_kernel():
    """Simple copy kernel: T.copy(a, b)."""

    @tilelang.jit
    def copy_kernel(a, b):
        a: T.Tensor((32, 64), T.float16)
        b: T.Tensor((32, 64), T.float16)

        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared((32, 64), T.float16)
            T.copy(a, a_shared)
            T.copy(a_shared, b)

    return copy_kernel.get_tir(None, None)


def _make_gemm_kernel():
    """Minimal gemm kernel: copy + gemm + copy."""

    @tilelang.jit
    def gemm_kernel(a, b, c):
        a: T.Tensor((32, 64), T.float16)
        b: T.Tensor((64, 32), T.float16)
        c: T.Tensor((32, 32), T.float32)

        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared((32, 64), T.float16)
            b_shared = T.alloc_shared((64, 32), T.float16)
            acc = T.alloc_fragment((32, 32), T.float32)
            T.copy(a, a_shared)
            T.copy(b, b_shared)
            T.gemm(a_shared, b_shared, acc, clear_accum=True)
            T.copy(acc, c)

    return gemm_kernel.get_tir(None, None, None)


# ---------------------------------------------------------------------------
# Core tests
# ---------------------------------------------------------------------------


def test_tir_to_sem_returns_semantic_program_for_copy_kernel():
    """tir_to_sem returns a SemanticProgram instance for a simple copy kernel."""
    prim_func = _make_copy_kernel()

    program = tir_to_sem(prim_func)

    assert isinstance(program, SemanticProgram)
    assert len(program.kernels) == 1
    kinds = _semantic_kinds(program.kernels[0].body)
    assert "thread_extent" in kinds
    ops = _semantic_tile_ops(program.kernels[0].body)
    assert "tl.tileop.copy" in ops


def test_tir_to_sem_returns_semantic_program_for_gemm_kernel():
    """tir_to_sem returns a SemanticProgram with gemm tile_op for a gemm kernel."""
    prim_func = _make_gemm_kernel()

    program = tir_to_sem(prim_func)

    assert isinstance(program, SemanticProgram)
    assert len(program.kernels) == 1
    ops = _semantic_tile_ops(program.kernels[0].body)
    assert {"tl.tileop.copy", "tl.tileop.gemm"}.issubset(ops)
    gemm_stmts = [
        stmt
        for stmt in _semantic_stmts(program.kernels[0].body)
        if stmt.kind == "tile_op" and dict(stmt.attrs).get("op") == "tl.tileop.gemm"
    ]
    assert len(gemm_stmts) == 1
    assert dict(gemm_stmts[0].attrs)["clear_accum"] == "1"


def test_tir_to_sem_agrees_with_extract_semantic_program_copy():
    """tir_to_sem and extract_semantic_program produce identical results for a copy kernel."""
    prim_func = _make_copy_kernel()

    program_new = tir_to_sem(prim_func)
    program_ref = extract_semantic_program(prim_func)

    assert program_new.name == program_ref.name
    assert len(program_new.kernels) == len(program_ref.kernels)
    assert program_new.kernels[0].grid == program_ref.kernels[0].grid
    assert program_new.kernels[0].threads == program_ref.kernels[0].threads
    assert _semantic_tile_ops(program_new.kernels[0].body) == _semantic_tile_ops(program_ref.kernels[0].body)
    assert _semantic_kinds(program_new.kernels[0].body) == _semantic_kinds(program_ref.kernels[0].body)


def test_tir_to_sem_agrees_with_extract_semantic_program_gemm():
    """tir_to_sem and extract_semantic_program produce identical results for a gemm kernel."""
    prim_func = _make_gemm_kernel()

    program_new = tir_to_sem(prim_func)
    program_ref = extract_semantic_program(prim_func)

    assert program_new.name == program_ref.name
    assert len(program_new.kernels) == len(program_ref.kernels)
    assert program_new.kernels[0].grid == program_ref.kernels[0].grid
    assert _semantic_tile_ops(program_new.kernels[0].body) == _semantic_tile_ops(program_ref.kernels[0].body)
    # Param names must match
    assert [b.name for b in program_new.params] == [b.name for b in program_ref.params]
    # alloc_buffers same set of scopes
    assert {b.scope for b in program_new.kernels[0].alloc_buffers} == {b.scope for b in program_ref.kernels[0].alloc_buffers}


def test_tir_to_sem_propagates_semantic_error_for_unsupported_call():
    """tir_to_sem lets TileLangSemanticError propagate for unsupported TIR nodes."""
    block_x = tirx.IterVar(
        Range(tirx.IntImm("int32", 0), tirx.IntImm("int32", 1)),
        tirx.Var("block_x", "int32"),
        tirx.IterVar.ThreadIndex,
        "blockIdx.x",
    )
    prim_func = tirx.PrimFunc(
        [],
        tirx.AttrStmt(
            block_x,
            "thread_extent",
            tirx.IntImm("int32", 1),
            tirx.Evaluate(tirx.call_intrin("handle", "tirx.tvm_call_packed", "unsupported_op")),
        ),
    ).with_attr("global_symbol", "bad_kernel")

    with pytest.raises(TileLangSemanticError, match="Unsupported TileLang tile operation"):
        tir_to_sem(prim_func)


def test_tir_to_sem_alloc_buffers_captured_for_gemm():
    """tir_to_sem captures shared/fragment alloc buffers for the gemm kernel."""
    prim_func = _make_gemm_kernel()

    program = tir_to_sem(prim_func)

    scopes = {b.scope for b in program.kernels[0].alloc_buffers}
    assert "shared.dyn" in scopes or "shared" in scopes
    assert "local.fragment" in scopes

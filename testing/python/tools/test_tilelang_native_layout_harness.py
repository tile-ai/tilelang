"""Independent contracts for the layout maintenance harness on native IR."""

from pathlib import Path
import sys

import numpy as np
import pytest
from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim

import tilelang.language as T
from tilelang.layout import Fragment


HARNESS = Path(__file__).resolve().parents[3] / "maint" / "layout_inference"
sys.path.insert(0, str(HARNESS))
try:
    import common
    import cute_model
    import oracle
finally:
    sys.path.remove(str(HARNESS))


def test_native_layout_annotation_keeps_tensor_identity_and_coordinates():
    layout = Fragment((4, 32), forward_thread_fn=lambda i, j: i * 8 + j // 4, forward_index_fn=lambda i, j: j % 4)

    @T.prim_func
    def kernel(A: T.Tensor((4, 32), "float32")):
        T.annotate_layout({A: layout})
        A[0, 0] = 1.0

    result = common.extract_layouts(kernel)
    assert set(result["buffers"]) == {"A"}
    assert result["buffers"]["A"] == common.layout_to_dict(layout)
    assert result["loops"] == {}
    tensor = kernel.params[0]
    assert isinstance(tensor.ty, tirx.TensorType)
    for i in range(4):
        for j in range(32):
            assert int(layout.map_forward_thread([i, j])[0]) == i * 8 + j // 4
            assert int(layout.map_forward_index([i, j])[0]) == j % 4


@pytest.mark.parametrize("begin,end,step,count", [(1, 10, 2, 5), (9, 0, -3, 3), (5, 2, 1, 0), (1, 4, -1, 0)])
def test_loop_key_counts_actual_native_domain(begin, end, step, count):
    i = ir.Var("i", "int32")
    loop = ir.For(i, begin, end, step, ir.ForKind.PARALLEL, [])
    assert common._loop_key(loop) == f"i[{count}]"


def test_loop_key_reads_nested_seq_and_does_not_cross_executable_statement():
    i, j = ir.Var("i", "int32"), ir.Var("j", "int32")
    inner = ir.For(j, 9, 0, -3, ir.ForKind.PARALLEL, [])
    outer = ir.For(i, 1, 10, 2, ir.ForKind.PARALLEL, [inner])
    assert common._loop_key(outer) == "i,j[5x3]"
    separated = ir.For(i, 1, 10, 2, ir.ForKind.PARALLEL, [ir.Evaluate(ir.IntImm("int32", 1)), inner])
    assert common._loop_key(separated) == "i[5]"


def test_loop_key_does_not_guess_unknown_step():
    loop = ir.For(ir.Var("i", "int32"), 0, 10, ir.Var("step", "int32"), ir.ForKind.PARALLEL, [])
    with pytest.raises(ValueError, match="proven nonzero step"):
        common._loop_key(loop)


@pytest.mark.parametrize("replicas,is_store", [(1, False), (1, True), (2, False), (2, True)])
def test_cute_scoring_and_independent_address_oracle(replicas, is_store):
    fragment = Fragment((4, 32), forward_thread_fn=lambda i, j: i * 8 + j // 4, forward_index_fn=lambda i, j: j % 4)
    if replicas > 1:
        fragment = fragment.replicate(replicas)
    measured = oracle.score_statement_oracle(fragment, (4, 32), 4, is_store)
    symbolic = cute_model.score_statement_cute(fragment, (4, 32), 4, is_store)
    assert isinstance(symbolic, cute_model.CuteScore), symbolic
    assert (symbolic.vector, symbolic.issue, symbolic.bw, symbolic.segments) == (
        measured.vector,
        measured.issue,
        measured.bw,
        measured.segments,
    )
    assert measured.vector == 4
    assert measured.issue == 512 * replicas
    assert measured.bw == 512 * (1 if is_store else replicas)
    assert measured.segments == 4 * (1 if is_store else replicas)


def test_integer_oracle_preserves_narrowing_and_widening():
    x = ir.Var("x", "uint32")
    expression = prim.Cast("uint32", prim.Cast("uint8", x))
    values = np.array([0, 1, 255, 256, 257, 65535, 2**32 - 1], dtype=np.uint32)
    np.testing.assert_array_equal(oracle.eval_expr_grid(expression, {x: values}), [0, 1, 255, 0, 1, 255, 255])


def test_integer_oracle_preserves_unsigned_overflow():
    x = ir.Var("x", "uint32")
    expression = x + tirx.const(1, "uint32")
    values = np.array([0, 2**32 - 1], dtype=np.uint32)
    np.testing.assert_array_equal(oracle.eval_expr_grid(expression, {x: values}), [1, 0])


@pytest.mark.parametrize("dtype,minimum,maximum", [("int32", -(2**31), 2**31 - 1), ("int64", -(2**63), 2**63 - 1)])
def test_integer_oracle_rejects_signed_overflow(dtype, minimum, maximum):
    x = ir.Var("x", dtype)
    with pytest.raises(oracle.Unenumerable, match="overflow"):
        oracle.eval_expr_grid(x + tirx.const(1, dtype), {x: np.array([maximum], dtype=dtype)})
    with pytest.raises(oracle.Unenumerable, match="overflow"):
        oracle.eval_expr_grid(x * tirx.const(-1, dtype), {x: np.array([minimum], dtype=dtype)})


def test_integer_oracle_floor_division_and_zero_negative_case():
    x, y = ir.Var("x", "int32"), ir.Var("y", "int32")
    values = np.array([-5, -4, -1, 0, 5], dtype=np.int32)
    np.testing.assert_array_equal(oracle.eval_expr_grid(x // y, {x: values, y: np.array(3)}), [-2, -2, -1, 0, 1])
    np.testing.assert_array_equal(oracle.eval_expr_grid(x % y, {x: values, y: np.array(3)}), [1, 2, 2, 0, 2])
    with pytest.raises(oracle.Unenumerable, match="division by zero"):
        oracle.eval_expr_grid(x // y, {x: values, y: np.array(0)})

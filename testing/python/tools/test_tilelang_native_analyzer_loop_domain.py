"""Static work estimates follow the native For condition for every step sign."""

import pytest
from tvm_tirx import ir, tirx

import tilelang.language as T
from tilelang.tools import Analyzer


@T.prim_func
def copy_four(A: T.Tensor((4,), T.float32), B: T.Tensor((4,), T.float32)):
    T.copy(A, B)


def _loop(begin, end, step):
    variable = ir.Var("i", ir.PrimType("int32"))
    return ir.For(variable, ir.const(begin), ir.const(end), ir.const(step), ir.ForKind.DEFAULT, copy_four.body)


@pytest.mark.parametrize(
    "begin,end,step,expected",
    [(0, 5, 1, 5), (2, 8, 3, 2), (2, 9, 3, 3), (5, 5, 1, 0), (5, 0, 2, 0), (5, 0, -1, 0), (8, 0, -2, 0), (5, 5, -3, 0)],
)
def test_finite_native_trip_count(begin, end, step, expected):
    analyzer = Analyzer(copy_four, None)
    assert analyzer._trip_count(ir.const(begin), ir.const(end), ir.const(step)) == expected


@pytest.mark.parametrize("begin,end", [(0, 5), (5, 5), (5, 0)])
def test_zero_step_keeps_explicit_rejection(begin, end):
    with pytest.raises(ValueError, match="zero loop step"):
        Analyzer(copy_four, None)._trip_count(ir.const(begin), ir.const(end), ir.const(0))


@pytest.mark.parametrize("step", [-1, -3])
def test_nonempty_negative_step_is_not_finite(step):
    with pytest.raises(ValueError, match="finite increasing loop"):
        Analyzer(copy_four, None)._trip_count(ir.const(0), ir.const(5), ir.const(step))


@pytest.mark.parametrize("dynamic_field", ["end", "step"])
def test_unknown_loop_domain_is_not_guessed(dynamic_field):
    symbolic = ir.Var("n", ir.PrimType("int32"))
    end = symbolic if dynamic_field == "end" else ir.const(5)
    step = symbolic if dynamic_field == "step" else ir.const(1)
    with pytest.raises(ValueError, match="needs a concrete"):
        Analyzer(copy_four, None)._trip_count(ir.const(0), end, step)


@pytest.mark.parametrize("begin,end,step,expected_bytes", [(0, 5, 1, 80), (5, 0, -1, 0), (5, 0, 1, 0)])
def test_actual_native_copy_work_preserves_source(begin, end, step, expected_bytes):
    function = copy_four.with_body(_loop(begin, end, step))
    tirx.analysis.check_well_formed(function)
    original = ir.save_json(function)
    analyzer = Analyzer(function, None).ir_pass()
    assert analyzer.total_global_bytes == expected_bytes
    assert analyzer.total_flops == 0
    assert ir.save_json(function) == original
    assert next(iter(analyzer.fn.functions.values())).same_as(function)


def test_actual_native_negative_nonempty_copy_rejects():
    function = copy_four.with_body(_loop(0, 5, -1))
    tirx.analysis.check_well_formed(function)
    with pytest.raises(ValueError, match="finite increasing loop"):
        Analyzer(function, None).ir_pass()


def test_empty_negative_loop_does_not_hide_following_copy():
    function = copy_four.with_body(ir.SeqStmt([_loop(5, 0, -1), copy_four.body]))
    tirx.analysis.check_well_formed(function)
    analyzer = Analyzer(function, None).ir_pass()
    assert analyzer.total_global_bytes == 16
    assert analyzer.loop_stack == []

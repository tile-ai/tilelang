import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx

from tilelang._compat.tensor import TensorView
from tilelang.transform import MergeIfStmt

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def transform(body, params=()):
    function = tirx.Function(params, body, ret_type=ir.TupleType([])).with_attr("global_symbol", "main")
    return MergeIfStmt()(ir.IRModule({"main": function}))["main"]


def test_adjacent_native_ifs_are_merged_without_mutating_input():
    condition = ir.Var("condition", "bool")
    output = TensorView.create("output", [2], "int32")
    first = ir.If(condition, output.store(0, 1), None)
    second = ir.If(condition, output.store(1, 2), None)
    result = transform([first, second], [condition, output.tensor])
    assert len(result.body.seq) == 1
    assert len(result.body.seq[0].then_case.seq) == 2
    assert len(first.then_case.seq) == 1
    assert len(second.then_case.seq) == 1
    assert result.body.seq[0].then_case.seq[0].same_as(first.then_case.seq[0])


def test_names_do_not_replace_variable_identity():
    first_condition = ir.Var("condition", "bool")
    second_condition = ir.Var("condition", "bool")
    body = [ir.If(first_condition, ir.Evaluate(1), None), ir.If(second_condition, ir.Evaluate(2), None)]
    assert len(transform(body, [first_condition, second_condition]).body.seq) == 2


@pytest.mark.parametrize("condition_kind", ["tensor_load", "opaque_call"])
def test_stateful_conditions_are_not_merged(condition_kind):
    memory = TensorView.create("memory", [1], "int32")
    condition = memory.load(0) > 0 if condition_kind == "tensor_load" else tirx.call_extern("guard", ty="bool")
    body = [ir.If(condition, memory.store(0, 1), None), ir.If(condition, memory.store(0, 2), None)]
    assert len(transform(body, [memory.tensor]).body.seq) == 2


def test_else_branch_and_intervening_statements_are_barriers():
    condition = ir.Var("condition", "bool")
    first = ir.If(condition, ir.Evaluate(1), None)
    with_else = ir.If(condition, ir.Evaluate(2), ir.Evaluate(3))
    barrier = ir.Evaluate(tirx.call_extern("barrier", ty="void"))
    body = [first, barrier, first, with_else, first]
    assert len(transform(body, [condition]).body.seq) == 5


def test_merge_is_recursive_and_idempotent():
    condition = ir.Var("condition", "bool")
    inner = ir.Var("inner", "bool")
    first = ir.If(condition, ir.If(inner, ir.Evaluate(1), None), None)
    second = ir.If(condition, ir.If(inner, ir.Evaluate(2), None), None)
    result = transform([first, second], [condition, inner])
    assert len(result.body.seq) == 1
    assert len(result.body.seq[0].then_case.seq) == 1
    assert len(result.body.seq[0].then_case.seq[0].then_case.seq) == 2
    repeated = MergeIfStmt()(ir.IRModule({"main": result}))["main"]
    assert tvm_ffi.structural_equal(result, repeated)


def test_merging_keeps_bindings_in_distinct_lexical_scopes():
    condition = ir.Var("condition", "bool")
    output = TensorView.create("output", [2], "int32")
    first_value = ir.Var("value", "int32")
    second_value = ir.Var("value", "int32")
    first = ir.If(condition, [ir.Bind(first_value, 1), output.store(0, first_value)], None)
    second = ir.If(condition, [ir.Bind(second_value, 2), output.store(1, second_value)], None)
    result = transform([first, second], [condition, output.tensor])
    first_scope, second_scope = result.body.seq[0].then_case.seq
    assert first_scope.op.name == second_scope.op.name == "tirx.scope"
    assert first_scope.body.seq[0].var.same_as(first_value)
    assert second_scope.body.seq[0].var.same_as(second_value)
    assert tirx.analysis.check_well_formed(result)


def test_merge_preserves_first_guard_location():
    condition = ir.Var("condition", "bool")
    location = ir.SourceLoc(ir.SourceName("kernel.py"), 2, 0, 3, 12)
    first = ir.If(condition, ir.Evaluate(1), None, loc=location)
    second = ir.If(condition, ir.Evaluate(2), None)
    assert transform([first, second], [condition]).body.seq[0].loc.same_as(location)


@pytest.mark.parametrize("condition", [False, True])
def test_native_cpu_execution_after_merge(condition):
    output = TensorView.create("output", [2], "int32")
    condition_var = ir.Var("condition", "bool")
    first_value = ir.Var("value", "int32")
    second_value = ir.Var("value", "int32")
    first = ir.If(condition_var, [ir.Bind(first_value, 11), output.store(0, first_value)], None)
    second = ir.If(condition_var, [ir.Bind(second_value, 23), output.store(1, second_value)], None)
    function = transform([first, second], [output.tensor, condition_var])
    executable = tirx.compile(function, target="c")
    actual = runtime.tensor(np.full(2, -1, dtype="int32"))
    executable(actual, condition)
    expected = [11, 23] if condition else [-1, -1]
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_function_declaration_is_preserved():
    assert transform(None).body is None

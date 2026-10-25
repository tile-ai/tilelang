import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

from tilelang._compat.tensor import TensorView


pytestmark = pytest.mark.usefixtures("native_layouts")


def prove(first, second, goal):
    def entries(values):
        return [[ir.StringImm(entry[0]), *entry[1:]] for entry in values]

    return tvm_ffi.get_global_func("tl.analysis.ConstrSetsProve")(entries(first), entries(second), goal)


@pytest.mark.parametrize("kind", ["pred", "assume"])
def test_merge_restores_definition_before_predicate(kind):
    index = ir.Var("index", "int32")
    inner, outer = ir.Var("inner", "bool"), ir.Var("outer", "bool")
    bindings = [["range", index, ir.const(0), ir.const(256)], ["bind", inner, index < 64], ["bind", outer, ir.prim.And(inner, index < 32)]]
    assert prove([[kind, outer]], bindings, index < 32)


def test_constraints_preserve_eager_binding_order():
    index, captured = ir.Var("index", "int32"), ir.Var("captured", "int32")
    entries = [["range", index, ir.const(0), ir.const(256)], ["pred", index < 64], ["bind", captured, index]]
    assert prove(entries, [], captured < 64)


def test_unknown_boolean_does_not_constrain_an_index():
    index, condition = ir.Var("index", "int32"), ir.Var("condition", "bool")
    assert not prove([["pred", condition]], [["range", index, ir.const(0), ir.const(256)]], index < 64)


def test_mutable_reads_in_separate_bindings_are_not_equated():
    tensor = TensorView.create("state", [1], "int32", scope="local")
    first, second = ir.Var("first", "int32"), ir.Var("second", "int32")
    assert not prove([["bind", first, tensor.load(0)]], [["bind", second, tensor.load(0)]], first == second)


def test_mutable_let_copies_renew_local_definitions():
    tensor = TensorView.create("state", [1], "int32", scope="local")
    captured = ir.Var("captured", "int32")
    first, second = ir.Var("first", "int32"), ir.Var("second", "int32")
    value = ir.prim.Let(captured, tensor.load(0), captured)
    assert not prove([["bind", first, value]], [["bind", second, value]], first == second)


def check_parallel(body, params, capfd):
    function = tirx.Function(params, body)
    result = tvm_ffi.get_global_func("tl.transform.VerifyParallelLoop")()(ir.IRModule({"main": function}))["main"]
    ir.assert_structural_equal(function, result)
    return capfd.readouterr().err


@pytest.mark.parametrize("injective,same_value,racy", [(True, False, False), (False, True, False), (False, False, True)])
def test_parallel_store_proofs_rename_inner_binds(injective, same_value, racy, capfd):
    output = TensorView.create("output", [4, 8], "int32")
    row, column = ir.Var("row", "int32"), ir.Var("column", "int32")
    first, second = ir.Var("first", "int32"), ir.Var("second", "int32")
    body = [
        ir.Bind(first, row),
        ir.Bind(second, column),
        output.store([first, second] if injective else [0, 0], ir.const(1) if same_value else row * 8 + column),
    ]
    body = ir.For(row, 0, 4, 1, ir.ForKind.PARALLEL, ir.For(column, 0, 8, 1, ir.ForKind.PARALLEL, body))
    assert ("Data race detected" in check_parallel(body, [output.tensor], capfd)) == racy


def test_parallel_condition_and_index_loads_are_conservative(capfd):
    indices = TensorView.create("indices", [4], "int32")
    output = TensorView.create("output", [4], "int32")
    index = ir.Var("index", "int32")
    condition = ir.Var("condition", "bool")
    body = [ir.Bind(condition, index < 4), ir.If(condition, output.store(indices.load(index), index), None)]
    body = ir.For(index, 0, 4, 1, ir.ForKind.PARALLEL, body)
    assert "Data race detected" in check_parallel(body, [indices.tensor, output.tensor], capfd)

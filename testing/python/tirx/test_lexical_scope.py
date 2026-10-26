import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("tilelang_builders")

import tvm_tirx
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.script import tirx as T

import tilelang.language as tl


def test_scope_script_and_json_roundtrip():
    @T.function
    def function(output: T.Tensor((4,), "float32")):
        with T.region("tl.scope", []):
            temporary = T.alloc_tensor(ty_args=[T.Tensor((4,), "float32", scope="local")])
            for index in range(4):
                temporary[index] = T.float32(index)
            for index in range(4):
                output[index] = temporary[index]

    scope = function.body.seq[0]
    assert scope.op.name == "tl.scope"
    assert not scope.args and not scope.body_params and not scope.result_vars
    assert tirx.analysis.check_well_formed(function)
    ir.assert_structural_equal(function, ir.load_json(ir.save_json(function)))
    parsed = tvm_tirx.script.from_source(function.script(), extra_vars={"T": T, "tl": tl})
    ir.assert_structural_equal(function, parsed)


@pytest.mark.parametrize("invalid_field", ["args", "body_params", "result_vars"])
def test_scope_rejects_operands_parameters_and_results(invalid_field):
    fields = {"args": [], "body_params": [], "attrs": None, "result_vars": []}
    fields[invalid_field] = [ir.Var("value", "int32")]
    with pytest.raises((TypeError, ValueError)):
        ir.RegionStmt("tl.scope", body=ir.SeqStmt([]), **fields)


def test_scope_preserves_owner_metadata():
    scope = ir.RegionStmt("tl.scope", [], [], {"name_hint": "kernel"}, ir.Evaluate(0))
    function = tirx.Function([], scope)
    assert tirx.analysis.check_well_formed(function)
    ir.assert_structural_equal(function, ir.load_json(ir.save_json(function)))
    parsed = tvm_tirx.script.from_source(function.script(), extra_vars={"T": T, "tl": tl})
    ir.assert_structural_equal(function, parsed)


def test_scope_bound_variables_cannot_escape():
    variable = ir.Var("value", "int32")
    scope = ir.RegionStmt("tl.scope", [], [], None, ir.Bind(variable, 1))
    function = tirx.Function([], [scope, ir.Evaluate(variable)])
    assert not tirx.analysis.check_well_formed(function, assert_mode=False)


def test_nested_scopes_execute_on_cpu():
    @T.function
    def function(output: T.Tensor((4,), "float32")):
        for outer in range(2):
            with T.region("tl.scope", []):
                temporary = T.alloc_tensor(ty_args=[T.Tensor((2,), "float32", scope="local")])
                with T.region("tl.scope", []):
                    for inner in range(2):
                        temporary[inner] = T.float32(outer * 2 + inner)
                for inner in range(2):
                    output[outer * 2 + inner] = temporary[inner]

    executable = tirx.compile(function, target="c")
    output = runtime.empty((4,), "float32")
    executable(output)
    np.testing.assert_array_equal(output.numpy(), np.arange(4, dtype="float32"))

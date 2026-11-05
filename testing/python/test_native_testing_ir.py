"""Queries keep exact op identity, per-function counts, and native type uses."""

import pytest
import tvm_tirx
from tvm_tirx import ir, tirx

from tilelang.testing.ir import (
    assert_call_count,
    collect_calls,
    collect_extern_calls,
    collect_nodes,
)


def _call(name="helper_test", value=1):
    return tirx.call_extern(name, ir.const(value), ty=ir.PrimType("int32"))


def _function(body, name="helper"):
    return tirx.Function([], body, ir.PrimType("void")).with_attr("global_symbol", name)


def test_native_queries_deduplicate_shared_objects_per_function():
    call = _call()
    stmt = ir.Evaluate(call)
    function = _function([stmt, stmt])
    assert len(collect_nodes(function, ir.Evaluate)) == 1
    assert_call_count(function, op="tirx.call_extern", count=1)
    module = tvm_tirx.IRModule({"first": function, "second": function})
    assert_call_count(module, op="tirx.call_extern", count=2)
    assert_call_count(module, op="tirx.call_extern", count=1, function="second")


def test_native_queries_follow_tensor_type_shape_uses():
    n = ir.Var("n", ir.PrimType("int32"))
    shape = n + 1
    tensor = ir.Var("A", tirx.TensorType([shape], "float32"))
    body = ir.Evaluate(ir.TensorLoad(tensor, [ir.const(0)]))
    additions = collect_nodes(body, ir.prim.Add)
    assert len(additions) == 1 and additions[0].same_as(shape)
    variables = collect_nodes(body, ir.Var)
    assert any(var.same_as(n) for var in variables)
    assert any(var.same_as(tensor) for var in variables)


def test_native_queries_do_not_visit_function_signature_or_module_metadata():
    signature_call = _call("signature_only")
    tensor = ir.Var("A", tirx.TensorType([signature_call], "float32"))
    function = tirx.Function([tensor], [ir.Evaluate(_call("body_only"))], ir.PrimType("void"))
    assert [str(call.args[0].value) for call in collect_calls(function, op="tirx.call_extern")] == ["body_only"]
    declaration = tirx.Function([], None, ir.PrimType("void"))
    module = tvm_tirx.IRModule({"fn": function, "declaration": declaration})
    assert_call_count(module, op="tirx.call_extern", count=1)
    assert collect_nodes(declaration, ir.Call) == []


def test_native_call_index_does_not_keep_old_function_body_or_caller_list():
    original = _function([ir.Evaluate(_call("one"))])
    calls = collect_calls(original, op="tirx.call_extern")
    calls.clear()
    assert_call_count(original, op="tirx.call_extern", count=1)
    changed = original.with_body([ir.Evaluate(_call("one")), ir.Evaluate(_call("two"))])
    assert_call_count(changed, op="tirx.call_extern", count=2)
    assert_call_count(original, op="tirx.call_extern", count=1)


def test_native_extern_query_matches_complete_symbol_only():
    function = _function([ir.Evaluate(_call("kernel")), ir.Evaluate(_call("kernel_extra"))])
    assert len(collect_extern_calls(function, symbol="kernel")) == 1
    assert collect_extern_calls(function, symbol="kern") == []
    with pytest.raises(TypeError, match="symbol must be a string"):
        collect_extern_calls(function, symbol=3)


def test_native_query_validation_and_diagnostics():
    function = _function([ir.Evaluate(_call("actual"))], "named_function")
    with pytest.raises(ValueError, match="No function"):
        collect_nodes(tvm_tirx.IRModule({"main": function}), ir.Call, function="missing")
    with pytest.raises(ValueError, match="only be selected"):
        collect_nodes(function, ir.Call, function="main")
    with pytest.raises(TypeError, match="node_type"):
        collect_nodes(function, str)
    with pytest.raises(ValueError, match="count must"):
        assert_call_count(function, op="tirx.call_extern", count=True)
    with pytest.raises(AssertionError, match="named_function"):
        assert_call_count(function, op="tirx.call_extern", count=0)
    with pytest.raises(AttributeError, match="not registered"):
        assert_call_count(function, op="tl.missing_testing_operator", count=0)

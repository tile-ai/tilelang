import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView
from tilelang.transform import LetInline

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def apply(body, parameters):
    function = tirx.Function(parameters, body, ret_type=ir.TupleType([])).with_attr("global_symbol", "inline_test")
    rewritten = LetInline()(ir.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(rewritten)
    ir.assert_structural_equal(rewritten, ir.load_json(ir.save_json(rewritten)))
    return rewritten


def bindings(function):
    result = []
    stmt_functor.post_order_visit(function.body, lambda node: result.append(node) if isinstance(node, ir.Bind) else None)
    return result


def test_frontend_pure_binding_and_adjacent_read_are_inlined():
    source = TensorView.create("source", [16], "float32")
    output = TensorView.create("output", [16], "float32")
    index = ir.Var("index", "int32")
    factor = ir.Var("factor", "float32")
    value = ir.Var("value", "float32")
    body = [ir.Bind(factor, ir.const(2.0, "float32")), ir.Bind(value, source.load(index) * factor), output.store(index, value)]
    function = apply(ir.For(index, 0, 16, 1, ir.ForKind.PARALLEL, body), [source.tensor, output.tensor])
    assert not bindings(function)


def test_read_binding_is_not_moved_across_a_store():
    output = TensorView.create("output", [2], "int32")
    saved = ir.Var("saved", "int32")
    function = apply([ir.Bind(saved, output.load(0)), output.store(0, 7), output.store(1, saved)], [output.tensor])
    assert len(bindings(function)) == 1
    executable = tirx.compile(function, target="c")
    actual = runtime.tensor(np.array([3, 0], dtype="int32"))
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), [7, 3])


def test_native_tensor_allocation_is_not_inlined():
    output = TensorView.create("output", [1], "int32")
    temporary = TensorView.create("temporary", [1], "int32", scope="local")
    function = apply([temporary.allocate(), temporary.store(0, 9), output.store(0, temporary.load(0))], [output.tensor])
    assert bindings(function)[0].var.same_as(temporary.tensor)
    executable = tirx.compile(function, target="c")
    actual = runtime.empty((1,), "int32")
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), [9])


def test_adjacent_read_used_after_a_write_is_not_replayed():
    output = TensorView.create("output", [3], "int32")
    saved = ir.Var("saved", "int32")
    function = apply(
        [ir.Bind(saved, output.load(0)), output.store(1, saved), output.store(0, 7), output.store(2, saved)],
        [output.tensor],
    )
    assert len(bindings(function)) == 1
    executable = tirx.compile(function, target="c")
    actual = runtime.tensor(np.array([3, 0, 0], dtype="int32"))
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), [7, 3, 3])


def test_opaque_binding_retains_single_evaluation():
    output = TensorView.create("output", [2], "int32")
    result = ir.Var("result", "int32")
    call = tirx.call_extern("effect", ty="int32")
    function = apply([ir.Bind(result, call), output.store(0, result), output.store(1, result)], [output.tensor])
    assert len(bindings(function)) == 1


def test_pure_let_bindings_do_not_escape_their_expression_scope():
    output = TensorView.create("output", [1], "int32")
    bound = ir.Var("bound", "int32")
    left = ir.prim.Let(bound, ir.const(3, "int32"), bound + 1)
    right = ir.prim.Let(bound, ir.const(8, "int32"), bound + 2)
    function = apply(output.store(0, left + right), [output.tensor])
    executable = tirx.compile(function, target="c")
    actual = runtime.empty((1,), "int32")
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), [14])


def test_pointer_metadata_is_rebound_when_a_pure_binding_is_removed():
    output = TensorView.create("output", [16], "float32")
    index = ir.Var("index", "int32")
    offset = ir.Var("offset", "int32")
    pointer = tvm_ffi.get_global_func("tl.MakeAccessPtrFromBufferLoad")(output.load(offset), 3)
    pointer = tvm_ffi.get_global_func("tl.LowerAccessPtrCall")(pointer)
    body = [ir.Bind(offset, index * 4), ir.Evaluate(tirx.call_extern("effect", pointer, ty="void"))]
    function = apply(ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, body), [output.tensor])
    assert not bindings(function)

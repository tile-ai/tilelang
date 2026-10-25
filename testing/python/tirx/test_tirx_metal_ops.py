import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView


pytestmark = pytest.mark.usefixtures("native_layouts")


def lower(name, arguments, threads=32):
    operation = tvm_ffi.get_global_func("tl.ParseOperator")(ir.Call(f"tl.tileop.{name}", arguments), {})
    thread = ir.Var("thread", "int32")
    target = Target("metal")
    with target:
        body = tvm_ffi.get_global_func("tl.testing.LowerOperator")(operation, target, ir.Range(0, threads), thread, {}, {})
    return thread, body


def calls(body, name):
    result = []
    stmt_functor.post_order_visit(
        body,
        lambda node: result.append(node) if isinstance(node, ir.Call) and isinstance(node.op, ir.Op) and node.op.name == name else None,
    )
    return result


@pytest.mark.parametrize(
    "scope,shape,name,count",
    [
        ("metal.simdgroup", [8, 8], "tirx.metal.make_filled_simdgroup_matrix", 1),
        ("metal.simdgroup", [16, 8], "tirx.metal.make_filled_simdgroup_matrix", 2),
        ("metal.cooperative_tensor", [16, 32], "tirx.metal.cooperative_tensor_fill", 1),
        ("metal.cooperative_tensor", [32, 32], "tirx.metal.cooperative_tensor_fill", 2),
    ],
)
def test_native_matrix_fill(scope, shape, name, count):
    output = TensorView.create("output", shape, "float16", scope=scope)
    _, body = lower("fill", [output.tensor, ir.const(3, "int32")])
    intrinsics = calls(body, name)
    assert len(intrinsics) == count
    assert [int(call.args[1]) for call in intrinsics] == list(range(count))
    assert all(str(call.args[2].ty) == "float16" for call in intrinsics)
    function = tirx.Function([output.tensor], body)
    assert tirx.analysis.check_well_formed(function)
    ir.assert_structural_equal(function, ir.load_json(ir.save_json(function)))


@pytest.mark.parametrize(
    "scope,shape,name",
    [
        ("metal.simdgroup", [8, 8], "tirx.metal.simdgroup_store"),
        ("metal.cooperative_tensor", [16, 32], "tirx.metal.cooperative_tensor_store"),
    ],
)
def test_matrix_copy_uses_physical_row_stride(scope, shape, name):
    source = TensorView.create("source", shape, "float32", scope=scope)
    output = TensorView.create("output", [shape[0] + 2, shape[1] + 4], "float32", strides=[shape[1] + 9, 1])
    region = output.region([ir.Range(1, shape[0] + 1), ir.Range(2, shape[1] + 2)])
    thread, body = lower("copy", [source.tensor, region])
    intrinsics = calls(body, name)
    assert len(intrinsics) == 1
    assert int(intrinsics[0].args[3]) == shape[1] + 9
    function = tirx.Function([source.tensor, output.tensor, thread], body)
    assert tirx.analysis.check_well_formed(function)
    with pytest.raises(ValueError, match="complete SIMD groups"):
        lower("copy", [source.tensor, region], threads=17)


@pytest.mark.parametrize("name", ["fill", "copy", "transpose"])
def test_metal_simt_operations_in_cpu_simulation(name):
    source = TensorView.create("source", [8, 16], "int32")
    output_shape = [16, 8] if name == "transpose" else [8, 16]
    output = TensorView.create("output", output_shape, "int32")
    arguments = [output.tensor, ir.const(11)] if name == "fill" else [source.tensor, output.tensor]
    thread, body = lower(name, arguments)
    function = tirx.Function([source.tensor, output.tensor], ir.For(thread, 0, 32, 1, ir.ForKind.DEFAULT, body)).with_attr(
        "global_symbol", "metal_simulation"
    )
    module = tirx.transform.VectorizeLoop(False)(ir.IRModule({"main": function}))
    executable = tirx.compile(module, target="c")
    values = np.arange(128, dtype="int32").reshape(8, 16)
    actual = runtime.tensor(np.zeros(output_shape, dtype="int32"))
    executable(runtime.tensor(values), actual)
    expected = np.full(output_shape, 11) if name == "fill" else values.T if name == "transpose" else values
    np.testing.assert_array_equal(actual.numpy(), expected)

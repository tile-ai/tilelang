import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx

from tilelang.transform import _tirx_passes

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def _allocation_count(module):
    allocations = []

    def visit(node, visitor):
        visitor.default_visit(node)
        if isinstance(node.op, ir.Op) and node.op.name == "tirx.alloc_tensor":
            allocations.append(node)

    tvm_ffi.structural_visit(module, (ir.Call, visit))
    return len(allocations)


@pytest.mark.parametrize("allow_local_inplace", [False, True])
@pytest.mark.parametrize("detect_inplace", [False, True])
@pytest.mark.parametrize("initialize_output", [False, True])
def test_local_inplace_requires_explicit_permission_and_safe_lifetime(allow_local_inplace, detect_inplace, initialize_output):
    local_type = tirx.TensorType([1], "int32", scope="local")
    first = ir.Var("first", local_type)
    second = ir.Var("second", local_type)
    output = ir.Var("output", tirx.TensorType([1], "int32"))
    body = [
        ir.Bind(first, ir.Call("tirx.alloc_tensor", [], ty_args=[local_type])),
        ir.TensorStore(first, [0], ir.const(3)),
        ir.Bind(second, ir.Call("tirx.alloc_tensor", [], ty_args=[local_type])),
    ]
    if initialize_output:
        body.append(ir.TensorStore(second, [0], ir.const(0)))
    body.extend(
        [
            ir.TensorStore(second, [0], ir.TensorLoad(first, [0]) * 2),
            ir.TensorStore(output, [0], ir.TensorLoad(second, [0])),
        ]
    )
    function = tirx.Function([output], body).with_attr("global_symbol", "local_inplace")
    module = _tirx_passes.StorageRewrite(
        enable_reuse=True,
        detect_inplace=detect_inplace,
        allow_local_inplace=allow_local_inplace,
    )(ir.IRModule({"main": function}))
    expected_allocations = 1 if allow_local_inplace and detect_inplace and not initialize_output else 2
    assert _allocation_count(module) == expected_allocations
    assert tirx.analysis.check_well_formed(module)
    compiled = tirx.compile(module, target="c")
    actual = runtime.empty((1,), "int32")
    compiled(actual)
    np.testing.assert_array_equal(actual.numpy(), [6])


def test_local_inplace_permission_does_not_enable_lifetime_reuse():
    local_type = tirx.TensorType([128], "int32", scope="local")
    first = ir.Var("first", local_type)
    second = ir.Var("second", local_type)
    output = ir.Var("output", tirx.TensorType([2], "int32"))
    body = [
        ir.Bind(first, ir.Call("tirx.alloc_tensor", [], ty_args=[local_type])),
        ir.TensorStore(first, [0], ir.const(3)),
        ir.TensorStore(output, [0], ir.TensorLoad(first, [0])),
        ir.Bind(second, ir.Call("tirx.alloc_tensor", [], ty_args=[local_type])),
        ir.TensorStore(second, [0], ir.const(6)),
        ir.TensorStore(output, [1], ir.TensorLoad(second, [0])),
    ]
    function = tirx.Function([output], body).with_attr("global_symbol", "local_separate")
    module = _tirx_passes.StorageRewrite(allow_local_inplace=True)(ir.IRModule({"main": function}))
    assert _allocation_count(module) == 2
    compiled = tirx.compile(module, target="c")
    actual = runtime.empty((2,), "int32")
    compiled(actual)
    np.testing.assert_array_equal(actual.numpy(), [3, 6])

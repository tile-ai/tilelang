import uuid

import pytest
import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

from tilelang.transform import _tirx_passes

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def _scope_with_property(value):
    scope = tvm_tirx.runtime.MemoryScope.register("test.storage_rewrite." + uuid.uuid4().hex)
    if value != "absent":
        tvm_tirx.runtime.MemoryScope.def_attr("runtime.memory_alloc_property")[scope] = value
    return scope


def _rewrite(func):
    @I.ir_module(check_well_formed=False)
    class Before:
        main = func

    return _tirx_passes.StorageRewrite()(Before)["main"]


@pytest.mark.parametrize("attributes", [None, {}])
def test_allocation_without_attributes(attributes):
    from tvm_tirx import ir, tirx

    tensor_type = tirx.TensorType((1,), "float32", scope="local")
    temporary = ir.Var("temporary", tensor_type)
    output = ir.Var("output", tirx.TensorType((1,), "float32"))
    allocation = ir.Call("tirx.alloc_tensor", [], attrs=attributes, ty_args=[tensor_type])
    function = tirx.Function(
        [output],
        [
            ir.Bind(temporary, allocation),
            ir.TensorStore(temporary, [0], ir.const(7.0, "float32")),
            ir.TensorStore(output, [0], ir.TensorLoad(temporary, [0])),
        ],
    )
    result = _rewrite(function)
    assert tirx.analysis.check_well_formed(result)
    assert result.body is not None


@pytest.mark.parametrize("scope_name", ["shared", "shared.dyn"])
@pytest.mark.parametrize("enable_reuse", [False, True])
def test_preserve_shared_allocations_for_downstream_planner(scope_name, enable_reuse):
    @T.function
    def function(output: T.Tensor((2,), "float32")):
        first = T.alloc_tensor(ty_args=[T.Tensor((128,), "float32", scope_name)])
        second = T.alloc_tensor(ty_args=[T.Tensor((128,), "float32", scope_name)])
        for index in T.serial(128):
            first[index] = T.float32(index)
        output[0] = first[127]
        for index in T.serial(128):
            second[index] = T.float32(index + 1)
        output[1] = second[127]

    module = tvm_tirx.ir.IRModule({"main": function})
    rewritten = _tirx_passes.StorageRewrite(enable_reuse=enable_reuse, preserve_shared_allocations=True)(module)
    tvm_tirx.ir.assert_structural_equal(rewritten["main"], function)
    assert tvm_tirx.tirx.analysis.check_well_formed(rewritten)

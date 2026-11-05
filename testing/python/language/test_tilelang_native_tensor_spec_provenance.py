"""Only parameter stride provenance needs a separate source ABI annotation."""

import numpy as np
import pytest
import torch
from tvm_tirx import ir, tirx

import tilelang
import tilelang.language as T
from tilelang.language import _native
from tilelang.testing.ir import collect_calls, collect_nodes
from tilelang.testing.packed import compile_packed_function


@pytest.mark.parametrize("constructor", ["decl_buffer", "decl_tensor", "strided_tensor"])
def test_local_views_retain_physical_layout_without_a_source_abi_marker(constructor):
    origin = 0 if constructor == "strided_tensor" else 2

    @T.prim_func
    def kernel(source: T.Tensor((16,), "int32"), output: T.Tensor((1,), "int32")):
        if constructor == "decl_buffer":
            view = T.decl_buffer((2,), "int32", strides=(3,), elem_offset=origin, data=source.data)
        elif constructor == "decl_tensor":
            view = T.decl_tensor(source.data, ty_args=[T.Buffer((2,), "int32", strides=(3,), elem_offset=origin)])
        else:
            view = T.StridedTensor((2,), (3,), "int32", data=source.data)
        output[0] = view[1]

    assert not collect_calls(kernel, op="tl.annotation")
    (declaration,) = collect_calls(kernel, op="tirx.decl_tensor")
    view = next(binding.var for binding in collect_nodes(kernel, ir.Bind) if binding.value.same_as(declaration))
    assert [int(stride) for stride in _native.tensor_strides(view)] == [3]
    assert int(_native.tensor_offset(view)) == origin
    for function in (kernel, ir.load_json(ir.save_json(kernel)), tirx.renew_def(kernel)):
        assert tirx.analysis.check_well_formed(function)
        assert not collect_calls(function, op="tl.annotation")
    execute = compile_packed_function(kernel)
    data = torch.arange(16, dtype=torch.int32) * 13 - 7
    output = torch.full((1,), -99, dtype=torch.int32)
    execute(data, output)
    np.testing.assert_array_equal(output.numpy(), data[origin + 3 : origin + 4].numpy())


@pytest.mark.parametrize("parameter_kind", ["annotation", "matched"])
def test_explicit_singleton_parameter_stride_still_rejects_another_descriptor(parameter_kind):
    annotation = T.handle if parameter_kind == "matched" else T.StridedTensor((1, 7), (7, 1), "int32")

    @T.prim_func
    def kernel(input_value: annotation, output: T.Tensor((1,), "int32")):
        if parameter_kind == "matched":
            source = T.match_buffer(input_value, (1, 7), "int32", strides=(7, 1))
        else:
            source = input_value
        view = T.decl_buffer((2,), "int32", strides=(3,), data=source.data)
        output[0] = view[1]

    lowered = tilelang.transform.LowerTensorSpec()(ir.IRModule({"main": kernel}))
    assert list(lowered["main"].attrs["tl.explicit_stride_params"]) == [True, False]
    assert tirx.analysis.check_well_formed(lowered)
    execute = compile_packed_function(kernel)
    data = torch.arange(7, dtype=torch.int32).view(1, 7)
    output = torch.full((1,), -99, dtype=torch.int32)
    execute(data, output)
    assert output.item() == 3
    # The singleton leading stride has the same addressed elements, but an
    # explicit source stride remains an observable argument requirement.
    incompatible = data.as_strided((1, 7), (5, 1))
    with pytest.raises(RuntimeError, match="strides"):
        execute(incompatible, output)

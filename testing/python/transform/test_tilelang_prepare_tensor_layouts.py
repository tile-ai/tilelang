"""Layout preparation fixes definition identity without changing tensor ABI."""

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim
from tvm_tirx.target import Target

import tilelang
from tilelang.testing.packed import compile_packed_function


def prepare(function):
    return tilelang.transform.PrepareTensorLayouts()(ir.IRModule({"main": function}))["main"]


def make_function(*, view=False, explicit_scalar=True):
    n = ir.Var("n", "int32")
    extent = prim.max(n, ir.const(0, "int32"))
    input_extent = 2 * extent + 1 if view else extent
    source_type = tirx.TensorType([input_extent], "int32", align=4)
    output_type = tirx.TensorType([extent], "int32", align=4)
    source = ir.Var("tensor", source_type)
    output = ir.Var("tensor", output_type)
    accessed = source
    statements = []
    if view:
        from tvm_tirx.tirx.layout import Axis, Iter

        memory = Axis.get("m")
        layout = tirx.TileLayout.from_iters([Iter(extent, 2, memory)], offset={memory: ir.const(1, "int32")})
        view_type = tirx.TensorType([extent], "int32", align=4, layout=layout)
        accessed = ir.Var("tensor", view_type)
        statements.append(
            ir.Bind(
                accessed,
                ir.Call(ir.Op.get("tirx.decl_tensor"), [source.data], ty=view_type, ty_args=[view_type]),
            )
        )
    index = ir.Var("i", "int32")
    statements.append(
        ir.For(
            index,
            ir.const(0, "int32"),
            extent,
            ir.const(1, "int32"),
            ir.ForKind.DEFAULT,
            [ir.TensorStore(output, [index], ir.TensorLoad(accessed, [index]) + 3)],
        )
    )
    params = [source, output, n] if explicit_scalar else [source, output]
    function = tirx.Function(params, statements, ret_type=ir.NoneType()).with_attr("global_symbol", "main").with_attr("target", Target("c"))
    return function


@pytest.mark.parametrize("view", [False, True])
@pytest.mark.parametrize("roundtrip", ["direct", "json", "renew"])
def test_prepare_rebinds_current_definitions_and_uses(view, roundtrip):
    function = make_function(view=view, explicit_scalar=False)
    if roundtrip == "json":
        function = ir.load_json(ir.save_json(function))
    elif roundtrip == "renew":
        function = tirx.renew_def(function)
    original = ir.save_json(function)
    prepared = prepare(function)
    assert ir.save_json(function) == original
    assert all(parameter.ty.layout is None for parameter in function.params)
    assert all(parameter.ty.layout is not None for parameter in prepared.params)
    assert not prepared.params[0].same_as(prepared.params[1])
    assert tirx.analysis.check_well_formed(prepared)
    assert tvm_ffi.structural_equal(prepare(prepared), prepared)
    module = tirx.transform.LowerTensorLayouts()(ir.IRModule({"main": prepared}))
    assert tirx.analysis.check_well_formed(module)
    module = tirx.transform.FlattenTensor()(module)
    assert tirx.analysis.check_well_formed(module)
    for before, after in zip(function.params, module["main"].params):
        assert tvm_ffi.structural_equal(before.ty, after.ty)


@pytest.mark.parametrize("view", [False, True])
@pytest.mark.parametrize("n", [-2, 0, 1, 7])
def test_prepared_layouts_keep_offset_stride_and_negative_scalar_execution(view, n):
    function = prepare(make_function(view=view))
    executable = compile_packed_function(function)
    extent = max(n, 0)
    input_extent = 2 * extent + 1 if view else extent
    source = np.arange(input_extent, dtype="int32") * 5 - 4
    output = np.full(extent, -991, dtype="int32")
    executable(tvm_ffi.from_dlpack(source), tvm_ffi.from_dlpack(output), n)
    expected = source[1::2] + 3 if view else source + 3
    np.testing.assert_array_equal(output, expected)


def test_prepare_preserves_native_declarations():
    variable = ir.Var("A", tirx.TensorType([ir.Var("n", "int32")], "float32"))
    function = tirx.Function([variable], [], attrs=ir.DictAttrs({"declaration": True}))
    result = prepare(function)
    assert result.same_as(function)
    assert result.params[0].same_as(variable)
    assert len(result.body.seq) == 0


def test_dead_parameter_aliases_remain_nullable_but_live_data_is_required():
    tensor_type = tirx.TensorType([1], "int32", align=4)
    unused = ir.Var("same", tensor_type)
    used = ir.Var("same", tensor_type)
    output = ir.Var("result", tensor_type)
    function = tirx.Function(
        [unused, used, output],
        [ir.TensorStore(output, [0], ir.TensorLoad(used, [0]) + 3)],
    ).with_attr("global_symbol", "main")
    executable = compile_packed_function(function)
    data = tvm_ffi.from_dlpack(np.array([7], dtype="int32"))
    result = np.full(1, -991, dtype="int32")
    destination = tvm_ffi.from_dlpack(result)
    executable(None, data, destination)
    np.testing.assert_array_equal(result, [10])
    result.fill(-991)
    with pytest.raises(RuntimeError, match="Tensor|non-NULL"):
        executable(data, None, destination)
    np.testing.assert_array_equal(result, [-991])


def test_unused_tensor_alias_keeps_pointer_callback_effects_and_errors():
    import ctypes

    seen = []
    storage = (ctypes.c_int32 * 1)(7)
    name = "tl.testing.unused_tensor_alias.pointer_callback"

    def pointer_callback(value):
        seen.append(value)
        if value < 0:
            raise ValueError("unused-alias-sentinel")
        return ctypes.c_void_p(ctypes.addressof(storage))

    tvm_ffi.register_global_func(name, pointer_callback, override=True)
    tensor_type = tirx.TensorType([1], "int32", align=4)
    unused = ir.Var("unused", tensor_type)
    output = ir.Var("output", tensor_type)
    argument = ir.Var("argument", "int32")
    pointer = tirx.call_packed(name, argument, ty=ir.PtrType("int32"))
    function = tirx.Function(
        [output, argument],
        [
            ir.Bind(unused, ir.Call("tirx.decl_tensor", [pointer], ty=tensor_type, ty_args=[tensor_type])),
            ir.TensorStore(output, [0], ir.const(37, "int32")),
        ],
    ).with_attr("global_symbol", "main")
    assert tirx.analysis.check_well_formed(function)
    executable = compile_packed_function(function)
    result = np.full(1, -991, dtype="int32")
    destination = tvm_ffi.from_dlpack(result)
    executable(destination, 4)
    assert seen == [4]
    np.testing.assert_array_equal(result, [37])
    result.fill(-991)
    with pytest.raises(ValueError, match="^unused-alias-sentinel$"):
        executable(destination, -1)
    assert seen == [4, -1]
    np.testing.assert_array_equal(result, [-991])
    executable(destination, 5)
    assert seen == [4, -1, 5]
    np.testing.assert_array_equal(result, [37])

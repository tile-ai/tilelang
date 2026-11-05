"""Native pointer/type annotations retain their runtime ABI through the facade."""

import ctypes

import pytest
import torch
from tvm_tirx import ir, tirx

import tilelang
import tilelang.cpu.language as T
from tilelang.testing.ir import collect_calls, collect_nodes


def _compile(function):
    return tilelang.compile(function, target="c", target_host="c", execution_backend="tvm_ffi")


def _round_trip(function):
    assert tirx.analysis.check_well_formed(function)
    restored = ir.load_json(ir.save_json(function))
    ir.assert_structural_equal(function, restored)
    assert tirx.analysis.check_well_formed(restored)
    return restored


@pytest.mark.parametrize("source_spec", [False, True])
def test_native_pointer_decl_tensor_keeps_the_data_address(source_spec):
    view_type = T.Tensor((4,), "int32") if source_spec else T.TensorType((4,), "int32")

    @T.prim_func
    def kernel(pointer: T.Ptr("int32"), output: T.TensorType((4,), "int32"), bias: ir.PrimType("int32")):
        view = T.decl_tensor(pointer, ty_args=[view_type])
        for i in T.serial(4):
            output[i] = view[i] + bias

    assert T.Ptr is ir.PtrType
    function = _round_trip(kernel)
    assert [parameter.name for parameter in function.params] == ["pointer", "output", "bias"]
    assert isinstance(function.params[0].ty, ir.PtrType)
    assert isinstance(function.params[1].ty, tirx.TensorType)
    assert function.params[2].ty == ir.PrimType("int32")
    (declaration,) = collect_calls(function, op="tirx.decl_tensor")
    assert declaration.args[0].same_as(function.params[0])
    assert isinstance(declaration.ty_args[0], tirx.TensorType)

    compiled = _compile(function)
    source = torch.tensor([-7, 0, 4, 23], dtype=torch.int32)
    output = torch.full((4,), -99, dtype=torch.int32)
    compiled(ctypes.c_void_p(source.data_ptr()), output, 11)
    torch.testing.assert_close(output, source + 11, rtol=0, atol=0)


def test_mixed_pointer_tensor_and_scalar_keep_dynamic_shape_bindings():
    extent = T.dynamic("extent", "int64")

    @T.prim_func
    def kernel(pointer: T.Ptr("int32"), output: T.TensorType((extent,), "int32"), bias: ir.PrimType("int32")):
        view = T.decl_tensor(pointer, ty_args=[T.Tensor(output.shape, "int32")])
        for i in T.serial(output.shape[0]):
            output[i] = view[i] + bias

    function = _round_trip(kernel)
    assert len(function.params) == 3
    (declaration,) = collect_calls(function, op="tirx.decl_tensor")
    assert declaration.ty.shape[0].same_as(function.params[1].ty.shape[0])
    compiled = _compile(function)
    for count, bias in [(1, -3), (5, 11), (9, -17)]:
        source = torch.arange(count, dtype=torch.int32) * 2
        output = torch.full_like(source, -99)
        compiled(ctypes.c_void_p(source.data_ptr()), output, bias)
        torch.testing.assert_close(output, source + bias, rtol=0, atol=0)


def test_native_body_annotations_materialize_deferred_parameters_in_signature_order():
    @T.prim_func
    def kernel(pointer, output, bias):
        # Declaration order deliberately differs from the public argument order.
        bias: ir.PrimType("int32")
        output: T.TensorType((4,), "int32")
        pointer: T.Ptr("int32")
        view = T.decl_tensor(pointer, ty_args=[T.TensorType((4,), "int32")])
        for i in T.serial(4):
            output[i] = view[i] + bias

    function = _round_trip(kernel)
    assert [parameter.name for parameter in function.params] == ["pointer", "output", "bias"]
    assert isinstance(function.params[0].ty, ir.PtrType)
    assert isinstance(function.params[1].ty, tirx.TensorType)
    assert function.params[2].ty == ir.PrimType("int32")
    compiled = _compile(function)
    source = torch.arange(4, dtype=torch.int32)
    output = torch.full_like(source, -99)
    compiled(ctypes.c_void_p(source.data_ptr()), output, -9)
    torch.testing.assert_close(output, source - 9, rtol=0, atol=0)


def test_eager_native_types_are_runtime_arguments_and_do_not_specialize_on_values():
    @T.prim_func(eager_jit=True)
    def kernel(pointer: T.Ptr("int32"), output: T.TensorType((4,), "int32"), bias: ir.PrimType("int32")):
        view = T.decl_tensor(pointer, ty_args=[T.Tensor((4,), "int32")])
        for i in T.serial(4):
            output[i] = view[i] + bias

    kernel.set_mode("eager")
    first = torch.arange(4, dtype=torch.int32)
    second = first + 20
    output = torch.full_like(first, -99)
    function = kernel.get_tir(ctypes.c_void_p(first.data_ptr()), output, 3)
    again = kernel.get_tir(ctypes.c_void_p(second.data_ptr()), output, 7)
    assert len(kernel.p1_cache) == 1
    ir.assert_structural_equal(function, again)
    assert len(function.params) == 3
    compiled = _compile(_round_trip(function))
    compiled(ctypes.c_void_p(second.data_ptr()), output, -5)
    torch.testing.assert_close(output, second - 5, rtol=0, atol=0)


def test_decl_tensor_source_constraints_reference_a_defined_native_tensor():
    @T.prim_func
    def kernel(pointer: T.Ptr("int32"), output: T.TensorType((4,), "int32")):
        view = T.decl_tensor(pointer, ty_args=[T.Buffer((4,), "int32", offset_factor=4, elem_offset=0)])
        for i in T.serial(4):
            output[i] = view[i]

    function = _round_trip(kernel)
    (declaration,) = collect_calls(function, op="tirx.decl_tensor")
    binding = next(node for node in collect_nodes(function, ir.Bind) if node.value.same_as(declaration))
    annotations = collect_calls(function, op="tl.annotation")
    assert len(annotations) == 1
    assert any(value.same_as(binding.var) for value in collect_nodes(annotations[0], ir.Var))
    compiled = _compile(function)
    source = torch.tensor([2, -4, 8, 16], dtype=torch.int32)
    output = torch.full_like(source, -99)
    compiled(ctypes.c_void_p(source.data_ptr()), output)
    torch.testing.assert_close(output, source, rtol=0, atol=0)


def test_legacy_matched_handles_keep_descriptor_meaning_with_native_scalar_annotation():
    @T.prim_func
    def kernel(input_handle: T.handle, output_handle: T.handle, extent: ir.PrimType("int64")):
        source = T.match_buffer(input_handle, (extent,), "int32")
        output = T.match_buffer(output_handle, (extent,), "int32")
        rank = T.tvm_struct_get(input_handle, 0, 4, "int32")
        for i in T.serial(extent):
            output[i] = source[i] + rank

    function = _round_trip(kernel)
    assert [parameter.name for parameter in function.params] == ["input_handle", "output_handle", "extent"]
    assert all(tirx.is_tensor_var(parameter) for parameter in function.params[:2])
    assert function.params[0].ty.shape[0].same_as(function.params[2])
    assert function.params[1].ty.shape[0].same_as(function.params[2])
    compiled = _compile(function)
    source = torch.tensor([9, -3, 0, 7, 4], dtype=torch.int32)
    output = torch.full_like(source, -99)
    compiled(source, output, 5)
    torch.testing.assert_close(output, source + 1, rtol=0, atol=0)
    output.fill_(-99)
    with pytest.raises(RuntimeError, match="shape"):
        compiled(source, output, 4)
    torch.testing.assert_close(output, torch.full_like(output, -99), rtol=0, atol=0)


@pytest.mark.parametrize("eager", [False, True])
def test_unsupported_native_type_is_not_a_compile_time_parameter(eager):
    annotation = ir.TupleType([])
    with pytest.raises(TypeError, match="native TileLang runtime parameter requires"):

        @T.prim_func(eager_jit=eager)
        def invalid(value: annotation):
            T.evaluate(0)

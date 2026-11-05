"""Explicit match_buffer parameters use the native tensor ABI and shape scope."""

import ctypes

import numpy as np
import pytest
import torch
from tvm_tirx import ir, tirx

import tilelang
import tilelang.language as T


def _compile(function):
    target = "c"
    return tilelang.compile(function, target=target, target_host=target, execution_backend="tvm_ffi")


def _linear_shape_copy():
    @T.prim_func
    def kernel(a: T.handle, b: T.handle):
        n = T.int64()
        # Match in reverse order: the public argument order remains a, b.
        B = T.match_buffer(b, (n * 32,), "float32")
        A = T.match_buffer(a, (n * 32,), "float32")
        for i in T.serial(n * 32):
            B[i] = A[i] + T.float32(1)

    return kernel


@pytest.mark.parametrize("round_trip", [False, True])
def test_match_buffer_binds_forward_shape_without_new_public_parameters(round_trip):
    function = _linear_shape_copy()
    assert len(function.params) == 2
    assert [parameter.name for parameter in function.params] == ["a", "b"]
    assert all(tirx.is_tensor_var(parameter) for parameter in function.params)
    assert tirx.analysis.check_well_formed(function)
    n = function.params[0].ty.shape[0].a
    assert n.same_as(function.params[1].ty.shape[0].a)
    if round_trip:
        function = ir.load_json(ir.save_json(function))
        assert tirx.analysis.check_well_formed(function)
    kernel = _compile(function)
    for count in [32, 64, 96]:
        source = torch.arange(count, dtype=torch.float32)
        output = torch.empty_like(source)
        kernel(source, output)
        np.testing.assert_array_equal(output.numpy(), source.numpy() + 1)
    with pytest.raises(RuntimeError, match="(shape|binding|divisible)"):
        kernel(torch.empty(33), torch.empty(33))
    with pytest.raises(RuntimeError, match="(shape|binding|conflict)"):
        kernel(torch.empty(32), torch.empty(64))


def test_match_buffer_preserves_later_scalar_and_runtime_argument_order():
    @T.prim_func
    def kernel(a: T.handle, n: T.int64, b: T.handle):
        A = T.match_buffer(a, (n,), "int32")
        B = T.match_buffer(b, (n,), "int32")
        for i in T.serial(n):
            B[i] = A[i] * 3

    assert [parameter.name for parameter in kernel.params] == ["a", "n", "b"]
    assert kernel.params[0].ty.shape[0].same_as(kernel.params[1])
    assert kernel.params[2].ty.shape[0].same_as(kernel.params[1])
    assert tirx.analysis.check_well_formed(kernel)
    compiled = _compile(kernel)
    source = torch.arange(5, dtype=torch.int32)
    output = torch.empty_like(source)
    compiled(source, 5, output)
    np.testing.assert_array_equal(output.numpy(), source.numpy() * 3)
    with pytest.raises(RuntimeError, match="(shape|binding|conflict)"):
        compiled(source, 4, output)


def test_match_buffer_binds_stride_and_byte_offset_from_noncontiguous_tensor():
    @T.prim_func
    def kernel(a: T.handle, b: T.handle):
        n = T.int64()
        stride = T.int64()
        offset = T.int64()
        A = T.match_buffer(a, (n,), "float32", strides=(stride,), elem_offset=offset, align=4)
        B = T.match_buffer(b, (n,), "float32")
        for i in T.serial(n):
            B[i] = A[i]

    assert tirx.analysis.check_well_formed(kernel)
    assert len(kernel.params) == 2
    compiled = _compile(kernel)
    source = torch.arange(19, dtype=torch.float32)[3:15:2]
    output = torch.empty_like(source)
    assert source.stride() == (2,) and source.storage_offset() == 3
    compiled(source, output)
    np.testing.assert_array_equal(output.numpy(), source.numpy())


def test_unmatched_handle_keeps_the_pointer_contract():
    @T.prim_func
    def kernel(pointer: T.handle):
        T.evaluate(pointer)

    assert kernel.params[0].ty == ir.PtrType("void", "global")
    assert kernel.body[0].value.same_as(kernel.params[0])
    assert tirx.analysis.check_well_formed(kernel)


@pytest.mark.parametrize("metadata_use", [False, True])
def test_last_match_selects_the_only_input_contract(metadata_use):
    @T.prim_func
    def kernel(a: T.handle, O: T.Tensor((8,), "int32")):
        unused = T.match_buffer(a, (4,), "float32")
        A = T.match_buffer(a, (8,), "int32")
        for i in T.serial(8):
            O[i] = A[i] + (unused.shape[0] if metadata_use else 1)

    assert len(kernel.params) == 2
    assert str(kernel.params[0].ty.dtype) == "int32"
    assert tirx.analysis.check_well_formed(kernel)
    compiled = _compile(kernel)
    source = torch.arange(8, dtype=torch.int32)
    output = torch.empty_like(source)
    compiled(source, output)
    np.testing.assert_array_equal(output.numpy(), source.numpy() + (4 if metadata_use else 1))
    with pytest.raises(RuntimeError, match="(dtype|int32)"):
        compiled(torch.zeros(8, dtype=torch.float32), output)
    with pytest.raises(RuntimeError, match="(shape|binding)"):
        compiled(torch.zeros(4, dtype=torch.int32), output)


@pytest.mark.parametrize("different_shape", [False, True])
def test_earlier_live_match_requires_explicit_shared_data(different_shape):
    # Old buffer_map kept the last Buffer and left the previous independent
    # data Var undefined, even for two equal tensor types. Report that invalid
    # source before handing an undefined TensorVar to native compiler passes.
    with pytest.raises(ValueError, match="Earlier match_buffer tensor.*no storage definition"):

        @T.prim_func
        def kernel(a: T.handle, O: T.Tensor((1,), "int32")):
            A = T.match_buffer(a, (8,), "int32")
            B = T.match_buffer(a, (4, 2) if different_shape else (8,), "int32")
            O[0] = A[0] + (B[0, 1] if different_shape else B[1])


@pytest.mark.parametrize("round_trip", [False, True])
def test_explicit_shared_data_keeps_both_tensor_views(round_trip):
    @T.prim_func
    def kernel(a: T.handle, O: T.Tensor((1,), "int32")):
        A = T.match_buffer(a, (8,), "int32")
        B = T.match_buffer(a, (4, 2), "int32", data=A.data)
        O[0] = A[1] + B[2, 1]

    assert len(kernel.params) == 2
    assert len(kernel.params[0].ty.shape) == 2
    assert tirx.analysis.check_well_formed(kernel)
    if round_trip:
        kernel = ir.load_json(ir.save_json(kernel))
        assert tirx.analysis.check_well_formed(kernel)
    compiled = _compile(kernel)
    source = torch.arange(8, dtype=torch.int32).reshape(4, 2)
    output = torch.zeros(1, dtype=torch.int32)
    compiled(source, output)
    assert output.item() == 6


@pytest.mark.parametrize("round_trip", [False, True])
def test_raw_handle_after_match_uses_the_original_descriptor(round_trip):
    @T.prim_func
    def kernel(a: T.handle, O: T.Tensor((1,), "int32")):
        A = T.match_buffer(a, (8,), "int32")
        O[0] = T.tvm_struct_get(a, 0, 4, "int32") + T.tvm_struct_get(a, 0, 6, "int32") + A[0]

    assert len(kernel.params) == 2
    assert tirx.analysis.check_well_formed(kernel)
    descriptor = kernel.body[0]
    assert isinstance(descriptor, ir.Bind)
    assert descriptor.value.op.name == "tl.tensor_descriptor_ptr"
    assert descriptor.value.args[0].same_as(kernel.params[0])
    assert isinstance(descriptor.var.ty, ir.PtrType)
    if round_trip:
        kernel = ir.load_json(ir.save_json(kernel))
        assert tirx.analysis.check_well_formed(kernel)
    compiled = _compile(kernel)
    source = torch.arange(8, dtype=torch.int32)
    output = torch.zeros(1, dtype=torch.int32)
    compiled(source, output)
    assert output.item() == 33


def test_descriptor_field_write_is_visible_to_the_next_read():
    @T.prim_func
    def kernel(a: T.handle, O: T.Tensor((2,), "int32")):
        A = T.match_buffer(a, (8,), "int32")
        old_device_id = T.tvm_struct_get(a, 0, 9, "int32")
        T.evaluate(T.tvm_struct_set(a, 0, 9, 7))
        O[0] = T.tvm_struct_get(a, 0, 9, "int32")
        T.evaluate(T.tvm_struct_set(a, 0, 9, old_device_id))
        O[1] = T.tvm_struct_get(a, 0, 9, "int32") + A[0]

    assert tirx.analysis.check_well_formed(kernel)
    compiled = _compile(kernel)
    output = torch.zeros(2, dtype=torch.int32)
    compiled(torch.arange(8, dtype=torch.int32), output)
    assert output.tolist() == [7, 0]


def test_matches_sharing_data_require_the_same_physical_pointer():
    @T.prim_func
    def kernel(a: T.handle, b: T.handle, O: T.Tensor((2,), "int32")):
        A = T.match_buffer(a, (8,), "int32")
        B = T.match_buffer(b, (8,), "int32", data=A.data)
        O[0] = A[1] + B[5]
        O[1] = A[7] - B[6]

    compiled = _compile(kernel)
    source = torch.arange(8, dtype=torch.int32)
    output = torch.empty(2, dtype=torch.int32)
    compiled(source, source, output)
    np.testing.assert_array_equal(output.numpy(), np.array([6, 1], dtype="int32"))
    with pytest.raises((ValueError, RuntimeError), match="(data|pointer|constraint|same)"):
        compiled(source, source.clone(), output)


def test_free_data_variable_is_bound_by_the_final_tensor_parameter():
    pointer = T.handle("int32")

    @T.prim_func
    def kernel(a: T.handle, O: T.Tensor((1,), "int32")):
        A = T.match_buffer(a, (8,), "int32", data=pointer)
        O[0] = A[1] + A[7]

    compiled = _compile(kernel)
    output = torch.empty(1, dtype=torch.int32)
    compiled(torch.arange(8, dtype=torch.int32), output)
    assert output.item() == 8


def test_explicit_data_pointer_parameter_keeps_argument_order():
    @T.prim_func
    def kernel(a: T.handle, data_pointer: T.handle("int32"), O: T.Tensor((1,), "int32")):
        A = T.match_buffer(a, (8,), "int32", data=data_pointer)
        O[0] = A[2]

    compiled = _compile(kernel)
    source = torch.arange(8, dtype=torch.int32)
    output = torch.empty(1, dtype=torch.int32)
    compiled(source, ctypes.c_void_p(source.data_ptr()), output)
    assert output.item() == 2
    other = source.clone()
    with pytest.raises((ValueError, RuntimeError), match="(data|pointer|constraint|same)"):
        compiled(source, ctypes.c_void_p(other.data_ptr()), output)


def test_explicit_shared_data_supports_different_element_types():
    @T.prim_func
    def kernel(a: T.handle, O: T.Tensor((1,), "int32")):
        A = T.match_buffer(a, (2,), "int32")
        B = T.match_buffer(a, (8,), "uint8", data=A.data)
        O[0] = A[1] + T.int32(B[2])

    compiled = _compile(kernel)
    source = torch.arange(8, dtype=torch.uint8)
    output = torch.empty(1, dtype=torch.int32)
    compiled(source, output)
    expected = np.frombuffer(source.numpy().tobytes(), dtype="int32")[1] + np.int32(2)
    assert output.item() == expected

"""Tensor declarations become executable storage constraints on native IR."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

import tilelang.language as T
from tilelang.language._native import buffer
from tilelang.testing.packed import compile_packed_function


def lower(function):
    transform = tvm_ffi.get_global_func("tl.transform.LowerTensorSpec")()
    result = transform(ir.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(result)
    return result


def compile_function(function):
    native = lower(function).with_attr("global_symbol", "main")
    return compile_packed_function(native, tilelang_owned=False)


@pytest.mark.parametrize("element_offset", [0, 4, 8])
def test_offset_factor_checks_and_uses_runtime_view_origin_once(element_offset):
    spec = buffer((7,), "float32", align=4, offset_factor=4)

    @T.prim_func
    def copy(A: spec, B: T.Tensor((7,), "float32")):
        for i in T.serial(7):
            B[i] = A[i] + T.float32(1)

    executable = compile_function(copy)
    values = np.arange(24, dtype="float32")
    storage = tvm_tirx.runtime.tensor(values)
    source = storage._create_view((7,), relative_byte_offset=element_offset * 4)
    output = tvm_tirx.runtime.tensor(np.zeros(7, dtype="float32"))
    executable(source, output)
    np.testing.assert_array_equal(output.numpy(), values[element_offset : element_offset + 7] + 1)


def test_offset_factor_rejects_misalignment_before_writing_output():
    spec = buffer((7,), "float32", align=4, offset_factor=4)

    @T.prim_func
    def copy(A: spec, B: T.Tensor((7,), "float32")):
        for i in T.serial(7):
            B[i] = A[i]

    executable = compile_function(copy)
    storage = tvm_tirx.runtime.tensor(np.arange(24, dtype="float32"))
    source = storage._create_view((7,), relative_byte_offset=4)
    sentinel = np.full(7, -7, dtype="float32")
    output = tvm_tirx.runtime.tensor(sentinel)
    with pytest.raises(ValueError, match="element offset must be divisible by 4"):
        executable(source, output)
    np.testing.assert_array_equal(output.numpy(), sentinel)


@pytest.mark.parametrize("extent", [1, 3, 7])
def test_auto_broadcast_keeps_physical_stride_abi_and_uses_zero_logical_stride(extent):
    n = T.dynamic("n", "int64")
    spec = buffer((n, 1), "float32", align=4, buffer_type="auto")

    @T.prim_func
    def broadcast(A: spec, B: T.Tensor((n, 5), "float32")):
        for i in T.serial(n):
            for j in T.serial(5):
                B[i, j] = A[i, j]

    native = lower(broadcast)
    assert native.params[0].same_as(broadcast.params[0])
    assert all(isinstance(step.stride, ir.Var) for step in native.params[0].ty.layout.shard)
    executable = compile_packed_function(native.with_attr("global_symbol", "main"), tilelang_owned=False)
    # The input has a nonunit outer stride and an irrelevant physical stride
    # on its singleton dimension. Both must be decoded before the logical view.
    values = np.arange(extent * 4, dtype="float32").reshape(extent, 4)[:, 1:2]
    source = tvm_ffi.from_dlpack(values)
    output = tvm_tirx.runtime.tensor(np.zeros((extent, 5), dtype="float32"))
    executable(source, output)
    np.testing.assert_array_equal(output.numpy(), np.repeat(values, 5, axis=1))


def test_constant_invalid_offset_factor_is_a_compile_error():
    spec = buffer((7,), "float32", elem_offset=3, offset_factor=4)

    @T.prim_func
    def invalid(A: spec):
        A[0] = T.float32(1)

    with pytest.raises(ValueError, match="violates offset_factor=4"):
        lower(invalid)


def test_local_declaration_keeps_definition_order_when_consuming_constraint():
    @T.prim_func
    def copy(A: T.Tensor((7,), "float32"), B: T.Tensor((7,), "float32")):
        workspace = T.alloc_buffer((7,), "float32", scope="local", elem_offset=0, offset_factor=4)
        for i in T.serial(7):
            workspace[i] = A[i] + T.float32(2)
        for i in T.serial(7):
            B[i] = workspace[i]

    executable = compile_function(copy)
    values = np.arange(7, dtype="float32")
    source = tvm_tirx.runtime.tensor(values)
    output = tvm_tirx.runtime.tensor(np.zeros(7, dtype="float32"))
    executable(source, output)
    np.testing.assert_array_equal(output.numpy(), values + 2)

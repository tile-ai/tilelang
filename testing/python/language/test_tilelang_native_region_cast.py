"""Fixed vector slices remain loads when consumed by the public Cast facade."""

from tilelang.language.tir import _constructors

import numpy as np
import pytest
import torch
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang
import tilelang.language as T


def test_cast_preserves_native_node_identity():
    value = ir.Var("value", "int32")
    cast = T.Cast("float32", value)
    assert isinstance(cast, ir.prim.Cast)
    assert isinstance(cast, _constructors.Cast)
    assert cast.value.same_as(value)
    assert cast.ty == ir.PrimType("float32")


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_region_cast_keeps_symbolic_base_and_lane_order(index_dtype):
    source = ir.Var("source", tirx.TensorType((16,), "float32"))
    index = ir.Var("index", index_dtype)
    region = source[index * 2 : index * 2 + 2]
    cast = T.Cast("float16x2", region)
    assert isinstance(cast, ir.prim.Cast)
    assert isinstance(cast.value, ir.TensorLoad)
    assert cast.value.source.same_as(source)
    (ramp,) = cast.value.indices
    assert isinstance(ramp, ir.prim.Ramp)
    ir.assert_structural_equal(ramp.base, index * 2)
    assert ramp.stride.ty == index.ty
    assert int(ramp.stride) == 1
    assert int(ramp.lanes) == 2
    assert cast.value.ty == ir.PrimType("float32x2")
    assert cast.ty == ir.PrimType("float16x2")


def test_region_cast_retains_singleton_outer_dimensions():
    source = ir.Var("source", tirx.TensorType((4, 8), "int32"))
    cast = T.Cast("float32x2", source[1:2, 3:5])
    assert cast.value.source.same_as(source)
    row, columns = cast.value.indices
    assert int(row) == 1
    assert isinstance(columns, ir.prim.Ramp)
    assert int(columns.base) == 3 and int(columns.lanes) == 2


def test_singleton_region_cast_is_scalar():
    source = ir.Var("source", tirx.TensorType((4,), "int32"))
    cast = T.Cast("float32", source[2:3])
    assert isinstance(cast.value, ir.TensorLoad)
    assert int(cast.value.indices[0]) == 2
    assert cast.ty == ir.PrimType("float32")


def test_region_cast_rejects_dynamic_lane_count():
    source = ir.Var("source", tirx.TensorType((16,), "int32"))
    size = ir.Var("size", "int32")
    with pytest.raises(ValueError, match="positive constant extent"):
        T.Cast("float32x2", source[0:size])


def test_region_cast_rejects_empty_slice():
    source = ir.Var("source", tirx.TensorType((16,), "int32"))
    with pytest.raises(ValueError, match="positive constant extent"):
        T.Cast("float32x2", source[2:2])


def test_region_cast_rejects_multiple_vector_dimensions():
    source = ir.Var("source", tirx.TensorType((4, 8), "int32"))
    with pytest.raises(ValueError, match="only the last dimension"):
        T.Cast("float32x4", source[0:2, 0:2])


def test_raw_buffer_store_realizes_a_region_as_one_vector_load():
    source = ir.Var("source", tirx.TensorType((8,), "int32"))
    dest = ir.Var("dest", tirx.TensorType((8,), "int32"))
    store = _constructors.BufferStore(dest, source[2:4], [ir.prim.Ramp(4, 1, 2)])
    assert isinstance(store, ir.TensorStore)
    assert isinstance(store.value, ir.TensorLoad)
    assert store.value.source.same_as(source)
    assert store.value.ty == ir.PrimType("int32x2")
    assert int(store.value.indices[0].base) == 2


@pytest.mark.parametrize("strided", [False, True])
def test_region_cast_executes_original_vector_slice_spelling_cpu(strided):
    @T.prim_func
    def kernel(A: T.Tensor((12,), "int32"), B: T.Tensor((4,), "float32")):
        alias = T.decl_buffer((4,), "int32", data=A.data, strides=(2 if strided else 1,), elem_offset=1, align=4)
        temporary = T.alloc_local((2,), "float32")
        for i in T.serial(2):
            temporary[0:2] = T.Cast("float32x2", alias[i * 2 : i * 2 + 2])
            B[i * 2 : i * 2 + 2] = temporary[0:2]

    assert tirx.analysis.check_well_formed(kernel)
    reconstructed = ir.load_json(ir.save_json(kernel))
    ir.assert_structural_equal(kernel, reconstructed, map_free_vars=True)
    target = "c"
    executable = tilelang.compile(reconstructed, target=target, target_host=target, execution_backend="tvm_ffi")
    values = torch.tensor([-91, 256, -32769, 17, 0, 8388607, -8388609, 42, 1, 2, 3, 4], dtype=torch.int32)
    output = torch.empty((4,), dtype=torch.float32)
    executable(values, output)
    indices = [1, 3, 5, 7] if strided else [1, 2, 3, 4]
    np.testing.assert_array_equal(output.numpy(), values[indices].float().numpy())


def test_thread_return_is_one_native_void_control_flow_call():
    call = T.thread_return()
    assert isinstance(call, ir.Call)
    assert call.op.name == "tirx.gpu_thread_return"
    assert call.ty == ir.PrimType("void")
    assert len(call.args) == 0 and len(call.ty_args) == 0
    call.validate()

    @T.prim_func
    def kernel(A: T.Tensor((64,), "int32")):
        with T.Kernel(1, threads=64):
            tx = T.get_thread_binding()
            if tx < 32:
                T.thread_return()
            A[tx] = tx

    calls = []
    tvm_ffi.structural_walk(kernel.body, (ir.Call, calls.append))
    assert len([call for call in calls if call.op.name == "tirx.gpu_thread_return"]) == 1
    assert tirx.analysis.check_well_formed(kernel)

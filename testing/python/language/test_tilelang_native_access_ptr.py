"""Pointer contracts retain tensor coordinates, element intervals and scope."""

import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx

import tilelang.language as T
from tilelang.language import _native
from tilelang.utils.language import retrieve_ptr, retrive_ptr_from_buffer_region


def _tensor(name="a", *, shape=(4, 8), strides=None, elem_offset=0, scope="shared"):
    spec = _native.buffer(shape, "float32", strides=strides, elem_offset=elem_offset, scope=scope)
    return ir.Var(name, spec.as_type())


def _constant(expr):
    result = sym.Analyzer().simplify(expr)
    assert isinstance(result, ir.IntImm)
    return int(result.value)


def test_tensor_access_keeps_logical_indices_and_layout_offset():
    tensor = _tensor(strides=(16, 1), elem_offset=5)
    pointer = T.access_ptr(ir.TensorLoad(tensor, [1, 2]), "rw", 2, 3)
    pointer.validate()
    assert pointer.op.name == "tl.access_ptr"
    assert not pointer.ty_args
    assert isinstance(pointer.ty, ir.PtrType)
    assert str(pointer.ty.element_type.dtype) == "float32"
    assert str(pointer.ty.scope) == "shared"
    load, extent, mask = pointer.args
    assert load.source.same_as(tensor)
    assert [_constant(index) for index in load.indices] == [1, 2]
    assert _constant(_native.tensor_offset(load.source)) == 5
    assert [_constant(x) for x in _native.tensor_strides(load.source)] == [16, 1]
    assert _constant(extent) == 6 and int(mask.value) == 3


def test_pointer_helpers_keep_previous_remaining_storage_extent():
    tensor = _tensor(strides=(16, 1), elem_offset=5)
    region = ir.TensorRegion(tensor, [ir.Range.from_min_extent(1, 2), ir.Range.from_min_extent(2, 3)], ir.TensorRegionType())
    pointer = retrieve_ptr(region, "r")
    assert [_constant(index) for index in pointer.args[0].indices] == [1, 2]
    assert _constant(pointer.args[1]) == 64 - 18
    assert tvm_ffi.structural_equal(pointer, retrive_ptr_from_buffer_region(region, "r"))
    leading = retrieve_ptr(region, "w", ignore_last_ndim=1)
    assert [_constant(index) for index in leading.args[0].indices] == [1, 0]
    assert _constant(leading.args[1]) == 64 - 16
    assert int(leading.args[2].value) == 2


def test_raw_pointer_access_has_element_units_and_preserves_storage_scope():
    tensor = _tensor(strides=(16, 1), elem_offset=5)
    pointer = T.tvm_access_ptr(T.type_annotation("float32"), tensor.data, 7, 9, 1)
    pointer.validate()
    assert pointer.op.name == "tl.ptr_access"
    assert not pointer.ty_args
    assert str(pointer.ty.scope) == "shared"
    assert str(pointer.ty.element_type.dtype) == "float32"
    base, offset, extent, mask = pointer.args
    assert base.op.name == "tirx.tensor_data_ptr" and base.args[0].same_as(tensor)
    assert [_constant(x) for x in (offset, extent, mask)] == [7, 9, 1]


def test_raw_pointer_dtype_override_reinterprets_before_element_offset():
    pointer = ir.Var("p", ir.PtrType(ir.PrimType("float32"), "shared"))
    result = T.tvm_access_ptr(T.type_annotation("uint8"), pointer, 3, 12, 3)
    result.validate()
    base = result.args[0]
    assert base.op.name == "tirx.reinterpret"
    assert base.args[0].same_as(pointer)
    assert str(base.ty.element_type.dtype) == "uint8"
    assert str(base.ty.scope) == "shared"
    assert _constant(result.args[1]) == 3
    assert _constant(result.args[2]) == 12


@pytest.mark.parametrize("mask", [-1, 4])
def test_pointer_mask_rejects_invalid_access_contract(mask):
    tensor = _tensor()
    with pytest.raises(ValueError, match="mask"):
        T.access_ptr(tensor, mask)
    with pytest.raises(ValueError, match="mask"):
        T.tvm_access_ptr(T.type_annotation("float32"), tensor.data, 0, 1, mask)


def test_raw_pointer_interval_rejects_negative_extent_and_untyped_input():
    tensor = _tensor()
    with pytest.raises(ValueError, match="negative"):
        T.tvm_access_ptr(T.type_annotation("float32"), tensor.data, 0, -1, 1)
    with pytest.raises(TypeError, match="typed pointer"):
        T.tvm_access_ptr(T.type_annotation("float32"), ir.const(123, "uint64"), 0, 1, 1)


def test_pointer_calls_are_well_formed_explicit_tensor_dependencies():
    @T.prim_func
    def kernel(a: T.Tensor((8,), "float32")):
        T.evaluate(T.access_ptr(a[2], "r", 4))
        T.evaluate(T.tvm_access_ptr(T.type_annotation("float32"), a.data, 3, 2, 1))

    assert tirx.analysis.check_well_formed(kernel)
    calls = []
    tvm_ffi.structural_walk(kernel, (ir.Call, calls.append), order="pre")
    pointer_calls = [call for call in calls if call.op.name in {"tl.access_ptr", "tl.ptr_access"}]
    assert len(pointer_calls) == 2
    for call in pointer_calls:
        call.validate()

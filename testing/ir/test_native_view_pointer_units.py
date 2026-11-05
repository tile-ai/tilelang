"""Native layout lowering preserves the address units and identities of views."""

import ctypes

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from tvm_tirx.tirx.layout import S

import tilelang.transform as tltr
from tilelang.testing.packed import compile_packed_function


def _tensor(name, dtype, shape=(64,), strides=None, origin=0):
    layout = "default" if strides is None else tirx.TileLayout(S[shape:strides] + origin)
    return ir.Var(name, tirx.TensorType(shape, dtype, layout=layout))


def _declare(tensor, pointer):
    return ir.Bind(tensor, ir.Call("tirx.decl_tensor", [pointer], ty_args=[tensor.ty]))


def _calls(value, name):
    result = []
    tvm_ffi.structural_walk(
        value, (ir.Call, lambda call: result.append(call) if isinstance(call.op, ir.Op) and call.op.name == name else None), order="pre"
    )
    return result


def _function(params, body, pointer):
    return tirx.Function(params, [*body, ir.Return(pointer)], pointer.ty).with_attr(
        {
            "global_symbol": "main",
            "target": tvm_tirx.target.Target("c"),
        }
    )


def _lower(function):
    assert tirx.analysis.check_well_formed(function)
    module = tirx.transform.LowerTensorLayouts()(ir.IRModule({"main": function}))
    assert tirx.analysis.check_well_formed(module)
    # The source type carries the offset unit. The enclosing pointer result
    # must agree even after a view has been resolved to another allocation.
    for call in _calls(module, "tl.ptr_access"):
        ir.assert_structural_equal(call.args[0].ty, call.ty)
    module = tirx.transform.FlattenTensor()(module)
    module = tltr.LowerAccessPtr()(module)
    module = ir.load_json(ir.save_json(module))
    assert tirx.analysis.check_well_formed(module)
    return module


def _compile(module):
    return compile_packed_function(module, tilelang_owned=False)


def _address(value):
    return (value.value or 0) if isinstance(value, ctypes.c_void_p) else int(value or 0)


@pytest.mark.parametrize(
    "storage_dtype,view_dtype,element_bytes",
    [
        ("float32", "bfloat16", 2),
        ("bfloat16", "float32", 4),
        ("float32", "float32", 4),
    ],
)
@pytest.mark.parametrize("depth", [1, 3])
def test_view_pointer_offsets_use_the_projected_element_width(storage_dtype, view_dtype, element_bytes, depth):
    pointer = ir.Var("pointer", ir.PtrType(storage_dtype))
    offset = ir.Var("offset", "int64")
    storage_bytes = tvm_ffi.dtype(storage_dtype).bits // 8
    storage = _tensor("storage", storage_dtype, shape=(65536 // storage_bytes,))
    body = [_declare(storage, pointer)]
    current = storage
    for index in range(depth):
        alias = _tensor(f"alias_{index}", view_dtype, shape=(65536 // element_bytes,))
        body.append(_declare(alias, current.data))
        current = alias
    address = ir.Call("tl.ptr_access", [current.data, offset, ir.IntImm("int32", 1), ir.IntImm("int32", 1)])
    compiled = _compile(_lower(_function([pointer, offset], body, address)))
    backing = np.zeros(65536, dtype="uint8")
    start = backing.ctypes.data
    for element_offset in (0, 1, 4096):
        assert _address(compiled(ctypes.c_void_p(start), element_offset)) == start + element_offset * element_bytes


@pytest.mark.parametrize(
    "storage_dtype,view_dtype,element_bytes",
    [
        ("float32", "bfloat16", 2),
        ("bfloat16", "float32", 4),
    ],
)
@pytest.mark.parametrize("byte_offset", [0, 32])
@pytest.mark.parametrize("strides,origin", [((7, 1), 3), ((7, -1), 3)])
def test_strided_view_address_keeps_its_byte_offset_and_origin(storage_dtype, view_dtype, element_bytes, byte_offset, strides, origin):
    pointer = ir.Var("pointer", ir.PtrType(storage_dtype))
    storage = _tensor("storage", storage_dtype)
    base = storage.data
    if byte_offset:
        base = tirx.ptr_byte_offset(base, byte_offset, ty=base.ty)
    view = _tensor("view", view_dtype, shape=(2, 3), strides=strides, origin=origin)
    alias = _tensor("alias", view_dtype, shape=(2, 3), strides=strides, origin=origin)
    body = [_declare(storage, pointer), _declare(view, base), _declare(alias, view.data)]
    address = tirx.address_of(ir.TensorLoad(alias, [1, 2]))
    compiled = _compile(_lower(_function([pointer], body, address)))
    backing = np.zeros(256, dtype="uint8")
    start = backing.ctypes.data
    expected = start + byte_offset + (origin + strides[0] + 2 * strides[1]) * element_bytes
    assert _address(compiled(ctypes.c_void_p(start))) == expected


@pytest.mark.parametrize("selected", [0, 1])
def test_equal_names_do_not_merge_distinct_storage_identities(selected):
    pointers = [ir.Var("pointer", ir.PtrType("float32")) for _ in range(2)]
    tensors = [_tensor("storage", "float32") for _ in range(2)]
    views = [_tensor("view", "bfloat16") for _ in range(2)]
    body = []
    for pointer, tensor, view in zip(pointers, tensors, views):
        body.extend([_declare(tensor, pointer), _declare(view, tensor.data)])
    result = ir.Call("tl.ptr_access", [views[selected].data, ir.IntImm("int32", 3), ir.IntImm("int32", 1), ir.IntImm("int32", 1)])
    compiled = _compile(_lower(_function(pointers, body, result)))
    backing = [np.zeros(256, dtype="uint8") for _ in range(2)]
    addresses = [array.ctypes.data for array in backing]
    actual = _address(compiled(*(ctypes.c_void_p(address) for address in addresses)))
    assert actual == addresses[selected] + 6
    assert actual != addresses[1 - selected] + 6


def test_pointer_projection_rejects_an_unbound_tensor():
    tensor = _tensor("undefined_storage", "bfloat16")
    function = _function([], [], tensor.data)
    with pytest.raises(RuntimeError, match="no visible definition"):
        tirx.transform.LowerTensorLayouts()(ir.IRModule({"main": function}))

"""Input descriptor identity survives native IR, flattening, and packed lowering."""

import ctypes

import numpy as np
import pytest
import tvm_tirx as tvm
from tvm_tirx import ir, tirx
from tvm_tirx.tirx.layout import S

import tilelang
from tilelang.language import _native
from tilelang.testing.packed import compile_packed_function
from tilelang.testing.ir import collect_calls, collect_nodes


class DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class DLTensor(ctypes.Structure):
    # DLPack C ABI. This caller-owned object makes a reconstructed descriptor
    # observably different, including nullable strides and field mutation.
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


def _descriptor(shape, strides=None, offset=0):
    # TensorType's default data alignment is 64 bytes. NumPy only guarantees
    # dtype alignment, so explicitly provide an aligned caller-owned base.
    storage = np.empty(128 + 16, dtype="int32")
    start = (-storage.ctypes.data % 64) // storage.itemsize
    data = storage[start : start + 128]
    data[:] = np.arange(128, dtype="int32")
    shape_array = (ctypes.c_int64 * len(shape))(*shape)
    strides_array = None if strides is None else (ctypes.c_int64 * len(shape))(*strides)
    descriptor = DLTensor(
        data.ctypes.data, DLDevice(1, 0), len(shape), DLDataType(0, 32, 1), shape_array, strides_array, offset * data.itemsize
    )
    descriptor.owners = (data, shape_array, strides_array)
    return descriptor


def _tensor(name, shape, strides=None, offset=0):
    layout = "default" if strides is None else tirx.TileLayout(S[shape:strides] + offset)
    return ir.Var(name, tirx.TensorType(shape, "int32", layout=layout))


def _function(params, body, ret_type="int32", target="c"):
    if isinstance(ret_type, str):
        ret_type = ir.PrimType(ret_type)
    return tirx.Function(params, body, ret_type).with_attr(
        {
            "global_symbol": "main",
            "target": tvm.target.Target(target, host="c"),
        }
    )


def _compile(function):
    assert tirx.analysis.check_well_formed(function)
    return compile_packed_function(function)


def _pointer(value):
    return (value.value or 0) if isinstance(value, ctypes.c_void_p) else int(value or 0)


@pytest.mark.parametrize(
    "shape,strides,offset",
    [
        ((2, 3), None, 0),
        ((2, 3), (5, 1), 2),
        ((2, 3), (5, -1), 2),
        ((), None, 0),
    ],
)
@pytest.mark.parametrize("round_trip", [False, True])
def test_borrowed_descriptor_is_the_callers_exact_pointer(shape, strides, offset, round_trip):
    tensor = _tensor("input", shape, strides, offset)
    function = _function([tensor], ir.Return(_native.tensor_descriptor_ptr(tensor)), ir.PtrType("void"))
    if round_trip:
        function = tirx.renew_def(ir.load_json(ir.save_json(function)))
    compiled = _compile(function)
    for _ in range(2):
        descriptor = _descriptor(shape, strides, offset)
        pointer = ctypes.c_void_p(ctypes.addressof(descriptor))
        assert _pointer(compiled(pointer)) == pointer.value


@pytest.mark.parametrize(
    "field,ty,expected", [(4, "int32", 2), (6, "int32", 32), (7, "int32", 1), (8, "uint64", 8), (9, "int32", 0), (10, "int32", 1)]
)
def test_original_descriptor_fields_are_observed(field, ty, expected):
    tensor = _tensor("input", (2, 3), (5, 1), 2)
    result = tirx.abi_field_get(_native.tensor_descriptor_ptr(tensor), 0, field, ty=ty)
    compiled = _compile(_function([tensor], ir.Return(result), ty))
    descriptor = _descriptor((2, 3), (5, 1), 2)
    assert compiled(ctypes.c_void_p(ctypes.addressof(descriptor))) == expected


@pytest.mark.parametrize("strides", [None, (3, 1)])
@pytest.mark.parametrize("field", [2, 3])
def test_shape_and_nullable_stride_pointer_identity(strides, field):
    tensor = _tensor("input", (2, 3), strides)
    pointer = tirx.abi_field_get(_native.tensor_descriptor_ptr(tensor), 0, field, ty=ir.PtrType("int64"))
    compiled = _compile(_function([tensor], ir.Return(pointer), ir.PtrType("int64")))
    descriptor = _descriptor((2, 3), strides)
    expected = ctypes.cast(descriptor.shape if field == 2 else descriptor.strides, ctypes.c_void_p).value or 0
    assert _pointer(compiled(ctypes.c_void_p(ctypes.addressof(descriptor)))) == expected


def test_descriptor_field_write_is_visible_to_caller_and_following_read():
    tensor = _tensor("input", (2, 3))
    descriptor_pointer = _native.tensor_descriptor_ptr(tensor)
    previous = ir.Var("previous_ndim", "int32")
    function = _function(
        [tensor],
        [
            ir.Bind(previous, tirx.abi_field_get(descriptor_pointer, 0, 4, ty="int32")),
            ir.Evaluate(tirx.abi_field_set(descriptor_pointer, 0, 4, ir.const(7, "int32"), ty="int32")),
            ir.Return(previous * 10 + tirx.abi_field_get(descriptor_pointer, 0, 4, ty="int32")),
        ],
    )
    compiled = _compile(function)
    descriptor = _descriptor((2, 3))
    assert compiled(ctypes.c_void_p(ctypes.addressof(descriptor))) == 27
    assert descriptor.ndim == 7


def test_later_scalar_shape_and_flattened_data_use_keep_parameter_identity():
    extent = ir.Var("extent", "int64")
    tensor = _tensor("input", (extent, 3))
    ndim = tirx.abi_field_get(_native.tensor_descriptor_ptr(tensor), 0, 4, ty="int32")
    function = _function([tensor, extent], ir.Return(tensor[1, 2] + ndim))
    module = tilelang.transform.LowerTensorDescriptors()(ir.IRModule.from_expr(function))
    module = tirx.transform.FlattenTensor()(module)
    flattened = module["main"]
    (getter,) = collect_calls(flattened, op="tl.input_tensor_descriptor")
    assert int(getter.args[0].value) == 0
    assert flattened.params[0].same_as(tensor)
    assert not collect_nodes(flattened, ir.TensorLoad)[0].source.same_as(flattened.params[0])
    assert tirx.analysis.check_well_formed(module)
    compiled = _compile(function)
    descriptor = _descriptor((2, 3))
    assert compiled(ctypes.c_void_p(ctypes.addressof(descriptor)), 2) == 7
    with pytest.raises(RuntimeError, match="shape|extent"):
        compiled(ctypes.c_void_p(ctypes.addressof(descriptor)), 3)


@pytest.mark.parametrize("target", ["llvm", "cuda"])
@pytest.mark.parametrize("round_trip", [False, True])
def test_inlined_getters_are_pinned_on_host_before_device_outlining(target, round_trip):
    tensor = _tensor("input", (2, 3))
    getter = _native.tensor_descriptor_ptr(tensor)
    region = ir.RegionStmt(
        "tirx.device_scope",
        [],
        [],
        {"target": tvm.target.Target(target)},
        [
            ir.Evaluate(tirx.call_extern("opaque_descriptor_consumer", getter, getter, ty="int32")),
        ],
    )
    function = _function([tensor], region, "void", target)
    if round_trip:
        function = tirx.renew_def(ir.load_json(ir.save_json(function)))
    outlined = tilelang.transform.OutlineDeviceRegions()(ir.IRModule.from_expr(function))
    assert tirx.analysis.check_well_formed(outlined)
    host = outlined["main"]
    (getter,) = collect_calls(host, op="tl.input_tensor_descriptor")
    assert int(getter.args[0].value) == 0
    assert tirx.is_tensor_var(host.params[0])
    devices = [value for key, value in outlined.functions.items() if key.name_hint != "main"]
    assert len(devices) == 1
    device = devices[0]
    assert not collect_calls(device, op="tl.input_tensor_descriptor")
    assert len(device.params) == 1 and isinstance(device.params[0].ty, ir.PtrType)
    (consumer,) = collect_calls(device, op="tirx.call_extern")
    assert consumer.args[1].same_as(device.params[0]) and consumer.args[2].same_as(device.params[0])
    packed = tilelang.transform.MakePackedAPI()(outlined)
    assert not collect_calls(packed, op="tl.input_tensor_descriptor")
    assert tirx.analysis.check_well_formed(packed)


@pytest.mark.parametrize("kind", ["allocation", "view"])
@pytest.mark.parametrize("boundary", ["packed", "outline", "flatten", "layout"])
def test_local_tensors_do_not_acquire_a_fabricated_descriptor(kind, boundary):
    source = _tensor("input", (2, 3))
    local = _tensor("local_tensor", (2, 3))
    value = (
        ir.Call("tirx.alloc_tensor", [], ty_args=[local.ty])
        if kind == "allocation"
        else ir.Call("tirx.decl_tensor", [source.data], ty_args=[local.ty])
    )
    function = _function([source], [ir.Bind(local, value), ir.Return(_native.tensor_descriptor_ptr(local))], ir.PtrType("void"))
    transform = {
        "packed": tilelang.transform.MakePackedAPI,
        "outline": tilelang.transform.OutlineDeviceRegions,
        "flatten": tirx.transform.FlattenTensor,
        "layout": tirx.transform.LowerTensorLayouts,
    }[boundary]
    with pytest.raises(ValueError, match="original packed tensor parameter"):
        module = tilelang.transform.LowerTensorDescriptors()(ir.IRModule.from_expr(function))
        transform()(module)


@pytest.mark.parametrize("argument", [lambda: ir.Var("pointer", ir.PtrType("void")), lambda: ir.const(3, "int32")])
def test_descriptor_getter_rejects_non_tensor_operands(argument):
    with pytest.raises((TypeError, ValueError)):
        call = _native.tensor_descriptor_ptr(argument())
        call.validate()


def test_descriptor_slot_is_remapped_when_an_earlier_parameter_is_pruned():
    # Names cannot identify inputs: these two parameters intentionally share one.
    unused = _tensor("input", (2, 3))
    used = _tensor("input", (2, 3))
    function = _function([unused, used], ir.Return(_native.tensor_descriptor_ptr(used)), ir.PtrType("void"))
    module = tilelang.transform.LowerTensorDescriptors()(ir.IRModule.from_expr(function))
    module = ir.load_json(ir.save_json(module))
    module = tilelang.transform.Simplify(simplify_arguments=True)(module)
    pruned = module["main"]
    assert len(pruned.params) == 1
    (getter,) = collect_calls(pruned, op="tl.input_tensor_descriptor")
    assert int(getter.args[0].value) == 0
    assert tirx.analysis.check_well_formed(module)
    compiled = _compile(pruned)
    descriptor = _descriptor((2, 3))
    address = ctypes.addressof(descriptor)
    assert _pointer(compiled(ctypes.c_void_p(address))) == address


def test_descriptor_only_parameter_is_required_even_without_data_accesses():
    tensor = _tensor("input", (2, 3))
    ndim = tirx.abi_field_get(_native.tensor_descriptor_ptr(tensor), 0, 4, ty="int32")
    compiled = _compile(_function([tensor], ir.Return(ndim)))
    # The Python FFI normalizes both forms of null to None. The required tensor
    # type check must reject both, even though the body only reads metadata.
    with pytest.raises(RuntimeError, match="expected Tensor"):
        compiled(None)
    with pytest.raises(RuntimeError, match="expected Tensor"):
        compiled(ctypes.c_void_p())
    descriptor = _descriptor((2, 3))
    assert compiled(ctypes.c_void_p(ctypes.addressof(descriptor))) == 2


@pytest.mark.parametrize("index", [-1, 1, 2])
def test_lowered_descriptor_slot_rejects_invalid_or_pointer_parameters(index):
    tensor = _tensor("input", (2, 3))
    pointer = ir.Var("pointer", ir.PtrType("void"))
    getter = ir.Call("tl.input_tensor_descriptor", [ir.const(index, "int32")])
    function = _function([tensor, pointer], ir.Return(getter), ir.PtrType("void"))
    with pytest.raises((ValueError, TypeError), match="argument index|tensor parameter"):
        tilelang.transform.MakePackedAPI()(ir.IRModule.from_expr(function))

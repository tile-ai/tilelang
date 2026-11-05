"""Physical byte carriers retain logical sub-byte shape, strides, and offsets."""

import ctypes
from functools import lru_cache

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.tirx.layout import S

from tilelang.testing.packed import compile_packed_function


class _Device(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int32), ("index", ctypes.c_int32)]


class _DType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _Device),
        ("ndim", ctypes.c_int32),
        ("dtype", _DType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


def _descriptor(shape, dtype, strides=None, byte_offset=0):
    # The probe only reads descriptor metadata. The aligned storage keeps the
    # ordinary native data-pointer contract valid, including for empty shapes.
    backing = np.zeros(256, dtype="uint8")
    data = backing.ctypes.data + (-backing.ctypes.data % 64)
    dimensions = (ctypes.c_int64 * len(shape))(*shape)
    steps = None if strides is None else (ctypes.c_int64 * len(shape))(*strides)
    dtype = tvm_ffi.dtype(dtype)
    result = _DLTensor(data, _Device(1, 0), len(shape), _DType(dtype.type_code, dtype.bits, dtype.lanes), dimensions, steps, byte_offset)
    result.owners = (backing, dimensions, steps)
    return result


def _invoke(compiled, descriptor, *arguments):
    return compiled(ctypes.c_void_p(ctypes.addressof(descriptor)), *arguments)


@lru_cache(None)
def _compile(dtype="float4_e2m1fn", policy=True, layout="compact", index_type="int64", later=False):
    extent = ir.Var("extent", index_type)
    stride, offset = ir.Var("stride", index_type), ir.Var("offset", index_type)
    shape = [ir.const(2, index_type), extent]
    if layout == "scalar":
        shape = []
    elif layout == "fixed_odd":
        shape = [ir.const(2, index_type), ir.const(5, index_type)]
    memory = "default"
    if layout in ("strided", "offset"):
        memory = tirx.TileLayout(S[shape : [stride, 1]] + (offset if layout == "offset" else 0))
    tensor = ir.Var("input", tirx.TensorType(shape, dtype, layout=memory))
    result = extent
    if layout in ("scalar", "fixed_odd"):
        result = ir.const(7, index_type)
    elif layout == "strided":
        result = extent * 1000 + stride
    elif layout == "offset":
        result = offset
    params = [tensor, extent] if later else [tensor]
    function = tirx.Function(params, ir.Return(result), ir.PrimType(index_type))
    function = function.with_attr("global_symbol", "main")
    assert tirx.analysis.check_well_formed(function)
    return compile_packed_function(function, tilelang_owned=policy)


@pytest.mark.parametrize("logical", ["float4_e2m1fn", "int4", "uint4", "int1", "uint2"])
@pytest.mark.parametrize("storage", ["float4_e2m1fnx2", "uint8", "int8", "float8_e4m3fn"])
def test_byte_storage_expands_only_the_innermost_shape(logical, storage):
    factor = 8 // tvm_ffi.dtype(logical).bits
    assert _invoke(_compile(logical), _descriptor((2, 5), storage, (5, 1))) == 5 * factor


@pytest.mark.parametrize("storage", ["float4_e2m1fnx2", "uint8", "int8"])
def test_default_native_fp4_stays_exact(storage):
    with pytest.raises(TypeError, match="dtype"):
        _invoke(_compile(policy=False), _descriptor((2, 5), storage))


@pytest.mark.parametrize("policy", [False, True])
def test_exact_native_scalar_dtype_does_not_expand_shape(policy):
    assert _invoke(_compile(policy=policy), _descriptor((2, 10), "float4_e2m1fn", (10, 1))) == 10


@pytest.mark.parametrize("strides,expected", [(None, 10010), ((8, 1), 10016), ((-8, 1), 9984)])
def test_explicit_outer_stride_uses_logical_elements(strides, expected):
    assert _invoke(_compile(layout="strided"), _descriptor((2, 5), "uint8", strides)) == expected


def test_later_scalar_must_agree_with_packed_shape():
    compiled = _compile(later=True)
    descriptor = _descriptor((2, 5), "uint8")
    assert _invoke(compiled, descriptor, 10) == 10
    with pytest.raises(RuntimeError, match="extent|shape"):
        _invoke(compiled, descriptor, 5)


@pytest.mark.parametrize("layout", ["compact", "strided"])
def test_nonunit_packed_inner_stride_is_rejected(layout):
    with pytest.raises(RuntimeError, match="stride"):
        _invoke(_compile(layout=layout), _descriptor((2, 5), "uint8", (20, 2)))


@pytest.mark.parametrize("strides", [None, (9, 3)])
def test_empty_compact_packed_tensor_has_no_access_stride_requirement(strides):
    assert _invoke(_compile(), _descriptor((2, 0), "uint8", strides)) == 0


@pytest.mark.parametrize("storage", ["uint16", "float4_e2m1fnx4", "bool"])
def test_nonbyte_carriers_are_not_numeric_conversions(storage):
    with pytest.raises(RuntimeError, match="dtype"):
        _invoke(_compile(), _descriptor((2, 5), storage))


def test_bool_does_not_enable_packing():
    with pytest.raises(RuntimeError, match="dtype"):
        _invoke(_compile("bool"), _descriptor((2, 5), "uint8"))


def test_scalar_fp4_has_no_packing_axis():
    compiled = _compile(layout="scalar")
    assert _invoke(compiled, _descriptor((), "float4_e2m1fn")) == 7
    with pytest.raises(RuntimeError, match="dtype"):
        _invoke(compiled, _descriptor((), "float4_e2m1fnx2"))


def test_odd_logical_extent_cannot_be_supplied_by_whole_bytes():
    with pytest.raises(RuntimeError, match="shape"):
        _invoke(_compile(layout="fixed_odd"), _descriptor((2, 3), "uint8"))


@pytest.mark.parametrize("index_type,physical", [("int32", 2**30), ("int64", 2**62), ("int64", -1)])
def test_shape_expansion_must_not_overflow(index_type, physical):
    with pytest.raises(RuntimeError, match="index range"):
        _invoke(_compile(index_type=index_type), _descriptor((2, physical), "uint8"))


@pytest.mark.parametrize("offset", [0, 1, 3])
def test_byte_offsets_are_scaled_to_logical_elements(offset):
    assert _invoke(_compile(layout="offset"), _descriptor((2, 5), "uint8", (8, 1), offset)) == 2 * offset


def test_byte_offset_expansion_must_not_overflow():
    with pytest.raises(RuntimeError, match="byte offset.*index range"):
        _invoke(_compile(layout="offset"), _descriptor((2, 5), "uint8", (8, 1), 2**62))


def test_nonpacked_tensor_metadata_contract_is_unchanged():
    descriptor = _descriptor((2, 5), "float32", (5, 1))
    assert _invoke(_compile("float32", policy=True), descriptor) == 5
    assert _invoke(_compile("float32", policy=False), descriptor) == 5


@pytest.mark.parametrize("stride", [2**63 - 1, -(2**63)])
def test_empty_compact_tensor_does_not_constrain_unused_stride(stride):
    # No physical or logical element is addressed; compactness ignores all strides.
    descriptor = _descriptor((2, 0), "uint8", (stride, 3))
    assert _invoke(_compile(), descriptor) == 0


@pytest.mark.parametrize("stride", [2**62 - 1, -(2**62)])
def test_explicit_stride_scale_keeps_in_range_extremes(stride):
    descriptor = _descriptor((2, 0), "uint8", (stride, 1))
    # An explicit strided annotation is a metadata contract even for empty tensors.
    assert _invoke(_compile(layout="strided"), descriptor) == 2 * stride


@pytest.mark.parametrize("stride", [2**62, -(2**62) - 1])
def test_explicit_stride_scaling_rejects_overflow(stride):
    descriptor = _descriptor((2, 0), "uint8", (stride, 1))
    with pytest.raises(RuntimeError, match="index range"):
        _invoke(_compile(layout="strided"), descriptor)


@pytest.mark.parametrize("stride", [2**63 - 1, -(2**63)])
def test_singleton_compact_axis_does_not_constrain_unused_stride(stride):
    extent = ir.Var("extent", "int64")
    tensor = ir.Var("input", tirx.TensorType([1, extent], "float4_e2m1fn"))
    function = tirx.Function([tensor], ir.Return(extent), ir.PrimType("int64"))
    function = function.with_attr("global_symbol", "main")
    compiled = compile_packed_function(function)
    descriptor = _descriptor((1, 5), "uint8", (stride, 1))
    assert _invoke(compiled, descriptor) == 10

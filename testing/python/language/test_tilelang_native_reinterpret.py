"""Keep TileLang's reinterpret spelling while constructing native typed bitcasts."""

import inspect

import numpy as np
import pytest
import torch
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang
import tilelang.language as T
import tilelang.testing


@pytest.mark.parametrize("dtype", ["uint32", T.uint32, tvm_ffi.dtype("uint32"), ir.PrimType("uint32"), torch.uint32, np.uint32])
def test_reinterpret_accepts_public_dtype_values(dtype):
    value = ir.Var("value", "float32")
    result = T.reinterpret(value, dtype)
    assert isinstance(result, ir.Call)
    assert result.op.name == "tirx.reinterpret"
    assert result.ty == ir.PrimType("uint32")
    assert result.args[0].same_as(value)


@pytest.mark.parametrize("dtype", ["uint32", T.uint32, ir.PrimType("uint32"), torch.uint32])
def test_legacy_dtype_first_spelling_keeps_deprecation_and_meaning(dtype):
    value = ir.Var("value", "float32")
    with pytest.warns(DeprecationWarning, match="T.reinterpret"):
        result = T.reinterpret(dtype, value)
    assert result.ty == ir.PrimType("uint32")
    assert result.args[0].same_as(value)


def test_python_literal_and_vector_values_use_native_bitcast_types():
    result = T.reinterpret(1.0, "uint32")
    assert result.args[0].ty == ir.PrimType("float32")
    assert float(result.args[0].value) == 1.0
    vector = ir.Var("v", "float32x2")
    converted = T.reinterpret(vector, T.uint32x2)
    assert converted.ty == ir.PrimType("uint32x2")
    assert converted.args[0].same_as(vector)


def test_handle_and_explicit_pointer_type_keep_native_address_contract():
    null = T.reinterpret(T.uint64(0), dtype="handle")
    assert null.ty == ir.PtrType("void")
    shared = ir.Var("shared", ir.PtrType("float32", "shared"))
    target = ir.PtrType("uint32", "shared")
    cast = T.reinterpret(shared, target)
    assert cast.ty.same_as(target)
    assert cast.args[0].same_as(shared)
    assert T.reinterpret(shared, T.uint64).ty == ir.PrimType("uint64")


@pytest.mark.parametrize("location_kind", ["source", "callsite", "unknown"])
def test_explicit_span_is_preserved_as_native_location(location_kind):
    loc = ir.SourceLoc(ir.SourceName("reinterpret_user.py"), 17, 4, 17, 40)
    if location_kind == "callsite":
        loc = ir.CallSiteLoc(loc, ir.SourceLoc(ir.SourceName("caller.py"), 9, 0, 9, 15))
    elif location_kind == "unknown":
        loc = ir.UnknownLoc()
    result = T.reinterpret(ir.Var("value", "float32"), "uint32", span=loc)
    assert result.loc.same_as(loc)
    restored = ir.load_json(ir.save_json(result))
    assert tvm_ffi.structural_equal(restored.loc, loc)


def test_omitted_span_uses_current_user_source_location():
    def source(A: T.Tensor((4,), "float32"), B: T.Tensor((4,), "uint32")):
        for i in T.serial(4):
            B[i] = T.reinterpret(A[i], T.uint32)

    function = T.prim_func(source)
    calls = []
    tvm_ffi.structural_walk(function, (ir.Call, calls.append))
    bitcasts = [call for call in calls if isinstance(call.op, ir.Op) and call.op.name == "tirx.reinterpret"]
    assert len(bitcasts) == 1
    loc = bitcasts[0].loc
    _, first_line = inspect.getsourcelines(source)
    assert isinstance(loc, ir.SourceLoc)
    assert loc.start_line == first_line + 2
    assert str(loc.source_name.name).endswith("test_tilelang_native_reinterpret.py")


def test_invalid_pointer_bitcasts_keep_native_type_errors():
    pointer = ir.Var("pointer", ir.PtrType("float32"))
    with pytest.raises(TypeError, match="64-bit integer target"):
        T.reinterpret(pointer, "uint32")
    with pytest.raises(TypeError, match="64-bit integer source"):
        T.reinterpret(T.int32(0), "handle")
    with pytest.raises(TypeError, match="PrimType or PtrType"):
        T.reinterpret(ir.Var("tensor", tirx.TensorType((4,), "float32")), "uint32")


# The oracle is the raw IEEE-754 payload, including signed zero, subnormal,
# infinity and NaN values. No floating arithmetic or tolerance is involved.
_BITS = np.array(
    [
        0x00000000,
        0x80000000,
        0x3F800000,
        0xBF800000,
        0x00800000,
        0x00000001,
        0x7F800000,
        0xFF800000,
        0x7FC01234,
        0x7FA00001,
        0x3F000000,
        0x3F800001,
    ],
    dtype=np.uint32,
)


@tilelang.testing.requires_llvm
@pytest.mark.parametrize("direction", ["float_to_bits", "bits_to_float"])
def test_llvm_executes_exact_payload_reinterpretation(direction):
    size = len(_BITS)

    if direction == "float_to_bits":

        @T.prim_func
        def convert(A: T.Tensor((size,), "float32"), B: T.Tensor((size,), "uint32")):
            for i in T.serial(size):
                B[i] = T.reinterpret(A[i], T.uint32)
    else:

        @T.prim_func
        def convert(A: T.Tensor((size,), "uint32"), B: T.Tensor((size,), "float32")):
            for i in T.serial(size):
                B[i] = T.reinterpret(A[i], T.float32)

    assert tirx.analysis.check_well_formed(convert)
    target = {"kind": "llvm", "mcpu": "generic"}
    compiled = tilelang.compile(convert, target=target, target_host=target, execution_backend="tvm_ffi")
    storage = torch.from_numpy(_BITS.copy()).clone()
    if direction == "float_to_bits":
        argument = storage.view(torch.float32)
        output = torch.empty(len(_BITS), dtype=torch.uint32)
    else:
        argument = storage
        output = torch.empty(len(_BITS), dtype=torch.float32)
    compiled(argument, output)
    actual = output.numpy().view(np.uint32)
    np.testing.assert_array_equal(actual, _BITS)

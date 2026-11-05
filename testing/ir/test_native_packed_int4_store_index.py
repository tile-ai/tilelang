"""Native packed stores consume proven contiguous lanes without losing casts."""

import re

import pytest
import tilelang  # noqa: F401 -- registers the TileLang CUDA build callback
import tvm_ffi
from tvm_tirx import ir, prim, tirx
from tvm_tirx.target import Target


def _shuffle(values, dtype="int32"):
    return prim.Shuffle([ir.const(value, dtype) for value in values], list(range(len(values))))


def _source(index, dtype="int4", *, params=(), declarations=(), read=False):
    lanes = index.ty.dtype.lanes
    pointer = ir.Var("output", ir.PtrType(ir.PrimType(dtype)))
    tensor = ir.Var("view", tirx.TensorType((512,), dtype, layout="default"))
    value = ir.Var("value", f"{dtype}x{lanes}")
    body = [
        ir.Bind(tensor, ir.Call("tirx.decl_tensor", [pointer], ty_args=[tensor.ty])),
        *declarations,
        ir.Evaluate(ir.TensorLoad(tensor, [index])) if read else ir.TensorStore(tensor, [index], value),
    ]
    function = tirx.Function([pointer, value, *params], body)
    function = function.with_attr("global_symbol", "packed_store")
    function = function.with_attr("calling_conv", ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    assert tirx.analysis.check_well_formed(function)
    assert not tirx.analysis.undefined_vars(function)
    calls = []
    tvm_ffi.structural_walk(function, (ir.Call, calls.append))
    for call in calls:
        call.validate()
    module = ir.IRModule({"packed_store": function})
    original = ir.save_json(module)
    try:
        build = tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")
        return build(module, Target({"kind": "cuda", "arch": "sm_90"})).inspect_source()
    finally:
        assert ir.save_json(module) == original


def _assert_carrier_store(source, dtype, lanes, base, index_dtype="int32"):
    carrier = {
        "int4": {2: "int8_t", 4: "int16_t", 8: "int", 16: "int2", 32: "int4"},
        "uint4": {2: "uint8_t", 4: "uint16_t", 8: "uint", 16: "uint2", 32: "uint4"},
    }[dtype][lanes]
    cast = {"int32": "", "int64": "(int64_t)", "int8": "(signed char)"}[index_dtype]
    assert f"*((({carrier}*)output) + {cast}{base // lanes}) = value;" in source
    assert f"tl_{dtype}_packed_store(" not in source


@pytest.mark.parametrize("dtype", ["int4", "uint4"])
@pytest.mark.parametrize("lanes", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("form", ["shuffle", "widen", "narrow_in_range", "bound_broadcast"])
def test_literal_lanes_keep_the_same_logical_address_and_carrier(dtype, lanes, form):
    # Original lane addresses are exactly [lanes, ..., 2*lanes-1]. Every
    # intermediate value fits even for the explicit int8 narrowing cases.
    values = list(range(lanes, lanes * 2))
    index = _shuffle(values)
    declarations = []
    if form == "widen":
        index = prim.Cast(f"int64x{lanes}", index)
    elif form == "narrow_in_range":
        index = prim.Cast(f"int8x{lanes}", index)
    elif form == "bound_broadcast":
        zero, one = ir.Var("zero", "int64"), ir.Var("one", "int64")
        declarations = [ir.Bind(zero, ir.const(0, "int64")), ir.Bind(one, ir.const(1, "int64"))]
        index = prim.Broadcast(zero, lanes) + prim.Cast(f"int64x{lanes}", index) * prim.Broadcast(one, lanes)
    # Only after proving that every narrow intermediate is in range, the
    # native address Ramp uses int32. The input expression remains unchanged;
    # overflowing and unproved casts are rejected by the cases below.
    index_dtype = {"shuffle": "int32", "widen": "int64", "narrow_in_range": "int32", "bound_broadcast": "int64"}[form]
    _assert_carrier_store(_source(index, dtype, declarations=declarations), dtype, lanes, lanes, index_dtype)


@pytest.mark.parametrize("dtype", ["int4", "uint4"])
@pytest.mark.parametrize("source_dtype,base", [("int8", 124), ("uint8", 252), ("int32", 2147483644), ("uint32", 4294967292)])
def test_widening_preserves_unsigned_values_and_last_in_range_lanes(dtype, source_dtype, base):
    values = list(range(base, base + 4))
    index = prim.Cast("int64x4", _shuffle(values, source_dtype))
    # These inspect generated addresses only; large base values are not GPU
    # execution inputs, and no out-of-bounds memory execution is claimed.
    _assert_carrier_store(_source(index, dtype), dtype, 4, base, "int64")


@pytest.mark.parametrize("dtype", ["int4", "uint4"])
def test_symbolic_uniform_base_keeps_the_existing_alignment_proof(dtype):
    base = ir.Var("base", "int64")
    index = prim.Broadcast(base * 4, 4) + prim.Cast("int64x4", _shuffle([0, 1, 2, 3]))
    source = _source(index, dtype, params=[base])
    assert "base" in source
    assert f"tl_{dtype}_packed_store(" not in source
    assert " = value;" in source


@pytest.mark.parametrize("dtype", ["int4", "uint4"])
@pytest.mark.parametrize(
    "case",
    ["odd_base", "stride_two", "repeated", "reverse", "non_affine", "narrow_wrap", "signedness_wrap", "add_overflow", "unsigned_underflow"],
)
def test_unsafe_lane_transactions_retain_the_original_rejection(dtype, case):
    if case in ("odd_base", "stride_two", "repeated", "reverse", "non_affine"):
        values = {
            "odd_base": [1, 2, 3, 4],
            "stride_two": [0, 2, 4, 6],
            "repeated": [0, 0, 0, 0],
            "reverse": [3, 2, 1, 0],
            "non_affine": [0, 1, 3, 4],
        }[case]
        index = _shuffle(values)
        assert values[0] % 4 != 0 or values != list(range(values[0], values[0] + 4))
    elif case == "narrow_wrap":
        index = prim.Cast("uint8x4", _shuffle([254, 255, 256, 257]))
        assert [value % 256 for value in [254, 255, 256, 257]] == [254, 255, 0, 1]
    elif case == "signedness_wrap":
        index = prim.Cast("int8x4", _shuffle([126, 127, 128, 129], "uint8"))
        assert [((value + 128) % 256) - 128 for value in [126, 127, 128, 129]] == [126, 127, -128, -127]
    elif case == "add_overflow":
        index = prim.Add(_shuffle([124, 125, 126, 127], "int8"), prim.Broadcast(ir.const(1, "int8"), 4))
        index = prim.Cast("int64x4", index)
        assert [((value + 1 + 128) % 256) - 128 for value in [124, 125, 126, 127]] == [125, 126, 127, -128]
    else:
        index = prim.Sub(_shuffle([0, 1, 2, 3], "uint8"), prim.Broadcast(ir.const(1, "uint8"), 4))
        index = prim.Cast("int64x4", index)
        assert [(value - 1) % 256 for value in [0, 1, 2, 3]] == [255, 0, 1, 2]
    with pytest.raises(RuntimeError, match="provably divisible by the vector lane count"):
        _source(index, dtype)


@pytest.mark.parametrize("dtype", ["int4", "uint4"])
@pytest.mark.parametrize("target_type", ["int64x4", "int8x4"])
def test_unproved_vector_cast_is_not_replaced_by_casting_only_its_base(dtype, target_type):
    base = ir.Var("base", "int32")
    index = prim.Cast(target_type, prim.Ramp(base, 1, 4))
    # For int64, base=INT32_MAX-1 crosses a signed source boundary. For
    # int8, base=126 wraps while narrowing. Unknown runtime bases cannot be
    # accepted just by casting the first lane and assuming all others follow.
    with pytest.raises(RuntimeError, match="provably divisible by the vector lane count"):
        _source(index, dtype, params=[base])


@pytest.mark.parametrize("dtype", ["int4", "uint4"])
def test_effectful_address_expression_is_not_normalized(dtype):
    base = tirx.call_extern("next_base", ty="int64")
    index = prim.Broadcast(base * 4, 4) + prim.Cast("int64x4", _shuffle([0, 1, 2, 3]))
    with pytest.raises(RuntimeError, match="provably divisible by the vector lane count"):
        _source(index, dtype)


@pytest.mark.parametrize("dtype", ["int4", "uint4"])
@pytest.mark.parametrize("lanes", [8, 16, 32])
@pytest.mark.parametrize("form", ["widened_shuffle", "bound_broadcast"])
def test_wide_gather_recovers_each_logical_lane_without_a_wide_index_carrier(dtype, lanes, form):
    # A source index carrier int64x8/16/32 has no CUDA vector spelling. The
    # memory consumer can still emit these in-bounds scalar nibble addresses.
    values = [1 + 3 * i for i in range(lanes)]
    index = prim.Cast(f"int64x{lanes}", _shuffle(values))
    declarations = []
    if form == "bound_broadcast":
        zero, one = ir.Var("zero", "int64"), ir.Var("one", "int64")
        declarations = [ir.Bind(zero, ir.const(0, "int64")), ir.Bind(one, ir.const(1, "int64"))]
        index = prim.Broadcast(zero, lanes) + index * prim.Broadcast(one, lanes)
    source = _source(index, dtype, declarations=declarations, read=True)
    qualifier = "unsigned" if dtype == "uint4" else "signed"
    addresses = re.findall(
        rf"tl_{dtype}_packed_load\(\(const {qualifier} char\*\)output, \(int64_t\)([0-9]+)\)",
        source,
    )
    assert [int(value) for value in addresses] == values, source
    assert f"make_longlong{lanes}(" not in source

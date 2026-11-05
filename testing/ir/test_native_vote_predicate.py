"""Predicate normalization preserves native types and rejects numeric loss."""

import pytest
from tvm_tirx import ir

import tilelang.language as T
from tilelang.language.builtin import _validate_predicate


@pytest.mark.parametrize("value,dtype", [(False, "bool"), (True, "bool"), (-1, "int32"), (0, "int32"), (1, "int32")])
def test_predicate_literals_preserve_kind_and_value(value, dtype):
    predicate = _validate_predicate(value, "any_sync")
    assert isinstance(predicate, ir.IntImm)
    assert predicate.ty.dtype == dtype
    assert predicate.value == value


@pytest.mark.parametrize("dtype", ["bool", "int8", "uint32", "int64", "int32x4", "uint8x8", "boolx4"])
def test_predicate_native_type_and_lanes_are_not_coerced(dtype):
    # This is the normalization contract; it does not assert device support
    # for a vector argument to an unvectorized collective.
    predicate = ir.Var("predicate", ir.PrimType(dtype))
    normalized = _validate_predicate(predicate, "any_sync")
    assert normalized.same_as(predicate)
    assert normalized.ty.dtype == dtype


@pytest.mark.parametrize(
    "value",
    [0.5, 1.0, ir.FloatImm("float16", 0.5), ir.Var("pointer", ir.PtrType("int32")), "predicate"],
)
def test_predicate_non_integer_inputs_are_rejected(value):
    with pytest.raises(TypeError, match="requires an integer or boolean predicate"):
        T.any_sync(value)


_MASK = ir.IntImm("uint32", 0xFFFFFFFF)
_PREDICATE = ir.IntImm("int32", 1)
_COLLECTIVES = [
    ("any_sync", [_MASK, _PREDICATE], "int32"),
    ("all_sync", [_MASK, _PREDICATE], "int32"),
    ("ballot_sync", [_MASK, _PREDICATE], "uint64"),
    ("ballot", [_PREDICATE], "uint64"),
    ("activemask", [], "uint64"),
    ("syncthreads_count", [_PREDICATE], "int32"),
    ("syncthreads_and", [_PREDICATE], "int32"),
    ("syncthreads_or", [_PREDICATE], "int32"),
    ("match_any_sync", [_MASK, _PREDICATE], "uint32"),
    ("match_all_sync", [_MASK, _PREDICATE], "uint32"),
]


@pytest.mark.parametrize("name,args,dtype", _COLLECTIVES, ids=[entry[0] for entry in _COLLECTIVES])
@pytest.mark.parametrize("type_source", ["operation", "argument", "explicit"])
def test_collective_result_types_survive_native_reinference(name, args, dtype, type_source):
    kwargs = {}
    if type_source == "argument":
        kwargs["ty_args"] = [ir.PrimType(dtype)]
    elif type_source == "explicit":
        kwargs["ty"] = ir.PrimType(dtype)
    call = ir.Call("tl." + name, args, **kwargs)
    assert call.ty.dtype == dtype
    assert ir.reinfer_type(call).dtype == dtype
    call.validate()


@pytest.mark.parametrize("kwargs", [{"ty": "float32"}, {"ty_args": [ir.PrimType("float32")]}])
def test_collective_rejects_a_conflicting_result_type(kwargs):
    call = ir.Call("tl.ballot_sync", [_MASK, _PREDICATE], **kwargs)
    with pytest.raises(TypeError, match="result type|result.*fixed"):
        call.validate()


@pytest.mark.parametrize("args", [[_MASK], [_MASK, _PREDICATE, _PREDICATE]])
def test_collective_rejects_wrong_operand_count(args):
    call = ir.Call("tl.any_sync", args)
    with pytest.raises(ValueError, match="runtime operands"):
        call.validate()


@pytest.mark.parametrize(
    "mask",
    [ir.const(0.5), ir.IntImm("uint64", 0xFFFFFFFF), ir.Var("mask", ir.PrimType("uint32x4"))],
)
def test_collective_rejects_invalid_mask_type(mask):
    call = ir.Call("tl.any_sync", [mask, _PREDICATE])
    with pytest.raises(TypeError, match="mask must be a 32-bit integer|must be scalar"):
        call.validate()


@pytest.mark.parametrize("predicate", [ir.const(0.5), ir.Var("predicate", ir.PrimType("int32x4"))])
def test_collective_rejects_lossy_or_unscalarized_predicate(predicate):
    call = ir.Call("tl.any_sync", [_MASK, predicate])
    with pytest.raises(TypeError, match="integer or boolean predicate|must be scalar"):
        call.validate()

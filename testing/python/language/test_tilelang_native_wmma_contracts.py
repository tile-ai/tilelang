"""WMMA validates primitive type values after native object reconstruction."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang.language as T


def _call():
    fragment = ir.Var("fragment", ir.PtrType("float16", "wmma.matrix_a"))
    pointer = ir.Var("pointer", ir.PtrType("float16"))
    stride = ir.Var("stride", "int32")
    return T.tvm_load_matrix_sync(fragment, 16, 16, 16, 0, pointer, stride, "row_major"), [fragment, pointer, stride]


def _roundtrip(call, params, mode):
    function = tirx.Function(params, [ir.Evaluate(call)], ir.NoneType())
    if mode == "json":
        function = ir.load_json(ir.save_json(function))
    elif mode == "renew":
        function = tirx.renew_def(function)
    calls = []
    tvm_ffi.structural_walk(function, (ir.Call, calls.append))
    assert len(calls) == 1
    return function, calls[0]


@pytest.mark.parametrize("mode", ["direct", "json", "renew"])
def test_wmma_void_result_uses_value_equality(mode):
    original, params = _call()
    function, call = _roundtrip(original, params, mode)
    assert call.ty == ir.PrimType("void")
    assert not call.ty_args
    assert call.op.name == "tl.wmma_load_matrix_sync"
    assert int(call.op.get_attr("TCallEffectKind")) == int(ir.CallEffectKind.Opaque)
    call.validate()
    assert tirx.analysis.check_well_formed(function)
    if mode == "json":
        # Deserialization does not promise the cached PrimType singleton.
        assert not call.ty.same_as(ir.PrimType("void"))
    if mode == "renew":
        assert all(not old.same_as(new) for old, new in zip(params, function.params, strict=True))


@pytest.mark.parametrize("mode", ["direct", "json", "renew"])
@pytest.mark.parametrize("result_type", [ir.PrimType("int32"), ir.PtrType("void")])
def test_wmma_rejects_nonvoid_results_after_reconstruction(mode, result_type):
    original, params = _call()
    invalid = ir.Call(original.op, original.args, ty=result_type)
    _, call = _roundtrip(invalid, params, mode)
    with pytest.raises(TypeError, match="WMMA load has a fixed void result"):
        call.validate()


@pytest.mark.parametrize("mode", ["direct", "json", "renew"])
def test_wmma_rejects_type_arguments_after_reconstruction(mode):
    original, params = _call()
    invalid = ir.Call(original.op, original.args, ty=ir.PrimType("void"), ty_args=[ir.PrimType("void")])
    _, call = _roundtrip(invalid, params, mode)
    with pytest.raises(TypeError, match="WMMA load has a fixed void result"):
        call.validate()

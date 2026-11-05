"""Tensor address lowering preserves observable parameter shape/stride/offset fields."""

import ctypes
from functools import lru_cache

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target
from tvm_tirx.tirx.layout import S

import tilelang
import tilelang.language as T
from tilelang.testing.packed import compile_packed_function
from test_native_packed_subbyte_abi import _descriptor, _invoke


def _tensor_type(shape, strides=None, offset=0):
    shape = [ir.const(value, "int64") for value in shape]
    layout = None if strides is None else tirx.TileLayout(S[shape:strides] + offset)
    return tirx.TensorType(shape, "int32", align=1, layout=layout)


@pytest.mark.parametrize("pattern", ["compact", "unit_symbol", "unit_constant", "strided", "offset"])
@pytest.mark.parametrize("roundtrip", ["direct", "json", "renew"])
def test_lowering_restores_parameter_definitions_types_and_metadata(pattern, roundtrip):
    stride = ir.Var("stride", "int64")
    shape = (2, 7) if pattern in ("strided", "offset") else (1, 7)
    strides = None if pattern == "compact" else [3 if pattern == "unit_constant" else stride, 1]
    offset = ir.Var("offset", "int64") if pattern == "offset" else 0
    ty = _tensor_type(shape, strides, offset)
    # Same spelling must never merge two parameter identities.
    src, dst = ir.Var("data", ty), ir.Var("data", ty)
    function = tirx.Function([src, dst], [ir.TensorStore(dst, [0, 0], ir.TensorLoad(src, [0, 0]) + 1)])
    function = function.with_attr("target", Target("c")).with_attr("global_symbol", "main")
    if roundtrip == "json":
        function = ir.load_json(ir.save_json(function))
    elif roundtrip == "renew":
        function = tirx.renew_def(function)
    function = function.with_attr("review.parameter", function.params[0])
    module = ir.IRModule({"main": function}, attrs={"review.parameter": function.params[0]})
    before = ir.save_json(module)
    result = tilelang.transform.LowerTensorLayouts()(module)
    lowered = result["main"]
    assert ir.save_json(module) == before
    assert not lowered.params[0].same_as(lowered.params[1])
    for index, original in enumerate(function.params):
        assert lowered.params[index].same_as(original)
        assert lowered.ty.arg_types[index].same_as(original.ty)
    assert lowered.attrs["review.parameter"].same_as(lowered.params[0])
    assert result.attrs["review.parameter"].same_as(lowered.params[0])
    assert tirx.analysis.check_well_formed(result)
    flattened = tirx.transform.FlattenTensor()(result)
    assert tirx.analysis.check_well_formed(flattened)
    for original, parameter in zip(function.params, flattened["main"].params):
        assert parameter.same_as(original)
    restored = ir.load_json(ir.save_json(flattened))
    # The module attribute references a variable outside the Function's
    # definition region. Verify that graph sharing explicitly, and compare
    # the closed Function where parameter definitions bind those references.
    assert restored.attrs["review.parameter"].same_as(restored["main"].params[0])
    ir.assert_structural_equal(flattened["main"], restored["main"])
    ir.assert_structural_equal(restored["main"], flattened["main"])
    assert tvm_ffi.structural_hash(flattened["main"]) == tvm_ffi.structural_hash(restored["main"])


@lru_cache(None)
def _singleton_reader(*, constant=False, cross_carrier=False, explicit_scalar=False):
    stride = ir.const(3, "int64") if constant else ir.Var("stride", "int64")
    first = ir.Var("first", _tensor_type((1, 7), [stride, 1]))
    params = [first]
    if cross_carrier:
        params.append(ir.Var("second", first.ty))
    if explicit_scalar:
        params.append(stride)
    function = tirx.Function(params, ir.Return(stride), ir.PrimType("int64"))
    return compile_packed_function(function.with_attr("global_symbol", "main"))


@pytest.mark.parametrize("stride", [0, 1, 3, 8, -17, 2**33 + 7])
def test_singleton_dynamic_stride_is_still_read_from_the_descriptor(stride):
    assert _invoke(_singleton_reader(), _descriptor((1, 7), "int32", (stride, 1))) == stride


def test_singleton_null_strides_use_dense_descriptor_default():
    assert _invoke(_singleton_reader(), _descriptor((1, 7), "int32")) == 7


def test_explicit_constant_stride_remains_an_argument_constraint():
    execute = _singleton_reader(constant=True)
    assert _invoke(execute, _descriptor((1, 7), "int32", (3, 1))) == 3
    with pytest.raises(RuntimeError, match="strides"):
        _invoke(execute, _descriptor((1, 7), "int32", (5, 1)))


def test_shared_stride_checks_every_descriptor_carrier():
    execute = _singleton_reader(cross_carrier=True)
    first = _descriptor((1, 7), "int32", (3, 1))
    same = _descriptor((1, 7), "int32", (3, 1))
    different = _descriptor((1, 7), "int32", (5, 1))
    assert _invoke(execute, first, ctypes.c_void_p(ctypes.addressof(same))) == 3
    with pytest.raises(RuntimeError, match="strides"):
        _invoke(execute, first, ctypes.c_void_p(ctypes.addressof(different)))


def test_explicit_stride_scalar_is_checked_against_descriptor():
    execute = _singleton_reader(explicit_scalar=True)
    first = _descriptor((1, 7), "int32", (5, 1))
    assert _invoke(execute, first, 5) == 5
    with pytest.raises(RuntimeError, match="strides"):
        _invoke(execute, first, 7)


@lru_cache(None)
def _address_reader():
    stride, offset = ir.Var("stride", "int64"), ir.Var("offset", "int64")
    source = ir.Var("input", _tensor_type((2, 3), [stride, 1], offset))
    value = ir.prim.Cast("int64", ir.TensorLoad(source, [1, 2])) + offset * 1000 + stride * 100000
    function = tirx.Function([source], ir.Return(value), ir.PrimType("int64"))
    return compile_packed_function(function.with_attr("global_symbol", "main"))


@pytest.mark.parametrize("stride,origin", [(3, 0), (5, 2), (11, 3)])
def test_restored_layout_does_not_reapply_stride_or_byte_offset(stride, origin):
    descriptor = _descriptor((2, 3), "int32", (stride, 1), origin * 4)
    backing = np.ctypeslib.as_array((ctypes.c_int32 * 40).from_address(descriptor.data))
    backing[:] = np.arange(40) * 13 + 2
    expected = int(backing[origin + stride + 2]) + origin * 1000 + stride * 100000
    assert _invoke(_address_reader(), descriptor) == expected


@lru_cache(None)
def _compact_reader():
    source = ir.Var("input", _tensor_type((1, 7)))
    function = tirx.Function([source], ir.Return(ir.TensorLoad(source, [0, 6])), ir.PrimType("int32"))
    return compile_packed_function(function.with_attr("global_symbol", "main"))


@pytest.mark.parametrize("stride", [-(2**60), 1, 200])
def test_implicit_compact_singleton_axis_keeps_ignored_stride(stride):
    descriptor = _descriptor((1, 7), "int32", (stride, 1))
    backing = np.ctypeslib.as_array((ctypes.c_int32 * 10).from_address(descriptor.data))
    backing[:] = np.arange(10) * 13 + 2
    assert _invoke(_compact_reader(), descriptor) == 80


def test_native_declarations_keep_original_object_and_signature():
    tensor = ir.Var("A", _tensor_type((1, 7), [ir.Var("stride", "int64"), 1]))
    function = tirx.Function([tensor], [], attrs=ir.DictAttrs({"declaration": True}))
    result = tilelang.transform.LowerTensorLayouts()(ir.IRModule({"decl": function}))["decl"]
    assert result.same_as(function)


@pytest.mark.parametrize("shape,dense,other", [((1, 7), (7, 1), (5, 1)), ((2, 0), (0, 1), (9, 3))])
@pytest.mark.parametrize("explicit", [False, True])
def test_source_stride_declaration_survives_default_address_equivalence(shape, dense, other, explicit):
    annotation = T.StridedTensor(shape, dense, "int32") if explicit else T.Tensor(shape, "int32")

    @T.prim_func
    def main(source: annotation):
        T.func_ret(ir.PrimType("int32"))
        T.ret(37)

    before = ir.save_json(main)
    module = ir.IRModule({"main": main.with_attr("target", Target("c"))})
    lowered = tilelang.transform.LowerTensorSpec()(module)
    flags = lowered["main"].attrs.get("tl.explicit_stride_params")
    if explicit:
        assert list(flags) == [True]
        assert all(isinstance(flag, bool) for flag in flags)
    else:
        assert flags is None
    lowered = tilelang.transform.LowerTensorLayouts()(lowered)
    assert lowered["main"].attrs.get("tl.explicit_stride_params") is None
    assert (lowered["main"].params[0].ty.layout is not None) == explicit
    assert tirx.analysis.check_well_formed(lowered)
    assert ir.save_json(main) == before

    execute = compile_packed_function(main)
    assert _invoke(execute, _descriptor(shape, "int32", dense)) == 37
    if explicit:
        with pytest.raises(RuntimeError, match="strides"):
            _invoke(execute, _descriptor(shape, "int32", other))
    else:
        assert _invoke(execute, _descriptor(shape, "int32", other)) == 37


@pytest.mark.parametrize("shape,strides", [((1, 7), (-17, 1)), ((2, 0), (9, 3))])
def test_native_default_layout_keeps_existing_compact_metadata_contract(shape, strides):
    tensor = ir.Var("input", tirx.TensorType(shape, "int32", align=1, layout="default"))
    function = tirx.Function([tensor], ir.Return(ir.const(17)), ir.PrimType("int32"))
    function = function.with_attr("global_symbol", "main")
    execute = compile_packed_function(function)
    assert _invoke(execute, _descriptor(shape, "int32", strides)) == 17


@pytest.mark.parametrize("roundtrip", ["direct", "json", "renew"])
def test_source_stride_flags_keep_parameter_positions_and_are_consumed(roundtrip):
    @T.prim_func
    def main(value: T.int32, source: T.StridedTensor((1, 7), (7, 1), "int32"), unused: T.int32):
        T.func_ret(ir.PrimType("int32"))
        T.ret(value)

    module = tilelang.transform.LowerTensorSpec()(ir.IRModule({"main": main.with_attr("target", Target("c"))}))
    assert list(module["main"].attrs["tl.explicit_stride_params"]) == [False, True, False]
    if roundtrip == "json":
        module = ir.load_json(ir.save_json(module))
    elif roundtrip == "renew":
        module = ir.IRModule({"main": tirx.renew_def(module["main"])})
    before = ir.save_json(module)
    result = tilelang.transform.LowerTensorLayouts()(module)
    assert ir.save_json(module) == before
    assert result["main"].attrs.get("tl.explicit_stride_params") is None
    assert result["main"].params[1].ty.layout is not None
    for before, after in zip(module["main"].params, result["main"].params):
        assert after.same_as(before)
        assert after.loc.same_as(before.loc)
    assert result["main"].loc.same_as(module["main"].loc)
    assert tirx.analysis.check_well_formed(result)


def test_stale_stride_metadata_does_not_silently_change_another_parameter():
    function = tirx.Function([ir.Var("value", "int32")], []).with_attr("target", Target("c"))
    function = function.with_attr("tl.explicit_stride_params", [False, True])
    with pytest.raises(ValueError, match="must match the signature"):
        tilelang.transform.LowerTensorLayouts()(ir.IRModule({"main": function}))

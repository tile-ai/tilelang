"""Typed substitutions of the legacy eager helper preserve a native signature."""

import numpy as np
import pytest
import torch
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.tirx.layout import Iter

import tilelang
import tilelang.language as T
from tilelang.language.eager.builder import substitute_primfunc
from tilelang.language._tensor_spec import tensor_offset
from tilelang.language._tensor_spec import tensor_strides
from tilelang.testing.packed import compile_packed_function


def _nodes(value, kind):
    result = []
    tvm_ffi.structural_walk(value, (kind, result.append), order="pre")
    return result


def _copy(n, *, layout=None):
    a = ir.Var("A", tirx.TensorType([n], "int32", layout=layout, align=4))
    b = ir.Var("B", tirx.TensorType([n], "int32", align=4))
    i = ir.Var("i", "int32")
    body = ir.For(i, 0, n, 1, ir.ForKind.DEFAULT, [ir.TensorStore(b, [i], ir.TensorLoad(a, [i]) + 2)])
    return tirx.Function([a, b], [body], attrs=ir.make_node("ir.DictAttrs", global_symbol="copy", tilelang_out_idx=[1]))


@pytest.mark.parametrize("size", [0, 1, 7])
def test_shape_substitution_updates_type_uses_without_mutating_input(size):
    n = ir.Var("n", "int32")
    function = _copy(n)
    original_hash = tvm_ffi.structural_hash(function)
    result = substitute_primfunc(function, {n: size})
    assert tirx.analysis.check_well_formed(result)
    assert len(result.params) == 2
    assert all(int(param.ty.shape[0]) == size for param in result.params)
    assert all(not var.same_as(n) for var in _nodes(result, ir.Var))
    assert len(result.ty.arg_types) == 2
    assert all(tvm_ffi.structural_equal(actual, param.ty) for actual, param in zip(result.ty.arg_types, result.params))
    assert list(result.attrs["tilelang_out_idx"]) == [1]
    assert tvm_ffi.structural_hash(function) == original_hash
    assert function.params[0].ty.shape[0].same_as(n)
    inputs = torch.arange(size, dtype=torch.int32)
    outputs = torch.full_like(inputs, -100)
    compile_packed_function(result, tilelang_owned=False)(inputs, outputs)
    torch.testing.assert_close(outputs, inputs + 2, rtol=0, atol=0)


def test_scalar_parameter_rename_keeps_arity_return_type_and_location():
    value = ir.Var("value", "int32")
    replacement = ir.Var("renamed", "int32")
    function = tirx.Function(
        [value], [ir.Return(value + 3)], ret_type=ir.PrimType("int32"), attrs=ir.make_node("ir.DictAttrs", global_symbol="add")
    )
    result = substitute_primfunc(function, {value: replacement})
    assert result.params[0].same_as(replacement)
    assert len(result.params) == len(function.params)
    assert result.ret_type == function.ret_type
    assert result.loc.same_as(function.loc)
    assert tirx.analysis.check_well_formed(result)
    assert compile_packed_function(result, tilelang_owned=False)(6) == 9


def test_tensor_rename_and_shape_substitution_share_one_identity():
    n = ir.Var("n", "int32")
    function = _copy(n)
    original = function.params[0]
    replacement = ir.Var("renamed", original.ty)
    result = substitute_primfunc(function, {original: replacement, n: 6})
    a = result.params[0]
    assert a.name == "renamed"
    assert int(a.ty.shape[0]) == 6
    assert all(load.source.same_as(a) for load in _nodes(result, ir.TensorLoad))
    assert replacement.ty.shape[0].same_as(n)
    assert tirx.analysis.check_well_formed(result)
    inp, out = torch.arange(6, dtype=torch.int32), torch.empty(6, dtype=torch.int32)
    compile_packed_function(result, tilelang_owned=False)(inp, out)
    torch.testing.assert_close(out, inp + 2, rtol=0, atol=0)


def test_non_contiguous_input_preserves_the_byte_offset_contract():
    n, stride, offset = (ir.Var(name, "int32") for name in ("n", "stride", "offset"))
    function = _copy(n, layout=tirx.TileLayout.from_iters([Iter(n, stride, "m")], offset={"m": offset}))
    result = substitute_primfunc(function, {n: 5, stride: 2, offset: 3})
    assert [int(x) for x in tensor_strides(result.params[0])] == [2]
    assert int(tensor_offset(result.params[0])) == 3
    assert tirx.analysis.check_well_formed(result)
    storage = torch.arange(20, dtype=torch.int32)
    inp = storage[3:13:2]
    out = torch.empty(5, dtype=torch.int32)
    assert inp.storage_offset() == 3 and not inp.is_contiguous()
    # Torch exports this slice with an already shifted data pointer and a zero
    # DLPack byte_offset, which must not satisfy an explicit 12-byte contract.
    with pytest.raises(ValueError, match=r"A.byte_offset"):
        compile_packed_function(result, tilelang_owned=False)(inp, out)
    zero_origin = substitute_primfunc(function, {n: 5, stride: 2, offset: 0})
    compile_packed_function(zero_origin, tilelang_owned=False)(inp, out)
    torch.testing.assert_close(out, inp + 2, rtol=0, atol=0)


def test_local_tensor_view_rewrites_origin_stride_type_arguments_and_uses():
    n, stride, offset = (ir.Var(name, "int32") for name in ("n", "stride", "offset"))
    size = n * 2 + 10
    source = ir.Var(
        "source",
        tirx.TensorType([size], "int32", align=4, layout=tirx.TileLayout.from_iters([Iter(size, stride, "m")], offset={"m": offset})),
    )
    output = ir.Var("output", tirx.TensorType([n], "int32", align=4))
    view = ir.Var(
        "view",
        tirx.TensorType([n], "int32", align=4, layout=tirx.TileLayout.from_iters([Iter(n, stride * 2, "m")], offset={"m": offset + 3})),
    )
    i = ir.Var("i", "int32")
    definition = ir.Bind(view, ir.Call("tirx.decl_tensor", [source.data], ty_args=[view.ty]))
    body = ir.For(i, 0, n, 1, ir.ForKind.DEFAULT, [ir.TensorStore(output, [i], ir.TensorLoad(view, [i]) + 2)])
    function = tirx.Function([source, output], [definition, body], attrs=ir.make_node("ir.DictAttrs", global_symbol="view_copy"))
    assert tirx.analysis.check_well_formed(function)
    result = substitute_primfunc(function, {n: 5, stride: 1, offset: 0})
    declaration = _nodes(result, ir.Bind)[0]
    assert int(tensor_offset(declaration.var)) == 3
    assert [int(x) for x in tensor_strides(declaration.var)] == [2]
    assert tvm_ffi.structural_equal(declaration.var.ty, declaration.value.ty_args[0])
    assert declaration.var.same_as(_nodes(result, ir.TensorLoad)[0].source)
    assert tirx.analysis.check_well_formed(result)
    inp, out = torch.arange(20, dtype=torch.int32), torch.empty(5, dtype=torch.int32)
    compile_packed_function(result, tilelang_owned=False)(inp, out)
    torch.testing.assert_close(out, inp[3:13:2] + 2, rtol=0, atol=0)


def test_same_spelling_does_not_substitute_distinct_identity():
    first, second = ir.Var("n", "int32"), ir.Var("n", "int32")
    tensor = ir.Var("A", tirx.TensorType([first, second], "int32", align=4))
    function = tirx.Function([tensor], [], attrs=ir.make_node("ir.DictAttrs", global_symbol="identity"))
    result = substitute_primfunc(function, {first: 3})
    assert int(result.params[0].ty.shape[0]) == 3
    assert result.params[0].ty.shape[1].same_as(second)
    assert tirx.analysis.check_well_formed(result)


def test_replacements_are_simultaneous():
    first, second = ir.Var("first", "int32"), ir.Var("second", "int32")
    tensor = ir.Var("A", tirx.TensorType([first, second], "int32", align=4))
    function = tirx.Function([tensor], [], attrs=ir.make_node("ir.DictAttrs", global_symbol="identity"))
    result = substitute_primfunc(function, {first: second, second: 2})
    assert result.params[0].ty.shape[0].same_as(second)
    assert int(result.params[0].ty.shape[1]) == 2
    assert tirx.analysis.check_well_formed(result)


@pytest.mark.parametrize("replacement", [7, ir.const(7, "int32")])
def test_substitution_does_not_silently_remove_scalar_parameters(replacement):
    param = ir.Var("n", "int32")
    function = tirx.Function([param], [], attrs=ir.make_node("ir.DictAttrs", global_symbol="identity"))
    with pytest.raises(TypeError, match="parameter.*Var"):
        substitute_primfunc(function, {param: replacement})


def test_local_definition_cannot_be_replaced_by_constant():
    function = _copy(ir.const(4, "int32"))
    loop = _nodes(function, ir.For)[0]
    with pytest.raises(TypeError, match="definition.*Var"):
        substitute_primfunc(function, {loop.var: 2})


def test_replacement_primitive_type_is_checked():
    n = ir.Var("n", "int32")
    function = _copy(n)
    with pytest.raises(TypeError, match="type"):
        substitute_primfunc(function, {n: ir.const(1.5, "float32")})


def test_parameter_dtype_change_is_rejected_without_touching_input():
    function = _copy(ir.const(4, "int32"))
    replacement = ir.Var("A_float", tirx.TensorType([4], "float32", align=4))
    original_hash = tvm_ffi.structural_hash(function)
    with pytest.raises(TypeError, match="type"):
        substitute_primfunc(function, {function.params[0]: replacement})
    assert tvm_ffi.structural_hash(function) == original_hash


def test_repeated_specializations_do_not_share_stale_types():
    n = ir.Var("n", "int32")
    function = _copy(n)
    first = substitute_primfunc(function, {n: 3})
    second = substitute_primfunc(function, {n: 8})
    assert int(first.params[0].ty.shape[0]) == 3
    assert int(second.params[0].ty.shape[0]) == 8
    assert not tvm_ffi.structural_equal(first, second)
    assert tvm_ffi.structural_hash(first) != tvm_ffi.structural_hash(second)
    assert first.params[0].same_as(_nodes(first, ir.TensorLoad)[0].source)
    assert second.params[0].same_as(_nodes(second, ir.TensorLoad)[0].source)


def test_public_eager_rebuild_keeps_constexpr_dynamic_shape_and_output_order():
    @tilelang.jit
    def add(A, increment: int = 4):
        width = T.const("width")
        A: T.Tensor((width,), "int32")
        plus = T.empty((width,), "int32")
        minus = T.empty((width,), "int32")
        for i in T.serial(width):
            plus[i] = A[i] + increment
            minus[i] = A[i] - increment
        return minus, plus

    first = add.get_tir(np.arange(7, dtype="int32"))
    second = add.get_tir(np.arange(11, dtype="int32"), increment=2)
    assert list(first.attrs["tilelang_out_idx"]) == [-1, -2]
    assert all(int(param.ty.shape[0]) == 7 for param in first.params)
    assert all(int(param.ty.shape[0]) == 11 for param in second.params)
    inp = torch.arange(11, dtype=torch.int32)
    plus, minus = torch.empty_like(inp), torch.empty_like(inp)
    compile_packed_function(second, tilelang_owned=False)(inp, plus, minus)
    torch.testing.assert_close(plus, inp + 2, rtol=0, atol=0)
    torch.testing.assert_close(minus, inp - 2, rtol=0, atol=0)

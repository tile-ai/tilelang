"""Address width must not erase explicit conversions or change the public ABI."""

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx
from tvm_tirx.ir import prim

import tilelang.transform as tr
from tilelang.testing.packed import compile_packed_function


def apply(function, width=None):
    config = {} if width is None else {"tl.config_index_bitwidth": width}
    with ir.transform.PassContext(config=config):
        result = tr.ConfigIndexBitwidth()(ir.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(result)
    assert all(old.same_as(new) for old, new in zip(function.params, result.params))
    return result


def source_index(function, source):
    loads = []
    tvm_ffi.structural_walk(function.body, (ir.TensorLoad, loads.append), order="pre")
    return next(load.indices[0] for load in loads if load.source.same_as(source))


def function_with_index(index, variable, size=1 << 40):
    source = ir.Var("source", tirx.TensorType([ir.IntImm("int64", size)], "int32", data_alignment=4))
    output = ir.Var("output", tirx.TensorType([1], "int32", data_alignment=4))
    function = tirx.Function([variable, source, output], [ir.TensorStore(output, [0], ir.TensorLoad(source, [index]))], ir.NoneType())
    return function.with_attr("global_symbol", "main"), source


def test_default_widens_multiplication_before_32_bit_overflow():
    x = ir.Var("x", "int32")
    function, source = function_with_index(x * ir.IntImm("int32", 65536), x)
    index = source_index(apply(function), source)
    assert index.ty == ir.PrimType("int64")
    concrete = tvm_ffi.structural_map(index, (ir.Var, lambda v: ir.IntImm("int32", 32768) if v.same_as(x) else v))
    assert int(sym.Analyzer().simplify(concrete)) == 2147483648


@pytest.mark.parametrize("width", [None, 32, 64])
def test_explicit_narrowing_cast_is_preserved_in_numeric_execution(width):
    x = ir.Var("x", "int32")
    index = prim.Cast("int32", prim.Cast("uint8", x))
    function, source = function_with_index(index, x, size=256)
    lowered = apply(function, width)
    compiled = compile_packed_function(lowered, tilelang_owned=False)
    values = np.arange(256, dtype="int32") * 7 + 3
    output = np.zeros(1, dtype="int32")
    for value in [0, 255, 256, 511, -1]:
        compiled(value, tvm_ffi.from_dlpack(values), tvm_ffi.from_dlpack(output))
        np.testing.assert_array_equal(output, [values[value % 256]])
    assert source_index(lowered, source).ty in (ir.PrimType("int32"), ir.PrimType("int64"))


def test_unsigned_wrap_is_kept_as_an_explicit_arithmetic_value():
    x = ir.Var("x", "uint32")
    original = x + ir.IntImm("uint32", 1)
    function, source = function_with_index(original, x)
    index = source_index(apply(function, 64), source)
    assert tvm_ffi.structural_equal(original, index)


def test_small_bounded_loop_can_keep_32_bit_addresses():
    i = ir.Var("i", "int32")
    source = ir.Var("source", tirx.TensorType([32], "int32"))
    dest = ir.Var("dest", tirx.TensorType([8], "int32"))
    function = tirx.Function(
        [source, dest], [ir.For(i, 0, 8, 1, ir.ForKind.DEFAULT, [ir.TensorStore(dest, [i], ir.TensorLoad(source, [i * 4]))])], ir.NoneType()
    )
    assert source_index(apply(function), source).ty == ir.PrimType("int32")


@pytest.mark.parametrize("width", [8, 16, 0, 128])
def test_invalid_width_is_a_configuration_error(width):
    x = ir.Var("x", "int32")
    function, _ = function_with_index(x, x)
    with pytest.raises(ValueError, match="32 or 64"):
        apply(function, width)

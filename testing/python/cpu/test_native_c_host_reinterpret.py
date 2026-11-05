"""C and C++ hosts preserve scalar bitcasts, including rvalues and lazy uses."""

import ctypes

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import codegen, ir, tirx
from tvm_tirx.target import Target

from tilelang.testing.packed import build_packed_module


def compile_host(function, cxx):
    executable = codegen.Executable(build_packed_module(function))
    executable.jit(cc="c++" if cxx else "cc", options=["-std=c++17" if cxx else "-std=c11"])
    return executable


@pytest.mark.parametrize("cxx", [False, True])
@pytest.mark.parametrize("dtype,word", [("float32", "uint32"), ("float64", "uint64")])
def test_reinterpret_preserves_rvalue_and_loaded_bits(dtype, word, cxx):
    source = ir.Var("source", tirx.TensorType([2], dtype, data_alignment=1))
    output = ir.Var("output", tirx.TensorType([3], word, data_alignment=1))
    expressions = [ir.const(-0.0, dtype), source[0], source[1] + ir.const(1.0, dtype)]
    function = tirx.Function(
        [source, output],
        [ir.TensorStore(output, [index], tirx.reinterpret(value, ty=ir.PrimType(word))) for index, value in enumerate(expressions)],
        ir.PrimType("void"),
    ).with_attr("global_symbol", "main")
    values = np.array([np.inf, 2.5], dtype=dtype)
    actual = np.zeros(3, dtype=word)
    compile_host(function, cxx)(tvm_ffi.from_dlpack(values), tvm_ffi.from_dlpack(actual))
    expected = np.array([-0.0, np.inf, 3.5], dtype=dtype).view(word)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("cxx", [False, True])
def test_reinterpret_does_not_evaluate_an_untaken_lazy_branch(cxx):
    pointer, flag = ir.Var("pointer", ir.PtrType("float32")), ir.Var("flag", "int32")
    tensor = ir.Var("tensor", tirx.TensorType([1], "float32", data_alignment=1))
    value = tirx.reinterpret(tensor[0], ty=ir.PrimType("uint32"))
    result = ir.Call("prim.if_then_else", [flag != 0, value, ir.const(42, "uint32")], ty=ir.PrimType("uint32"))
    function = tirx.Function(
        [pointer, flag],
        [ir.Bind(tensor, ir.Call("tirx.decl_tensor", [pointer], ty_args=[tensor.ty])), ir.Return(result)],
        ir.PrimType("uint32"),
    ).with_attr("global_symbol", "main")
    executable = compile_host(function, cxx)
    assert executable(ctypes.c_void_p(), 0) == 42
    storage = ctypes.c_float(-0.0)
    assert executable(ctypes.cast(ctypes.pointer(storage), ctypes.c_void_p), 1) == 0x80000000


@pytest.mark.parametrize("source,target", [("int1", "uint1"), ("uint1", "int1")])
def test_subbyte_bitcast_requires_storage_legalization(source, target):
    value = ir.Var("value", source)
    cast = tirx.reinterpret(value, ty=ir.PrimType(target))
    function = tirx.Function([value], [ir.Return(cast)], ir.PrimType(target)).with_attr("global_symbol", "main")
    assert tirx.analysis.check_well_formed(function)
    with pytest.raises(TypeError, match="requires byte-sized primitive storage"):
        tvm_ffi.get_global_func("target.build.tilelang_c_host")(ir.IRModule({"main": function}), Target("c"))

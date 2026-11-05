"""Native statement simplification preserves signature and type definitions."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

import tilelang.language as T
from tilelang.transform.simplify import Simplify, _Simplify
from tilelang.testing.packed import compile_packed_function


NATIVE_TARGETS = [
    "c",
    pytest.param("llvm", marks=pytest.mark.skipif(not tvm_tirx.runtime.enabled("llvm"), reason="LLVM support is not built")),
]


def _compile_semantics(function, target):
    if target == "c":
        return compile_packed_function(function, tilelang_owned=False)
    return tirx.compile(function, target="llvm")


def test_simplify_uses_native_rules_without_dropping_public_arguments():
    @T.prim_func
    def kernel(unused: T.Tensor((4,), "int32"), output: T.Tensor((4,), "int32")):
        for i in T.serial(4):
            output[i] = i // 4

    result = Simplify()(ir.IRModule.from_expr(kernel))["kernel"]
    assert len(result.params) == 2
    stores = []
    tvm_ffi.structural_walk(result, (ir.TensorStore, stores.append), order="pre")
    assert len(stores) == 1
    assert isinstance(stores[0].value, ir.IntImm) and stores[0].value.value == 0
    assert tirx.analysis.check_well_formed(result)


@pytest.mark.parametrize("target", NATIVE_TARGETS)
def test_explicit_argument_cleanup_drops_only_dead_static_parameters(target):
    @T.prim_func
    def kernel(unused: T.Tensor((4,), "int32"), output: T.Tensor((4,), "int32")):
        for i in T.serial(4):
            output[i] = 9

    result = _Simplify(kernel)
    assert len(result.params) == 1
    assert result.params[0].name == "output"
    actual = tvm_tirx.runtime.tensor(np.zeros(4, dtype="int32"))
    _compile_semantics(result, target)(actual)
    np.testing.assert_array_equal(actual.numpy(), np.full(4, 9, dtype="int32"))


@pytest.mark.parametrize("target", NATIVE_TARGETS)
def test_cleanup_keeps_parameter_that_defines_a_live_shape(target):
    n = T.dynamic("n")

    @T.prim_func
    def kernel(shape_owner: T.Tensor((n,), "int32"), output: T.Tensor((1,), "int32")):
        output[0] = n

    result = _Simplify(kernel)
    assert len(result.params) == 2
    assert result.params[0].name == "shape_owner"
    assert tirx.analysis.check_well_formed(result)
    owner = tvm_tirx.runtime.tensor(np.zeros(13, dtype="int32"))
    actual = tvm_tirx.runtime.tensor(np.zeros(1, dtype="int32"))
    _compile_semantics(result, target)(owner, actual)
    np.testing.assert_array_equal(actual.numpy(), np.array([13], dtype="int32"))

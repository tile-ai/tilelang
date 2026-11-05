"""Finiteness stays selectable as a TileLang intrinsic with native types."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

import tilelang.language as T


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
@pytest.mark.parametrize("lanes", [1, 4])
def test_isfinite_producer_retains_intrinsic_type_and_location(dtype, lanes):
    primitive = ir.PrimType(tvm_ffi.dtype(dtype).with_lanes(lanes))
    variable = ir.Var("x", primitive)
    location = ir.SourceLoc(ir.SourceName("isfinite-input.py"), 7, 3, 7, 12)
    value = T.isfinite(variable, loc=location)
    assert isinstance(value, ir.Call)
    assert value.op.name == "tl.isfinite"
    assert value.ty == ir.PrimType(tvm_ffi.dtype("bool").with_lanes(lanes))
    assert value.loc.same_as(location)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_isfinite_portable_lowering_matches_numpy(dtype):
    values = np.array([0.0, -0.0, 3.5, -7.25, np.inf, -np.inf, np.nan], dtype=dtype)

    @T.prim_func
    def kernel(A: T.Tensor((7,), dtype), B: T.Tensor((7,), "int32")):
        for i in T.serial(7):
            B[i] = T.if_then_else(T.isfinite(A[i]), 1, 0)

    output = tvm_tirx.runtime.tensor(np.empty(7, dtype="int32"))
    tirx.compile(kernel, target="c")(tvm_tirx.runtime.tensor(values), output)
    np.testing.assert_array_equal(output.numpy(), np.isfinite(values).astype("int32"))


def test_isfinite_integer_preserves_native_result():
    from tvm_tirx.ir import prim

    variable = ir.Var("x", ir.PrimType("int32"))
    assert tvm_ffi.structural_equal(T.isfinite(variable), prim.isfinite(variable))

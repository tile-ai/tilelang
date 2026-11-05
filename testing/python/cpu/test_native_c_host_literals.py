"""The C host emits reducer infinities and keeps IEEE scalar literal values."""

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import codegen, ir, tirx

from tilelang.testing.packed import build_packed_module


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("compiler,standard", [("cc", "c11"), ("c++", "c++17")])
def test_host_float_literals_execute_in_c_and_cpp(dtype, compiler, standard):
    output = ir.Var("output", tirx.TensorType([5], dtype, align=8))
    values = [float("inf"), -float("inf"), float("nan"), -0.0, 1.25]
    function = tirx.Function(
        [output],
        [ir.TensorStore(output, [index], ir.FloatImm(dtype, value)) for index, value in enumerate(values)],
    ).with_attr("global_symbol", "main")
    original = ir.save_json(function)
    executable = codegen.Executable(build_packed_module(function))
    executable.jit(cc=compiler, options=[f"-std={standard}"])
    result = np.zeros(5, dtype=dtype)
    executable(tvm_ffi.from_dlpack(result))
    assert np.isposinf(result[0]) and np.isneginf(result[1])
    assert np.isnan(result[2])
    assert result[3] == 0 and np.signbit(result[3])
    assert result[4] == 1.25
    assert ir.save_json(function) == original

import numpy as np
import pytest

from tvm_tirx import ir, runtime, tirx


@pytest.mark.parametrize("value", [0, 37, -1, -2147483648])
def test_reinterpret_constant_materializes_addressable_value(value):
    output = ir.Var("output", tirx.TensorType([1], "uint32"))
    result = tirx.reinterpret(ir.const(value, "int32"), ty="uint32")
    function = tirx.Function([output], ir.TensorStore(output, [ir.const(0)], result)).with_attr("global_symbol", "reinterpret_constant")
    executable = tirx.compile(function, target="c")
    actual = runtime.empty((1,), "uint32")
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), np.array([value], dtype="int32").view("uint32"))

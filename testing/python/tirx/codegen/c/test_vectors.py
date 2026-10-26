import numpy as np
import pytest

from tvm_tirx import ir, runtime, tirx
from tvm_tirx.ir import prim


@pytest.mark.parametrize("lanes", [2, 4, 8, 16])
@pytest.mark.parametrize(
    "source_dtype,destination_dtype",
    [
        ("int32", "float32"),
        ("float32", "int32"),
        ("float32", "float64"),
        ("int32", "int8"),
        ("int32", "bool"),
        ("float32", "float16"),
    ],
)
def test_vector_casts_convert_each_lane(lanes, source_dtype, destination_dtype):
    source = ir.Var("source", tirx.TensorType((lanes + 2,), source_dtype))
    output = ir.Var("output", tirx.TensorType((lanes + 2,), destination_dtype))
    positions = prim.Ramp(ir.const(1), ir.const(1), lanes)
    value = prim.Cast(f"{destination_dtype}x{lanes}", ir.TensorLoad(source, [positions]))
    function = tirx.Function([source, output], ir.TensorStore(output, [positions], value))
    executable = tirx.compile(function.with_attr("global_symbol", "vector_cast"), target="c")
    values = np.linspace(-3, 4, lanes + 2).astype(source_dtype)
    actual = runtime.tensor(np.zeros((lanes + 2,), dtype=destination_dtype))
    executable(runtime.tensor(values), actual)
    expected = np.zeros((lanes + 2,), dtype=destination_dtype)
    expected[1:-1] = values[1:-1].astype(destination_dtype)
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("lanes", [2, 4, 8, 16])
def test_vector_arithmetic_broadcast_min_max(lanes):
    source = ir.Var("source", tirx.TensorType((lanes,), "float32"))
    output = ir.Var("output", tirx.TensorType((lanes,), "float32"))
    positions = prim.Ramp(ir.const(0), ir.const(1), lanes)
    values = ir.TensorLoad(source, [positions])
    result = prim.Max(
        prim.Min(
            values * prim.Broadcast(ir.const(2.5, "float32"), lanes),
            prim.Broadcast(ir.const(4.0, "float32"), lanes),
        ),
        prim.Broadcast(ir.const(-2.0, "float32"), lanes),
    )
    function = tirx.Function([source, output], ir.TensorStore(output, [positions], result))
    executable = tirx.compile(function.with_attr("global_symbol", "vector_arithmetic"), target="c")
    values = np.linspace(-3, 4, lanes).astype("float32")
    actual = runtime.empty((lanes,), "float32")
    executable(runtime.tensor(values), actual)
    np.testing.assert_array_equal(actual.numpy(), np.clip(values * 2.5, -2.0, 4.0))

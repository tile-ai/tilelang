import numpy as np
import pytest

from tvm_tirx import ir, runtime, tirx
from tvm_tirx.ir import prim
from tvm_tirx.target import Target


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2", "float4_e2m1fn"])
@pytest.mark.parametrize("lanes", [1, 2, 8, 16])
@pytest.mark.parametrize("operation", ["equal", "not_equal", "less", "less_equal", "greater", "greater_equal"])
def test_low_precision_comparison_is_numeric(dtype, lanes, operation):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device unavailable")
    architecture = int(device.compute_version.replace(".", ""))
    if architecture < (100 if dtype.startswith("float4") else 89):
        pytest.skip("CUDA device does not support this low-precision type")
    target = Target(
        {
            "kind": "cuda",
            "arch": f"sm_{architecture}",
            "backend_config": {"cuda": {"compiler": "nvcc"}},
        },
        host="c",
    )
    left = ir.Var("left", tirx.TensorType([32], "float32"))
    right = ir.Var("right", tirx.TensorType([32], "float32"))
    output = ir.Var("output", tirx.TensorType([32], "bool"))
    thread = ir.Var("thread", "int32")
    indices = [thread if lanes == 1 else prim.Ramp(thread * lanes, 1, lanes)]
    operand_type = dtype if lanes == 1 else f"{dtype}x{lanes}"
    left_value = prim.Cast(operand_type, ir.TensorLoad(left, indices))
    right_value = prim.Cast(operand_type, ir.TensorLoad(right, indices))
    comparison = {
        "equal": prim.EQ,
        "not_equal": prim.NE,
        "less": prim.LT,
        "less_equal": prim.LE,
        "greater": prim.GT,
        "greater_equal": prim.GE,
    }[operation](left_value, right_value)
    body = ir.RegionStmt(
        "tirx.launch_thread",
        [ir.StringImm("threadIdx.x"), ir.const(32 // lanes)],
        [thread],
        None,
        ir.TensorStore(output, indices, comparison),
    )
    function = tirx.Function([left, right, output], body).with_attr("global_symbol", "low_precision_comparison")
    executable = tirx.compile(function, target=target)
    left_array = np.tile(np.array([-4, -2, -1, -0.0, 0, 0.5, 2, 4], dtype="float32"), 4)
    right_array = np.repeat(np.array([-2, -0.0, 0.5, 4], dtype="float32"), 8)
    if dtype.startswith("float8"):
        left_array[0] = np.nan
        right_array[-1] = np.nan
    expected = getattr(np, operation)(left_array, right_array)
    actual = runtime.empty((32,), "bool", device=device)
    executable(
        runtime.tensor(left_array, device=device),
        runtime.tensor(right_array, device=device),
        actual,
    )
    np.testing.assert_array_equal(actual.numpy(), expected)

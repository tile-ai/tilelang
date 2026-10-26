import numpy as np
import pytest

from tvm_tirx import ir, runtime, tirx
from tvm_tirx.ir import prim
from tvm_tirx.target import Target


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("lanes", [2, 8, 12, 16])
@pytest.mark.parametrize(
    "operation",
    [
        "add",
        "subtract",
        "multiply",
        "minimum",
        "maximum",
        "fused_multiply_add",
        "fused_add_multiply",
    ],
)
def test_packed_math_preserves_each_lane(dtype, lanes, operation):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device unavailable")
    architecture = int(device.compute_version.replace(".", ""))
    if dtype == "bfloat16" and architecture < 80:
        pytest.skip("bfloat16 arithmetic requires SM80")
    target = Target(
        {
            "kind": "cuda",
            "arch": f"sm_{architecture}",
            "backend_config": {"cuda": {"compiler": "nvcc"}},
        },
        host="c",
    )
    left = ir.Var("left", tirx.TensorType([lanes], "float32"))
    right = ir.Var("right", tirx.TensorType([lanes], "float32"))
    output = ir.Var("output", tirx.TensorType([lanes], "float32"))
    indices = [prim.Ramp(0, 1, lanes)]
    left_value = prim.Cast(f"{dtype}x{lanes}", ir.TensorLoad(left, indices))
    right_value = prim.Cast(f"{dtype}x{lanes}", ir.TensorLoad(right, indices))
    value = {
        "add": lambda: left_value + right_value,
        "subtract": lambda: left_value - right_value,
        "multiply": lambda: left_value * right_value,
        "minimum": lambda: prim.Min(left_value, right_value),
        "maximum": lambda: prim.Max(left_value, right_value),
        "fused_multiply_add": lambda: left_value * right_value + left_value,
        "fused_add_multiply": lambda: left_value + left_value * right_value,
    }[operation]()
    body = ir.RegionStmt(
        "tirx.launch_thread",
        [ir.StringImm("threadIdx.x"), ir.const(1)],
        [ir.Var("thread", "int32")],
        None,
        ir.TensorStore(output, indices, prim.Cast(f"float32x{lanes}", value)),
    )
    function = tirx.Function([left, right, output], body).with_attr("global_symbol", "packed_math")
    executable = tirx.compile(function, target=target)
    left_array = np.linspace(-3.25, 4.75, lanes, dtype="float32")
    right_array = np.linspace(1.75, -2.25, lanes, dtype="float32")
    storage_dtype = getattr(pytest.importorskip("ml_dtypes"), dtype) if dtype == "bfloat16" else dtype
    rounded_left = left_array.astype(storage_dtype).astype("float32")
    rounded_right = right_array.astype(storage_dtype).astype("float32")
    if operation.startswith("fused_"):
        expected = rounded_left.astype("float64") * rounded_right + rounded_left
    else:
        expected = getattr(np, operation)(rounded_left, rounded_right)
    expected = expected.astype(storage_dtype).astype("float32")
    actual = runtime.empty((lanes,), "float32", device=device)
    executable(
        runtime.tensor(left_array, device=device),
        runtime.tensor(right_array, device=device),
        actual,
    )
    np.testing.assert_array_equal(actual.numpy(), expected)

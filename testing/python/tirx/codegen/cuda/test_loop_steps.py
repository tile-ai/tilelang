import numpy as np
import pytest

from tvm_tirx import ir, runtime, tirx
from tvm_tirx.target import Target


@pytest.mark.gpu
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("bounds", [(1, 11, 3), (9, -1, -3), (3, 7, -1), (8, 2, 1)])
def test_cuda_loop_step_respects_direction(dynamic, bounds):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device unavailable")
    output = ir.Var("output", tirx.TensorType([16], "int32"))
    thread = ir.Var("thread", "int32")
    index = ir.Var("index", "int32")
    parameters = [output]
    loop_bounds = bounds
    if dynamic:
        loop_bounds = [ir.Var(name, "int32") for name in ("begin", "end", "step")]
        parameters.extend(loop_bounds)
    loop = ir.For(index, *loop_bounds, ir.ForKind.DEFAULT, ir.TensorStore(output, [index], index + 1))
    body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(1)], [thread], None, loop)
    function = tirx.Function(parameters, body, ret_type=ir.TupleType([])).with_attr("global_symbol", "loop_steps")
    target = Target(
        {
            "kind": "cuda",
            "arch": "sm_" + device.compute_version.replace(".", ""),
            "backend_config": {"cuda": {"compiler": "nvcc"}},
        },
        host="c",
    )
    executable = tirx.compile(function, target=target)
    actual = runtime.tensor(np.full(16, -99, dtype="int32"), device=device)
    executable(actual, *bounds) if dynamic else executable(actual)
    expected = np.full(16, -99, dtype="int32")
    for iteration in range(*bounds):
        expected[iteration] = iteration + 1
    np.testing.assert_array_equal(actual.numpy(), expected)

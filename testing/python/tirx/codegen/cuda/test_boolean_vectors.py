import numpy as np
import pytest

from tvm_tirx import ir, runtime, tirx
from tvm_tirx.ir import prim
from tvm_tirx.target import Target


@pytest.mark.gpu
@pytest.mark.parametrize("lanes", [2, 3, 4, 8, 16, 32])
def test_boolean_vector_storage_and_selection(lanes):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device unavailable")
    target = Target(
        {
            "kind": "cuda",
            "arch": f"sm_{device.compute_version.replace('.', '')}",
            "backend_config": {"cuda": {"compiler": "nvcc"}},
        },
        host="c",
    )
    source = ir.Var("source", tirx.TensorType([lanes], "float32"))
    flags = ir.Var("flags", tirx.TensorType([lanes], "bool"))
    output = ir.Var("output", tirx.TensorType([lanes], "float32"))
    indices = [prim.Ramp(0, 1, lanes)]
    values = ir.TensorLoad(source, indices)
    zero = prim.Broadcast(ir.const(0, "float32"), lanes)
    one = prim.Broadcast(ir.const(1, "float32"), lanes)
    stores = ir.SeqStmt(
        [
            ir.TensorStore(flags, indices, values > zero),
            ir.TensorStore(
                output,
                indices,
                prim.Select(prim.Not(ir.TensorLoad(flags, indices)), values + one, values - one),
            ),
        ]
    )
    body = ir.RegionStmt(
        "tirx.launch_thread",
        [ir.StringImm("threadIdx.x"), ir.const(1)],
        [ir.Var("thread", "int32")],
        None,
        stores,
    )
    function = tirx.Function([source, flags, output], body).with_attr("global_symbol", "boolean_vectors")
    executable = tirx.compile(function, target=target)
    values = np.arange(lanes, dtype="float32") - lanes // 2
    actual_flags = runtime.empty((lanes,), "bool", device=device)
    actual = runtime.empty((lanes,), "float32", device=device)
    executable(runtime.tensor(values, device=device), actual_flags, actual)
    np.testing.assert_array_equal(actual_flags.numpy(), values > 0)
    np.testing.assert_array_equal(actual.numpy(), np.where(values <= 0, values + 1, values - 1))

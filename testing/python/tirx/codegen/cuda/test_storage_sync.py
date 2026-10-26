import numpy as np
import pytest

from tvm_tirx import ir, runtime, tirx
from tvm_tirx.target import Target
from tilelang._compat.intrinsics import gpu_storage_sync


@pytest.mark.gpu
@pytest.mark.parametrize("scope,participants", [("shared", 32), ("shared", 64), ("warp", 32), ("cluster", 128)])
def test_partial_storage_sync_preserves_its_participants(scope, participants, compile_tilelang):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device unavailable")
    if scope == "cluster" and int(device.compute_version.split(".")[0]) < 9:
        pytest.skip("Cluster barriers require SM90 or newer")
    thread = ir.Var("thread", "int32")
    shared = ir.Var("shared", tirx.TensorType([128], "int32", scope="shared"))
    output = ir.Var("output", tirx.TensorType([128], "int32"))
    operands = []
    if scope == "shared":
        operands.extend([ir.const(3), ir.const(participants)])
    synchronize = gpu_storage_sync(scope, *operands)
    synchronize.validate()
    body = [
        ir.Bind(shared, ir.Call("tirx.alloc_tensor", [], ty_args=[shared.ty])),
        ir.If(
            thread < participants,
            [
                ir.TensorStore(shared, [thread], thread),
                ir.Evaluate(synchronize),
                ir.TensorStore(output, [thread], ir.TensorLoad(shared, [thread ^ 1])),
            ],
            ir.TensorStore(output, [thread], ir.const(-1)),
        ),
    ]
    launch = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(128)], [thread], None, body)
    function = tirx.Function([output], launch).with_attr("global_symbol", "partial_sync")
    target = Target(
        {
            "kind": "cuda",
            "arch": f"sm_{device.compute_version.replace('.', '')}",
            "backend_config": {"cuda": {"compiler": "nvcc"}},
        },
        host="c",
    )
    compiled = compile_tilelang(function, target)
    source = compiled.mod.imports[0].inspect_source()
    expected_sync = {
        "shared": "bar.sync",
        "warp": "__syncwarp",
        "cluster": "barrier.cluster.wait.aligned",
    }
    assert expected_sync[scope] in source
    assert "__syncthreads" not in source
    actual = runtime.empty((128,), "int32", device=device)
    compiled(actual)
    expected = np.full(128, -1, dtype="int32")
    expected[:participants] = np.arange(participants) ^ 1
    np.testing.assert_array_equal(actual.numpy(), expected)

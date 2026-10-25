import numpy as np
import pytest
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.target import Target

from tilelang._compat.tensor import TensorView
from testing.python.tirx.test_tirx_thread_sync import barriers, launch, lower

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def _delayed_exchange(bounds, cross_thread):
    shared = TensorView.create("shared", [4, 128], "int32", scope="shared")
    output = TensorView.create("output", [4, 128], "int32")
    thread = ir.Var("thread", "int32")
    time = ir.Var("time", "int32")
    begin, end, step = bounds
    ordinal = (time - begin) // step
    source_thread = (thread + 64) % 128 if cross_thread else thread
    loop = ir.For(
        time,
        begin,
        end,
        step,
        ir.ForKind.DEFAULT,
        [
            ir.If(ordinal < 4, shared.store([ordinal, thread], ordinal * 128 + thread), None),
            ir.If(ordinal >= 3, output.store([ordinal - 3, thread], shared.load([ordinal - 3, source_thread])), None),
        ],
    )
    return lower(launch(thread, [shared.allocate(), loop]), [output.tensor])


@pytest.mark.parametrize("bounds", [(0, 7, 1), (2, 23, 3), (12, -2, -2)])
@pytest.mark.parametrize("cross_thread", [False, True])
def test_signed_loops_detect_nonadjacent_carried_conflicts(bounds, cross_thread):
    assert bool(barriers(_delayed_exchange(bounds, cross_thread))) is cross_thread


@pytest.mark.parametrize("bounds", [(0, 7, 1), (12, -2, -2)])
def test_distant_loop_carried_exchange_executes(bounds):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device unavailable")
    module = _delayed_exchange(bounds, True)
    target = Target(
        {"kind": "cuda", "arch": f"sm_{device.compute_version.replace('.', '')}", "backend_config": {"cuda": {"compiler": "nvcc"}}},
        host="c",
    )
    executable = tirx.compile(module, target=target)
    output = runtime.empty((4, 128), "int32", device=device)
    executable(output)
    expected = np.arange(4, dtype="int32")[:, None] * 128 + (np.arange(128, dtype="int32") + 64) % 128
    np.testing.assert_array_equal(output.numpy(), expected)

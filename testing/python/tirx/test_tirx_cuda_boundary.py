import numpy as np
import pytest
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor


@pytest.mark.gpu
def test_rewritten_native_kernel_runs_on_cuda():
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("A TIRx CUDA runtime and CUDA device are required")

    @T.function
    def add_one(source: T.Tensor((256,), "float32"), destination: T.Tensor((256,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=128))
        index = T.cuda.block_idx("x") * 128 + T.cuda.thread_idx("x")
        destination[index] = source[index] + T.float32(1)

    def rewrite(node):
        if isinstance(node, ir.FloatImm) and float(node) == 1.0:
            return ir.FloatImm(node.ty, 2.0)
        return None

    rewritten = stmt_functor.ir_transform(add_one, None, rewrite)
    version = device.compute_version.replace(".", "")
    target = Target({"kind": "cuda", "arch": f"sm_{version}", "backend_config": {"cuda": {"compiler": "nvcc"}}}, host="c")
    executable = tirx.compile(rewritten, target=target)
    input_array = np.arange(256, dtype="float32")
    source = runtime.tensor(input_array, device=device)
    actual = runtime.empty((256,), "float32", device=device)
    executable(source, actual)
    np.testing.assert_array_equal(actual.numpy(), input_array + 2.0)

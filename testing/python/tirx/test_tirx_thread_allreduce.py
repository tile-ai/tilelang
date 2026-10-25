import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.script import tirx as script
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor


@pytest.mark.parametrize("width", [12, 32, 48, 64, 100, 201])
@pytest.mark.parametrize("operation", ["sum", "max"])
def test_native_thread_allreduce_delegation_executes(width, operation, tilelang_builders):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device unavailable")
    target = Target(
        {"kind": "cuda", "arch": f"sm_{device.compute_version.replace('.', '')}", "backend_config": {"cuda": {"compiler": "nvcc"}}},
        host="c",
    )
    combine = script.Lambda(
        [script.float32, script.float32],
        (lambda left, right: (left + right,)) if operation == "sum" else (lambda left, right: (script.max(left, right),)),
    )
    initial = 0.0 if operation == "sum" else -np.finfo("float32").max

    @script.function
    def allreduce(source: script.Tensor((2, width), "float32"), output: script.Tensor((2,), "float32")):
        with script.launch_thread("threadIdx.y", 2) as group, script.launch_thread("threadIdx.x", width) as thread:
            reduced = script.alloc_tensor(ty_args=[script.Tensor((1,), "float32", scope="local")])
            script.gpu_thread_allreduce(combine, (script.float32(initial),), (source[group, thread],), True, (reduced[0],), (thread,))
            if thread == 0:
                output[group] = reduced[0]

    module = ir.IRModule({"main": allreduce.with_attr("target", target)})
    with target:
        module = tvm_ffi.get_global_func("tl.transform.LowerThreadAllreduce")()(module)
    assert tirx.analysis.check_well_formed(module)
    remaining = []
    stmt_functor.post_order_visit(
        module,
        lambda node: remaining.append(node)
        if isinstance(node, ir.Call) and isinstance(node.op, ir.Op) and node.op.name == "tirx.gpu_thread_allreduce"
        else None,
    )
    assert not remaining
    executable = tirx.compile(module, target=target)
    values = ((np.arange(2 * width).reshape(2, width) % 23) - 11).astype("float32")
    actual = runtime.empty((2,), "float32", device=device)
    executable(runtime.tensor(values, device=device), actual)
    expected = values.sum(axis=1) if operation == "sum" else values.max(axis=1)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-6)

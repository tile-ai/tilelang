import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, sym, tirx
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def collect(root, kind):
    result = []
    stmt_functor.post_order_visit(root, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


def lower(function, target=None):
    target = target or Target({"kind": "cuda", "arch": "sm_90"}, host="c")
    module = tirx.transform.BindTarget(target)(ir.IRModule({"main": function}))
    module = tvm_ffi.get_global_func("tl.cuda.transform.LowerL2Persistent")()(module)
    assert tirx.analysis.check_well_formed(module)
    ir.assert_structural_equal(module, ir.load_json(ir.save_json(module)))
    return module


def kernel(source, output, ratio=0.75):
    thread = ir.Var("thread", "int32")
    scope = ir.RegionStmt(
        "tirx.scope",
        [],
        [],
        {"l2_hit_ratio_map": {source.tensor: ir.const(ratio, "float32")}},
        output.store(thread, source.load(thread) + 1),
    )
    body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(16)], [thread], None, scope)
    return tirx.Function([source.tensor, output.tensor], body).with_attr("global_symbol", "l2_test")


def split(module):
    module = tvm_ffi.get_global_func("tl.transform.SplitHostDevice")()(module)
    assert tirx.analysis.check_well_formed(module)
    ir.assert_structural_equal(module, ir.load_json(ir.save_json(module)))
    assert not any(isinstance(call.op, ir.Op) and call.op.name == "tl.l2_persistent_hint" for call in collect(module, ir.Call))
    return module


def test_l2_metadata_remains_lexical_then_moves_to_host_calls():
    source = TensorView.create("source", [16], "float32")
    output = TensorView.create("output", [16], "float32")
    module = lower(kernel(source, output))
    assert "l2_persistent_map" not in module["main"].attrs
    hints = [call for call in collect(module, ir.Call) if isinstance(call.op, ir.Op) and call.op.name == "tl.l2_persistent_hint"]
    assert len(hints) == 1
    assert hints[0].args[0].same_as(source.tensor)
    module = split(module)
    calls = [call for call in collect(module["main"], ir.Call) if isinstance(call.op, ir.Op) and call.op.name == "tirx.call_packed"]
    names = [str(call.args[0].value) for call in calls]
    assert names == [
        "runtime.get_cuda_stream",
        "tl.cuda.stream_set_access_policy_window",
        "tl.cuda.stream_reset_access_policy_window",
    ]
    assert int(sym.Analyzer().simplify(calls[1].args[2])) == 64


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_l2_hit_ratios_are_rejected(ratio):
    source = TensorView.create("source", [16], "float32")
    output = TensorView.create("output", [16], "float32")
    with pytest.raises(ValueError, match="finite constant"):
        lower(kernel(source, output, ratio))


def test_conditional_l2_configuration_is_not_hoisted():
    source = TensorView.create("source", [16], "float32")
    output = TensorView.create("output", [16], "float32")
    function = kernel(source, output)
    launch = function.body.seq[0]
    branch = ir.If(launch.body_params[0] == 0, launch.body, None)
    body = ir.RegionStmt(launch.op, launch.args, launch.body_params, launch.attrs, branch)
    function = tirx.Function(function.params, body).with_attr("global_symbol", "conditional_l2")
    with pytest.raises(ValueError, match="unconditionally"):
        split(lower(function))


@pytest.mark.gpu
def test_l2_configured_native_kernel_executes():
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device unavailable")
    source = TensorView.create("source", [16], "float32")
    output = TensorView.create("output", [16], "float32")
    target = Target(
        {"kind": "cuda", "arch": f"sm_{device.compute_version.replace('.', '')}", "backend_config": {"cuda": {"compiler": "nvcc"}}},
        host="c",
    )
    module = split(lower(kernel(source, output), target))
    executable = tirx.compile(module, target=target)
    expected = np.arange(16, dtype="float32")
    actual = runtime.empty((16,), "float32", device=device)
    executable(runtime.tensor(expected, device=device), actual)
    np.testing.assert_array_equal(actual.numpy(), expected + 1)

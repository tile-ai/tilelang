import numpy as np
import pytest

from tvm_tirx import ir, runtime, tirx
from tvm_tirx.target import Target


@pytest.mark.gpu
@pytest.mark.parametrize("message", [[], ['bad "index" 100%'], ["first", "second"]])
def test_device_assertion_uses_device_error_path(message):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device unavailable")
    output = ir.Var("output", tirx.TensorType([1], "int32"))
    value = ir.Var("value", "int32")
    body = [
        ir.AssertStmt(value >= 0, ir.StringImm("AssertionError"), [ir.StringImm(part) for part in message]),
        ir.TensorStore(output, [ir.const(0)], value),
    ]
    thread = ir.Var("thread", "int32")
    launch = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(1)], [thread], None, body)
    function = tirx.Function([output, value], launch).with_attr("global_symbol", "check_value")
    target = Target({"kind": "cuda", "arch": f"sm_{device.compute_version.replace('.', '')}"}, host="c")
    compiled = tirx.compile(function, target=target)
    source = compiled.mod.imports[0].inspect_source()
    assert 'asm volatile("trap;")' in source
    assert "TVMFFIErrorSetRaised" not in source
    assert "return -1" not in source
    actual = runtime.empty((1,), "int32", device=device)
    compiled(actual, 37)
    np.testing.assert_array_equal(actual.numpy(), [37])
